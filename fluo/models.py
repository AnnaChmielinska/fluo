#!/usr/bin/env python3
from collections import OrderedDict
from abc import ABCMeta, abstractmethod
import numpy as np
import scipy
import scipy.interpolate
from lmfit import Parameters, Model

class AbstractModel(): 
    __metaclass__ = ABCMeta
    
    def __init__(self, user_kwargs=None):
        self.user_kwargs = user_kwargs

    def make_model(self, **independent_var): 
        return GenericModel(
    self.model_function(**independent_var), 
    missing='drop', 
    name=self.__class__.__name__
    )

    @abstractmethod
    def model_function(self, **independent_var):
        raise NotImplementedError()

    @abstractmethod
    def make_parameters(self):
        raise NotImplementedError()

class GlobalModel(AbstractModel):
    
    def __init__(self, FitterClasses, shared=None):
        self.FitterClasses = FitterClasses
        self.shared = shared
        if self.shared is None:
            self.shared = []   
        self.local_ModelClass = self.make_local_atrribute(FitterClasses, 'ModelClass')
        self.local_independent_var = self.make_local_atrribute(FitterClasses, 'independent_var')
        self.local_dependent_var = self.make_local_atrribute(FitterClasses, 'dependent_var')
        self.local_indexes = self.make_local_indexes(self.local_dependent_var)
        self.statistic = FitterClasses[0].statistic # statistic in every Fitter class should be the same
        self.local_parameters = self.make_local_atrribute(FitterClasses, 'parameters')
        self._parameters, self.parameters_references = self.glue_parameters()

    @staticmethod
    def make_local_atrribute(fitters, atrr):
        return [getattr(fitter, atrr) for fitter in fitters]

    @staticmethod
    def make_local_indexes(arrs):
        return np.cumsum([len(arr) for arr in arrs])[:-1]

    def model_function(self, **independent_var):
        return self.global_eval(**independent_var)

    def make_parameters(self):
        return self._parameters

    def global_eval(self, independent_var):
        def inner_global_eval(**params):
            for name, value in params.items():
                fitter_i, local_name = self.parameters_references[name]
                self.local_parameters[fitter_i][local_name].value = value
            global_eval = []
            for i, local_fitter in enumerate(self.FitterClasses):
                model_i = local_fitter.ModelClass.make_model(**independent_var[i])
                local_eval = model_i.eval(**self.local_parameters[i])
                global_eval.append(local_eval)
            return np.concatenate(global_eval)
        return inner_global_eval

    def glue_parameters(self):
        parameters_references = dict()
        all_params = Parameters()
        for i, params_i in enumerate(self.local_parameters):
            for old_name, param in params_i.items():
                new_name = old_name + '_file%d' % (i + 1)
                parameters_references[new_name] = (i, param.name)
                all_params.add(new_name,
                               value=param.value,
                               vary=param.vary,
                               min=param.min,
                               max=param.max,
                               expr=param.expr)
        for param_name, param in all_params.items():
            for constraint in self.shared:
                if param_name.startswith(constraint) and not param_name.endswith('_file1'):
                    param.expr = constraint + '_file1'
        return all_params, parameters_references

class ConvolvedExponential(AbstractModel):

    def model_function(self, **independent_var):
        return self.convolved_exponential(**independent_var)

    def make_parameters(self):
        nonlinear_pars = Exponential(self.user_kwargs).make_parameters()
        if self.user_kwargs['--shift'] is None:
            nonlinear_pars.add(
                'shift', 
                value=0., 
                vary=True)
        else:
            nonlinear_pars.add(
                'shift', 
                value=getValue(self.user_kwargs['--shift']), 
                vary=doesVary(self.user_kwargs['--shift']))
        return nonlinear_pars

    def convolved_exponential(self, time, instrument_response):
        def inner_convolved_exponential(**params):
            taus = {key: value for key, value in params.items() if
                    key.startswith('tau')}
            exp = Exponential.exponential(time)(**taus)
            conv_exp = np.zeros(exp.shape)
            for i in range(exp.shape[1]):
                shifted = self.shift_decay(time, instrument_response, params['shift'])
                conv_exp[:, i] = self.convolve(shifted, exp[:, i])
            return conv_exp
        return inner_convolved_exponential

    @staticmethod
    def shift_decay(time, intensity, shift):
        """
        Shift decay in time x-axis.
        """
        s = scipy.interpolate.interp1d(
            time,
            intensity,
            kind='slinear',
            bounds_error=False,
            fill_value=0.0)
        return s(time + shift)

    @staticmethod
    def convolve(left, right):
        return np.convolve(left, right, mode='full')[:len(right)]

class Exponential(AbstractModel):

    def model_function(self, **independent_var):
        return self.exponential(**independent_var)
    
    def make_parameters(self):
        fit_components  = int(self.user_kwargs['<lt-fit_components>'])
        nonlinear_pars = Parameters()
        for i, tau in zip(range(fit_components), self.user_kwargs['--tau']):
            if tau is None:
                nonlinear_pars.add(
                    'tau%d' % (i + 1), 
                    value=1., 
                    vary=True, 
                    min=1E-6)
            else:
                nonlinear_pars.add(
                    'tau%d' % (i + 1), 
                    value=getValue(tau), 
                    vary=doesVary(tau), 
                    min=1E-6)
        return nonlinear_pars

    @staticmethod
    def exponential(time):
        def inner_exponential(**taus):
            # taus = sorted_values(taus)
            taus = np.asarray(list(taus.values())) # may fail if not sorted
            return np.exp(-time[:, None] / taus[None, :])
        return inner_exponential


class Linear(AbstractModel):
    
    def model_function(self, independent_var):
        return self.linear(independent_var)

    def make_parameters(self):
        fit_components = int(self.user_kwargs['<lt-fit_components>'])
        linear_pars = Parameters()
        for i, amplitude in zip(range(fit_components), self.user_kwargs['--amplitude']):
            if amplitude is None:
                linear_pars.add(
                    'amplitude%d' % (i + 1), 
                    value=1., 
                    vary=True)
            else:
                linear_pars.add(
                    'amplitude%d' % (i + 1), 
                    value=getValue(amplitude),
                    vary=doesVary(amplitude))
        if self.user_kwargs['--offset'] is None:
            linear_pars.add(
                'offset', 
                value=0., 
                vary=True,
                )
        else:
            linear_pars.add(
                'offset', 
                value=getValue(self.user_kwargs['--offset']),
                vary=doesVary(self.user_kwargs['--offset'])
                )
        return linear_pars
    
    @staticmethod
    def linear(independent_var):
        def inner_linear(**linear_params):
            offset = linear_params.pop('offset')
            # amplitudes = sorted_values(linear_params)
            amplitudes = np.asarray(list(linear_params.values())) # may fail if not sorted
            return independent_var.dot(amplitudes) + offset
        return inner_linear

class Linearize():
    
    def __init__(self, ModelClass):
        self.ModelClass = ModelClass

    def make_model(self, **independent_var): 
        return GenericModel(self.model_function(**independent_var), missing='drop', name=self.ModelClass.__class__.__name__)

    def model_function(self, **independent_var):
        return self.composite(
    linear_func=Linear.linear, 
    nonlinear_func=self.ModelClass.model_function(**independent_var))
         
    def make_parameters(self):
        nonlinear_params = self.ModelClass.make_parameters()
        linear_params = Linear(self.user_kwargs).make_parameters()
        for param_name, param in linear_params.items():
            nonlinear_params.add(
            param_name,
            value=param.value,
            vary=param.vary,
            min=param.min,
            max=param.max,
            expr=param.expr)
        
        return nonlinear_params
    
    @staticmethod
    def composite(linear_func, nonlinear_func):
        def inner_composite(**params):          
            nonlinear_params = {
                key: params[key] for key in params.keys() if (
                    key.startswith('tau') or key.startswith('shift')
                    )
                    }
            linear_params = {
                key: params[key] for key in params.keys() if (
                    key.startswith('amplitude') or key.startswith('offset')
                    )
                    } 
            return linear_func(nonlinear_func(**nonlinear_params))(**linear_params)

        return inner_composite
 
    def __getattr__(self, attr):
        return getattr(self.ModelClass, attr)


class GenericModel(Model):

    def __init__(self, func, independent_vars=None,
                 param_names=None, missing='none', prefix='', name=None, **kws):
        super().__init__(func, independent_vars, param_names,
                 missing, prefix, name, **kws)
        self._statistic = None

    def _residual(self, params, data, weights, **kwargs):
        model = self.eval(params, **kwargs)
        result = self._statistic.objective_func(model, data)
        return np.asarray(result).ravel()

    def generic_fit(self, data, statistic, params,
            iter_cb=None, scale_covar=True, verbose=False, fit_kws=None,
            **kwargs):
        self._statistic = statistic
        method = self._statistic.optimization_method
        weights = None
        return super().fit(data, params, weights, method,
            iter_cb, scale_covar, verbose, fit_kws,
            **kwargs)

def doesVary(parameter, default=True):
    '''Helper function for `lmfit.Parameter`; checks if parameter does vary during
    fit (value is preceded by `~` sign).'''
    if isinstance(parameter, str): # only strings, not NoneType or Boolean
        if parameter.startswith('~'):
            return True
        else:
            return False
    return default # if not specified in parameter, return default value

def getValue(parameter):
    ''' Helper function for lmfit parameter; sets parameter value. '''
    if isinstance(parameter, str): # only strings, not NoneType or Boolean
        # parameter = parameter.lstrip('~egl') # leave only number
        try:
            return float(parameter.lstrip('~egl')) # from string to number
        except ValueError:
            return None
    return parameter

def sorted_values(parameters):
    """Sorts parameters values."""
    parameters = OrderedDict(sorted(parameters.items()))
    sorted_vals = np.asarray(list(parameters.values()))
    return sorted_vals