# -*- coding: utf-8 -*-

"""Provide `fluo.Model` object.

`fluo.Model` is an object for model definition and evaluation.
It wraps the `fluo.GenericModel`. `fluo.GenericModel` overrides the lmfit.Model to utilize
Statistic object in fit.

"""

from abc import ABCMeta, abstractmethod
import random
import numpy as np
import scipy
import scipy.interpolate
import lmfit
import tqdm

class Model():
    """Wrapper around `fluo.GenericModel`.

    fluo.Model is an abstract class for fluo.Model objects.

    Parameters
    ----------
    model_components : int
        Number of components in model (i. e. number of exponents).
    model_parameters : dict
        Dict with names of parameters encoded by keys (str)
        and values with dictionary.

    Attributes
    ----------
    name : str

    Methods
    -------
    make_model : fluo.GenericModel

    """

    __metaclass__ = ABCMeta

    def __init__(self, model_components, model_parameters):
        """Initialize Model object.

        Parameters
        ----------
        model_components : int
            Number of components in model (i. e. number of exponents).
        model_parameters : dict
            Dict with names of parameters encoded by keys (str)
            and values with dictionary.

        """
        self.model_components = model_components
        self.model_parameters = model_parameters
        self.name = self.__class__.__name__

    def make_model(self, **independent_var):
        """Make a Model for evaluation and fitting.

        Prameteres
        ----------
        independent_var : dict
            Keyword arguments for model evaluation.

        Returns
        -------
        fluo.GenericModel

        """
        return GenericModel(
            self._model_function(**independent_var),
            missing='omit',
            name=self.name
            )

    @abstractmethod
    def _model_function(self, **independent_var):
        """Make model function for fluo.GenericModel."""
        raise NotImplementedError()

    @abstractmethod
    def make_parameters(self):
        """Make lmfit.Parameters."""
        raise NotImplementedError()

class GlobalModel(Model):
    """GlobalModel object composed from multiple `fluo.Fitter` instances.

    Parameters
    ----------
    fitter_classes : list of fluo.Fitter
    shared : list of str, optional
        List of parameters names shared between fitted measurements.

    Attributes
    ----------
    name : str
    local_independent_var : list of dict
        Independent variables for a local model evaluation. Dict with names
        of independent variables encoded by keys (str) and values as ndarrays.
    local_dependent_var : list of ndarray
        List of 1D ndarrays with dependent variable for fitting.
    local_indexes : ndarray
        Indexes separating individual measurements.
    statistic : fluo.Statistic
        Global Statistic class for simultaneous fit.

    Methods
    -------
    make_model : fluo.GenericModel
        Global Model for simultaneous fitting.
    make_parameters : lmfit.Parameters
    make_local_atrribute : list
        List of local attribute of fitter_classes.

    """

    def __init__(self, fitter_classes, shared=None):
        """Initialize GlobalModel object.

        Parameters
        ----------
        fitter_classes : list of fluo.Fitter
        shared : list of str, optional
            List of parameters names shared between fitted measurements.

        """
        self.fitter_classes = fitter_classes
        self.shared = shared
        if self.shared is None:
            self.shared = []
        self.name = '{}({})'.format(
            self.__class__.__name__,
            # name in every Fitter class should be the same
            fitter_classes[0].model_class.name
            )
        self.local_independent_var = self.make_local_atrribute(fitter_classes, 'independent_var')
        self.local_dependent_var = self.make_local_atrribute(fitter_classes, 'dependent_var')
        self.local_indexes = self._make_local_indexes(self.local_dependent_var)
        # statistic in every Fitter class should be the same
        self.statistic = fitter_classes[0].statistic
        self._local_parameters = self.make_local_atrribute(
            fitter_classes,
            'parameters')
        self._parameters, self._parameters_references = self._glue_parameters()

    def _model_function(self, **independent_var):
        return self._global_eval(**independent_var)

    def make_parameters(self):
        """Make parameters for GlobalModel evaluation.

        Returns
        -------
        lmfit.Parameters

        """
        return self._parameters

    def _global_eval(self, independent_var):
        """Global evaluation of local Models.

        Prameteres
        ----------
        independent_var : list
            List with independent_var for local Model evaluations.

        """
        def _inner_global_eval(**params):
            for name, value in params.items():
                fitter_i, local_name = self._parameters_references[name]
                self._local_parameters[fitter_i][local_name].value = value
            _global_eval = []
            for i, local_fitter in enumerate(self.fitter_classes):
                model_i = local_fitter.model_class.make_model(
                    **independent_var[i]
                    )
                local_eval = model_i.eval(**self._local_parameters[i])
                _global_eval.append(local_eval)
            return np.concatenate(_global_eval)
        return _inner_global_eval

    def _glue_parameters(self):
        """Glue local Parameters, giving them new global names."""
        _parameters_references = dict()
        all_params = lmfit.Parameters()
        for i, params_i in enumerate(self._local_parameters):
            for old_name, param in params_i.items():
                new_name = old_name + '_file%d' % (i + 1)
                _parameters_references[new_name] = (i, param.name)
                all_params.add(new_name,
                               value=param.value,
                               vary=param.vary,
                               min=param.min,
                               max=param.max,
                               expr=param.expr)
        for param_name, param in all_params.items():
            for constraint in self.shared:
                if param_name.startswith(constraint) and \
                not param_name.endswith('_file1'):
                    param.expr = constraint + '_file1'
        return all_params, _parameters_references

    @staticmethod
    def make_local_atrribute(fitters, atrribute):
        """Make list of arrtibute of local Fitter.

        Parameters
        ----------
        fitters : list of fluo.Fitter
        atrribute: str
            fluo.Fitter attribute name.

        Returns
        -------
        list
            List with arrtibute of local Fitters.

        """
        return [getattr(fitter, atrribute) for fitter in fitters]

    @staticmethod
    def _make_local_indexes(arrs):
        """List indexes where local measurements are glued toghether."""
        return np.cumsum([len(arr) for arr in arrs])[:-1]


class AddConstant():
    """Add a constant to a fluo.Model.

    Parameters
    ----------
    model_class : fluo.Model

    Methods
    -------
    make_model : fluo.GenericModel
    make_parameters : lmfit.Parameters

    """

    def __init__(self, model_class):
        """Initialize AddConstant object.

        Parameters
        ----------
        model_class : fluo.Model

        """
        self.model_class = model_class
        self.name = '{}({})'.format(
            self.__class__.__name__,
            self.model_class.name
            )

    def make_model(self, **independent_var):
        """Make a Model for evaluation and fitting.

        Prameteres
        ----------
        independent_var : dict
            Dictionary with independen variable for model evaluation.

        Returns
        -------
        fluo.GenericModel

        """
        return GenericModel(
            self._model_function(**independent_var),
            missing='omit',
            name=self.name
            )

    def _model_function(self, **independent_var):
        """Make a function for `fluo.GenericModel`."""
        return self._add_constant(**independent_var)

    def make_parameters(self):
        """Make Parameters for Model evaluation.

        Returns
        -------
        lmfit.Parameters

        """
        pars = self.model_class.make_parameters()
        pars.add(
            'offset',
            **self.model_parameters.get('offset', {'value': 0.1, 'vary': True})
            )
        return pars

    def _add_constant(self, **independent_var):
        """Evaluate model and add constant to it."""
        model = self.model_class.make_model(**independent_var)
        def _inner_add_constant(**params):
            offset = params.pop('offset')
            return model.eval(**params) + offset

        return _inner_add_constant

    def __getattr__(self, atrribute):
        """Return attribute from `model_class`."""
        return getattr(self.model_class, atrribute)


class Linearize():
    """Linear combination of a Model.

    Wrapper around fluo.Model. Makes dot product of a Model and linear
    coeficients. Utilizes Linear class.

    Parameters
    ----------
    model_class : fluo.Model

    Methods
    -------
    make_model : fluo.GenericModel
    make_parameters : lmfit.Parameters

    """

    def __init__(self, model_class):
        """Initialize Linearize object.

        Parameters
        ----------
        model_class : fluo.Model

        """
        self.model_class = model_class
        self.name = '{}({})'.format(
            self.__class__.__name__,
            self.model_class.name
            )

    def make_model(self, **independent_var):
        """Make a Model for evaluation and fitting.

        Prameteres
        ----------
        independent_var : dict
            Dictionary with independen variable for model evaluation.

        Returns
        -------
        fluo.GenericModel

        """
        return GenericModel(
            self._model_function(**independent_var),
            missing='omit',
            name=self.name)

    def _model_function(self, **independent_var):
        """Make a function for `fluo.GenericModel`."""
        return self._composite(**independent_var)

    def make_parameters(self):
        """Make Parameters for Model evaluation.

        Returns
        -------
        lmfit.Parameters

        """
        nonlinear_params = self.model_class.make_parameters()
        linear_params = Linear(self.model_components, self.model_parameters).make_parameters()
        for param_name, param in linear_params.items():
            nonlinear_params.add(
                param_name,
                value=param.value,
                vary=param.vary,
                min=param.min,
                max=param.max,
                expr=param.expr
                )

        return nonlinear_params

    def _composite(self, **independent_var):
        """Combine linearly non-linear model."""
        linear_func = Linear.linear
        nonlinear_model = self.model_class.make_model(**independent_var)
        def _inner_composite(**params):
            nonlinear_params = {
                key: params[key] for key in params.keys() if (
                    key.startswith('tau') or key.startswith('shift')
                    )
                }
            linear_params = {
                key: params[key] for key in params.keys() if (
                    key.startswith('amplitude')
                    )
                }
            return linear_func(
                nonlinear_model.eval(**nonlinear_params)
                )(**linear_params)

        return _inner_composite

    def __getattr__(self, atrribute):
        """Return attribute from `model_class`."""
        return getattr(self.model_class, atrribute)

class Convolve():
    """Convolve a Model with instrument response.

    Wrapper around fluo.Model.

    Parameters
    ----------
    model_class : fluo.Model
    convolution_method : str, optional
        'discrete' by default. Accepts the following str: 'discrete',
        'monte_carlo' (for Monte Carlo convolution).

    Methods
    -------
    make_model : fluo.GenericModel
    make_parameters : lmfit.Parameters
    shift_decay : ndarray
    convolve : ndarray
    monte_carlo_convolve : ndarray

    """

    def __init__(self, model_class, convolution_method='discrete', **convolution_kwargs):
        """Initialize AddConstant object.

        Parameters
        ----------
        model_class : fluo.Model
        convolution_method : str, optional
            'discrete' by default. Accepts the following str: 'discrete',
            'monte_carlo' (for Monte Carlo convolution).

        """
        self.model_class = model_class
        self.convolution_method = convolution_method
        self.convolution_kwargs = convolution_kwargs
        self.__convolve = self._allowed_convolutions[convolution_method]
        self.name = '{}({})'.format(
            self.__class__.__name__,
            self.model_class.name
            )

    def make_model(self, **independent_var):
        """Make a Model for evaluation and fitting.

        Prameteres
        ----------
        independent_var : dict
            Keyword arguments for model evaluation.

        Returns
        -------
        fluo.GenericModel
        """
        return GenericModel(
            self._model_function(**independent_var),
            missing='omit',
            name=self.name
            )

    def _model_function(self, **independent_var):
        """Make a function for `fluo.GenericModel`."""
        return self._convolve(**independent_var)

    def make_parameters(self):
        """Make Parameters for Model evaluation.

        Returns
        -------
        lmfit.Parameters

        """
        nonlinear_pars = self.model_class.make_parameters()
        nonlinear_pars.add(
            'shift',
            **self.model_parameters.get('shift', {'value': 1, 'vary': True})
            )
        return nonlinear_pars

    def _convolve(self, **independent_var):
        """Evaluate and convolve model with instrument response."""
        independent_var = independent_var.copy()
        time = independent_var['time']
        instrument_response = independent_var.pop('instrument_response')

        def _inner_convolve(**params):
            params = params.copy()
            shifted_instrument_response = self.shift_decay(
                time,
                instrument_response,
                params.pop('shift')
                )
            to_convolve_with = self.model_class.make_model(
                **independent_var
                ).eval(**params)
            ncols, *nrows = to_convolve_with.shape
            try:
                convolved = np.zeros(to_convolve_with.shape)
                for i in range(*nrows):
                    convolved[:, i] = self.__convolve(
                        shifted_instrument_response,
                        to_convolve_with[:, i],
                        **self.convolution_kwargs
                        )
            except TypeError:
                convolved = self.__convolve(
                    shifted_instrument_response,
                    to_convolve_with, **self.convolution_kwargs)
            return convolved

        return _inner_convolve

    @property
    def _allowed_convolutions(self):
        """Allowed convolution methods."""
        return dict(
            discrete=self.convolve,
            monte_carlo=self.monte_carlo_convolve
            )

    @staticmethod
    def shift_decay(x_var, y_var, shift):
        """
        Shift y-axis variable on x-axis.

        Parameters
        ----------
        x_var : ndarray
        y_var : ndarray
        shift : float

        Returns
        -------
        ndarray

        """
        y_var_interpolated = scipy.interpolate.interp1d(
            x_var,
            y_var,
            kind='slinear',
            bounds_error=False,
            fill_value=0.0)
        return y_var_interpolated(x_var + shift)

    @staticmethod
    def convolve(left, right):
        """Discrete convolution of `left` with `right`.

        Parameters
        ----------
        left : ndarray
            Left 1D ndarray
        right : ndarray
            Right 1D ndarray

        Returns
        -------
        ndarray

        """
        return np.convolve(left, right, mode='full')[:len(right)]

    @staticmethod
    def monte_carlo_convolve(left, right, peak_cnts=None, verbose=True):
        """Monte Carlo convolution of `left` with `right`.

        Simulates distorted fluorescence decay distorted by instrument
        response with Poisson distributed observed values.

        Parameters
        ----------
        left : ndarray
            1D array
        right : ndarray
            1D array (should be the same length as `left`).
        peak_cnts : int, optional
            Counts in maximum (max of `left` by default).
        verbose : bool
            Print simulation progress.

        Returns
        -------
        ndarray

        """
        left_max = np.max(left)
        # probability distribution of left scalled to 1
        probability_left = list(left/left_max)
        index_max = len(probability_left)-1
        # probability distribution of right scalled to 1
        probability_right = list(right/np.max(right))
        print()
        print('[[Wait until simulation is done. It may take some time.]]')
        print()
        monte_carlo_convolution = [0] * len(probability_left)
        if peak_cnts == None:
            peak_cnts = left_max
        else:
            peak_cnts = int(peak_cnts)
        progress_bar = tqdm.tqdm(total=peak_cnts+1)
        while max(monte_carlo_convolution) <= peak_cnts: # stops when peak_counts is reached
            last_iter_max = max(monte_carlo_convolution)
            index_left = draw_from_probability_distribution(probability_left)
            index_right = draw_from_probability_distribution(probability_right)
            index_drawn = index_left + index_right  # draw channel number
            if index_drawn <= index_max:  # channel must be in range
                monte_carlo_convolution[index_drawn] += 1  # add count in channel
            if verbose:
                progress_bar.update(max(monte_carlo_convolution)-last_iter_max)
        progress_bar.close()
        return np.asarray(monte_carlo_convolution)

    def __getattr__(self, atrribute):
        """Return attribute from `model_class`."""
        return getattr(self.model_class, atrribute)

class Exponential(Model):
    """Exponential Model.

    Parameters
    ----------
    model_components : int
        Number of components in model (i. e. number of exponents).
    model_parameters : dict
        Dict with names of parameters encoded by keys (str)
        and values with dictionary.

    Attributes
    ----------
    name : str

    Methods
    -------
    make_model : fluo.GenericModel
    make_parameters : lmfit.Parameters
    exponential : ndarray

    """

    def _model_function(self, **independent_var):
        """Make a function for `fluo.GenericModel`."""
        return self.exponential(**independent_var)

    def make_parameters(self):
        """Make Parameters for Model evaluation.

        Returns
        -------
        lmfit.Parameters

        """
        nonlinear_pars = lmfit.Parameters()
        for i in range(self.model_components):
            nonlinear_pars.add(
                'tau{}'.format(i+1),
                **self.model_parameters.get(
                    'tau{}'.format(i+1),
                    {'value': 1, 'vary': True, 'min': 1E-6}
                    )
                )
        return nonlinear_pars

    @staticmethod
    def exponential(time):
        """Exponential decay."""
        def inner_exponential(**taus):
            taus = np.asarray(list(taus.values())) # may fail if not sorted
            return np.exp(-time[:, None] / taus[None, :])
        return inner_exponential

class Linear(Model):
    """Linear Model.

    Make a dot product of an independent variable and linear coeficients.

    Parameters
    ----------
    model_components : int
        Number of components in model (i. e. number of exponents).
    model_parameters : dict
        Dict with names of parameters encoded by keys (str)
        and values with dictionary.

    Attributes
    ----------
    name : str

    Methods
    -------
    make_model : fluo.GenericModel
    make_parameters : lmfit.Parameters
    exponential : ndarray

    """

    def _model_function(self, independent_var):
        """Make a function for `fluo.GenericModel`."""
        return self.linear(independent_var)

    def make_parameters(self):
        """Make Parameters for Model evaluation.

        Returns
        -------
        lmfit.Parameters

        """
        linear_pars = lmfit.Parameters()
        for i in range(self.model_components):
            linear_pars.add(
                'amplitude{}'.format(i+1),
                **self.model_parameters.get(
                    'amplitude{}'.format(i+1),
                    {'value': 0.5, 'vary': True}
                )
            )
        return linear_pars

    @staticmethod
    def linear(independent_var):
        """Linear combination.

        Parameters
        ----------
        independent_var : ndarray
            2D ndarray for a dot product.

        Returns
        -------
        ndarray
            1D ndarray with dot product of an independent variable and
            linear coeficients.

        """
        def inner_linear(**amplitudes):
            amplitudes = np.asarray(list(amplitudes.values())) # may fail if not sorted
            return independent_var.dot(amplitudes)
        return inner_linear


class GenericModel(lmfit.Model):
    """Create a model from a user-supplied model function.

    Overrides `lmfit.Model`

    """

    def __init__(self, func, independent_vars=None,
                 param_names=None, missing='none', prefix='', name=None, **kws):
        """Initialize GenericModel object.

        Overrides `lmfit.Model` __init__ to introduce fluo.Statistic.

        """
        super().__init__(func, independent_vars, param_names, missing, prefix,
        name, **kws)
        self._statistic = None

    def _residual(self, params, data, weights, **kwargs):
        """Return the residual.

        Overrides `lmfit.Model` _residual to introduce fluo.Statistic.
        The definition of residual funtion is provided by Statistic object.

        """
        model = self.eval(params, **kwargs)
        result = self._statistic.objective_func(model, data)
        return np.asarray(result).ravel()

    def generic_fit(self, data, statistic, params,
            iter_cb=None, scale_covar=True, verbose=False, fit_kws=None,
            **kwargs):
        """Fit the model to the data using the supplied Parameters.

        Overrides `lmfit.fit` method to utilize Statistic object in fit.

        Parameters
        ----------
        data : ndarray
            Array of data to be fit.
        statistic : fluo.Statistic
        params : lmfit.Parameters
        iter_cb : callable, optional
             Callback function to call at each iteration (default is None).
        scale_covar : bool, optional
             Whether to automatically scale the covariance matrix when
             calculating uncertainties (default is True, `leastsq` method only).
        verbose: bool, optional
             Whether to print a message when a new parameter is added because
             of a hint (default is True).
        fit_kws: dict, optional
             Options to pass to the minimizer being used.
        **kwargs: optional
             Arguments to pass to the  model function, possibly overriding
             params.

        Returns
        -------
        lmfit.ModelResult

        """
        self._statistic = statistic
        method = self._statistic.optimization_method
        weights = None
        return super().fit(data, params, weights, method, iter_cb, scale_covar,
    verbose, fit_kws, **kwargs)


def draw_from_probability_distribution(distribution):
    """Draw from an arbitrary distribution using acceptance-rejection method.

    Parameters
    ----------
    distribution : ndarray
        1D ndarray with probabalities distribution (scalled to 1).

    Returns
    -------
    int
        Drawn index.

    """
    x_max = len(distribution)-1
    y_min = min(distribution)

    accepted = False
    while not accepted:
        x_random = random.randint(0, x_max)
        y_random = random.uniform(y_min, 1.)
        if y_random <= distribution[x_random]:
            accepted = True
            return x_random
