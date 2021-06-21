import uuid
import pytest
import os
import tempfile
import hls4ml.model
import numpy as np
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras.layers import Input, Conv2D, Dense, Activation, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.models import Sequential
from qkeras import QDense, QConv2D, quantized_bits

np.random.seed(1234567890)


@pytest.fixture(scope='module')
def model1():
    model = Sequential([
        Dense(10, input_shape=(20,), name='fc1', kernel_initializer='lecun_uniform',
              bias_initializer='lecun_uniform')
    ])
    return model


@pytest.fixture(scope='module')
def model2():
    model = Sequential([
        Dense(20, input_shape=(10,), name='fc1', kernel_initializer='lecun_uniform',
              bias_initializer='lecun_uniform', activation='relu'),
        QDense(5, name='fc2', kernel_initializer='lecun_uniform',
               bias_initializer='lecun_uniform', kernel_quantizer=quantized_bits(15, 5),
               bias_quantizer=quantized_bits(10, 2)),
        Activation(activation='softmax', name='softmax')
    ])
    return model


@pytest.fixture(scope='module')
def model3():
    model = Sequential([
        Conv2D(8, kernel_size=(2, 2), input_shape=(16, 16, 3),
               kernel_initializer='lecun_uniform',
               bias_initializer='lecun_uniform', name='conv1'),
        Flatten(),
        Dense(1, kernel_initializer='lecun_uniform', bias_initializer='lecun_uniform', name='fc1', activation='tanh')
    ])
    return model


@pytest.fixture(scope='module')
def model4():
    model = Sequential(
        [
            Input(shape=(28, 28, 1)),
            Conv2D(16, kernel_size=(3, 3), activation='relu',
                   kernel_initializer='lecun_uniform',
                   bias_initializer='lecun_uniform'),
            MaxPooling2D(pool_size=(2, 2)),
            QConv2D(16, kernel_size=(3, 3), activation='relu',
                    kernel_initializer='lecun_uniform',
                    bias_initializer='lecun_uniform',
                    kernel_quantizer=quantized_bits(15, 2, alpha=1),
                    bias_quantizer=quantized_bits(11, 3, alpha=1)),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dropout(0.5),
            QDense(10,
                   kernel_initializer='lecun_uniform',
                   bias_initializer='lecun_uniform',
                   kernel_quantizer=quantized_bits(14, 2, alpha=1),
                   bias_quantizer=quantized_bits(12, 1, alpha=1)),
            Activation(activation='softmax', name='softmax'),
        ]
    )
    return model


@pytest.fixture(scope='module')
def model5():
    model = Sequential([
        Dense(64, input_shape=(16,), name='fc1',
              bias_initializer='lecun_uniform',
              kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.0001)),
        Activation(activation='relu', name='relu1'),
        Dense(32, name='fc2',
              bias_initializer='lecun_uniform',
              kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.0001)),
        Activation(activation='relu', name='relu2'),
        Dense(32, name='fc3',
              bias_initializer='lecun_uniform',
              kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.0001)),
        Activation(activation='relu', name='relu3'),
        Dense(5, name='output',
              bias_initializer='lecun_uniform',
              kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.0001)),
        Activation(activation='softmax', name='softmax')
    ])
    return model


@pytest.fixture(scope='module')
def model6():
    model = Sequential([
        Dense(64, input_shape=(16,), name='fc1',
              use_bias=False,
              kernel_initializer='lecun_uniform', kernel_regularizer=l2(0.0001)),
        Activation(activation='relu', name='relu1'),
        Dense(32, name='fc2',
              use_bias=False,
              kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.0001)),
        Activation(activation='relu', name='relu2'),
        Dense(32, name='fc3',
              use_bias=False,
              kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.0001)),
        Activation(activation='relu', name='relu3'),
        Dense(32, name='fc4',
              use_bias=False,
              kernel_initializer='lecun_uniform', kernel_regularizer=l2(0.0001)),
        Activation(activation='relu', name='relu4'),
        Dense(32, name='fc5',
              use_bias=False,
              kernel_initializer='lecun_uniform'),
        Activation(activation='relu', name='relu5'),
        Dense(32, name='fc6',
              bias_initializer='lecun_uniform',
              kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.0001)),
        Activation(activation='relu', name='relu6'),
        Dense(5, name='output',
              bias_initializer='lecun_uniform',
              kernel_initializer='lecun_uniform'),
        Activation(activation='softmax', name='softmax')
    ])
    return model


@pytest.fixture(scope='module')
def models(model1, model2, model3, model4, model5, model6):
    return [model1, model2, model3, model4, model5, model6]


@pytest.fixture(scope='module')
def qkeras_models(model2, model4):
    return [model2, model4]


@pytest.fixture(scope='module')
def optimizers():
    return ['fuse_batch_norm']


def check_model(model, plot, test_inputs=None, config_updates=None):
    config = hls4ml.utils.config_from_keras_model(model=model, granularity='name')

    for value in config['LayerName'].values():
        value['Trace'] = True

    if config_updates is not None:
        for key, value in config_updates.items():
            config[key] = value

    with tempfile.TemporaryDirectory() as tmp_dir:
        output_dir = uuid.uuid4().hex

        while os.path.exists(os.path.join(tmp_dir, output_dir)):
            output_dir = uuid.uuid4().hex

        hls_model = hls4ml.converters.convert_from_keras_model(model,
                                                               hls_config=config,
                                                               output_dir=os.path.join(tmp_dir, output_dir),
                                                               fpga_part='xczu7cg-fbvb900-1-e')

        wp, wph, ap, aph = hls4ml.model.profiling.numerical(model, hls_model, X=test_inputs, plot=plot)

    assert wp is not None
    assert wph is not None

    if test_inputs is not None:
        assert ap is not None
        assert aph is not None
    else:
        assert ap is None
        assert aph is None

    if config_updates is not None:
        for key, value in config_updates.items():
            assert key in config
            assert config[key] == value


def test_numerical_finishes_no_test_inputs_boxplot(models):
    for model in models:
        check_model(model, 'boxplot')


def test_numerical_finishes_with_test_inputs_boxplot(models):
    for model in models:
        input_shape = (5,) + model.layers[0].input_shape[1:]
        check_model(model, 'boxplot', np.random.rand(*input_shape))


@pytest.mark.skip(reason='violinplot might not be implemented')
def test_numerical_finishes_no_test_inputs_violinplot(models):
    for model in models:
        check_model(model, 'violinplot')


@pytest.mark.skip(reason='violinplot might not be implemented')
def test_numerical_finishes_with_test_inputs_violinplot(models):
    for model in models:
        input_shape = (5,) + model.layers[0].input_shape[1:]
        check_model(model, 'violinplot', np.random.rand(*input_shape))


def test_numerical_finishes_no_test_inputs_histogram(models):
    for model in models:
        check_model(model, 'histogram')


def test_numerical_finishes_with_test_inputs_histogram(models):
    for model in models:
        input_shape = (5,) + model.layers[0].input_shape[1:]
        check_model(model, 'histogram', np.random.rand(*input_shape))


@pytest.mark.skip(reason='FacetGrid might not be implemented')
def test_numerical_finishes_no_test_inputs_facetgrid(models):
    for model in models:
        check_model(model, 'FacetGrid')


@pytest.mark.skip(reason='FacetGrid might not be implemented')
def test_numerical_finishes_with_test_inputs_facetgrid(models):
    for model in models:
        input_shape = (5,) + model.layers[0].input_shape[1:]
        check_model(model, 'FacetGrid', np.random.rand(*input_shape))


def test_numerical_finishes_explicit_optimizers_no_test_inputs_boxplot(models, optimizers):
    for model in models:
        check_model(model, 'boxplot', config_updates={'Optimizers': optimizers})


def test_numerical_finishes_explicit_optimizers_with_test_inputs_boxplot(models, optimizers):
    for model in models:
        input_shape = (5,) + model.layers[0].input_shape[1:]
        check_model(model, 'boxplot', np.random.rand(*input_shape),
                    config_updates={'Optimizers': optimizers})


@pytest.mark.skip(reason='violinplot might not be implemented')
def test_numerical_finishes_explicit_optimizers_no_test_inputs_violinplot(models, optimizers):
    for model in models:
        check_model(model, 'violinplot', config_updates={'Optimizers': optimizers})


@pytest.mark.skip(reason='violinplot might not be implemented')
def test_numerical_finishes_explicit_optimizers_with_test_inputs_violinplot(models, optimizers):
    for model in models:
        input_shape = (5,) + model.layers[0].input_shape[1:]
        check_model(model, 'violinplot', np.random.rand(*input_shape),
                    config_updates={'Optimizers': optimizers})


def test_numerical_finishes_explicit_optimizers_no_test_inputs_histogram(models, optimizers):
    for model in models:
        check_model(model, 'histogram', config_updates={'Optimizers': optimizers})


def test_numerical_finishes_explicit_optimizers_with_test_inputs_histogram(models, optimizers):
    for model in models:
        input_shape = (5,) + model.layers[0].input_shape[1:]
        check_model(model, 'histogram', np.random.rand(*input_shape),
                    config_updates={'Optimizers': optimizers})


@pytest.mark.skip(reason='FacetGrid might not be implemented')
def test_numerical_finishes_explicit_optimizers_no_test_inputs_facetgrid(models, optimizers):
    for model in models:
        check_model(model, 'FacetGrid', config_updates={'Optimizers': optimizers})


@pytest.mark.skip(reason='FacetGrid might not be implemented')
def test_numerical_finishes_explicit_optimizers_with_test_inputs_facetgrid(models, optimizers):
    for model in models:
        input_shape = (5,) + model.layers[0].input_shape[1:]
        check_model(model, 'FacetGrid', np.random.rand(*input_shape),
                    config_updates={'Optimizers': optimizers})


def test_numerical_finishes_explicit_skip_optimizers_no_test_inputs_boxplot(models, optimizers):
    for model in models:
        check_model(model, 'boxplot', config_updates={'SkipOptimizers': optimizers})


def test_numerical_finishes_explicit_skip_optimizers_with_test_inputs_boxplot(models, optimizers):
    for model in models:
        input_shape = (5,) + model.layers[0].input_shape[1:]
        check_model(model, 'boxplot', np.random.rand(*input_shape), config_updates={'SkipOptimizers': optimizers})


@pytest.mark.skip(reason='violinplot might not be implemented')
def test_numerical_finishes_explicit_skip_optimizers_no_test_inputs_violinplot(models, optimizers):
    for model in models:
        check_model(model, 'violinplot', config_updates={'SkipOptimizers': optimizers})


@pytest.mark.skip(reason='violinplot might not be implemented')
def test_numerical_finishes_explicit_skip_optimizers_with_test_inputs_violinplot(models, optimizers):
    for model in models:
        input_shape = (5,) + model.layers[0].input_shape[1:]
        check_model(model, 'violinplot', np.random.rand(*input_shape), config_updates={'SkipOptimizers': optimizers})


def test_numerical_finishes_explicit_skip_optimizers_no_test_inputs_histogram(models, optimizers):
    for model in models:
        check_model(model, 'histogram', config_updates={'SkipOptimizers': optimizers})


def test_numerical_finishes_explicit_skip_optimizers_with_test_inputs_histogram(models, optimizers):
    for model in models:
        input_shape = (5,) + model.layers[0].input_shape[1:]
        check_model(model, 'histogram', np.random.rand(*input_shape), config_updates={'SkipOptimizers': optimizers})


@pytest.mark.skip(reason='FacetGrid might not be implemented')
def test_numerical_finishes_explicit_skip_optimizers_no_test_inputs_facetgrid(models, optimizers):
    for model in models:
        check_model(model, 'FacetGrid', config_updates={'SkipOptimizers': optimizers})


@pytest.mark.skip(reason='FacetGrid might not be implemented')
def test_numerical_finishes_explicit_skip_optimizers_with_test_inputs_facetgrid(models, optimizers):
    for model in models:
        input_shape = (5,) + model.layers[0].input_shape[1:]
        check_model(model, 'FacetGrid', np.random.rand(*input_shape), config_updates={'SkipOptimizers': optimizers})
