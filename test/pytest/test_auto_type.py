import pytest
import re
import hls4ml
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


def test_config_from_keras_default_no_calls(mocker, models):
    mocked_set_types = mocker.patch('hls4ml.utils.config.set_data_types_from_keras_model')
    mocked_accum_t = mocker.patch('hls4ml.utils.config.set_accum_from_keras_model')

    for model in models:
        hls4ml.utils.config.config_from_keras_model(model, granularity='name', data_type_mode='default')

        mocked_set_types.assert_not_called()
        mocked_accum_t.assert_not_called()


def test_config_from_keras_auto_one_call(mocker, models):
    mocked_set_types = mocker.patch('hls4ml.utils.config.set_data_types_from_keras_model')
    mocked_accum_t = mocker.patch('hls4ml.utils.config.set_accum_from_keras_model')

    expected_calls = []

    for i, model in enumerate(models, 10):
        config = hls4ml.utils.config.config_from_keras_model(model, granularity='name', data_type_mode='auto',
                                                             max_bits=i, test_inputs=[i - 1, i + 1])
        expected_calls.append(mocker.call(config, model, max_bits=i, test_inputs=[i - 1, i + 1]))
        mocked_accum_t.assert_not_called()

    mocked_set_types.assert_has_calls(expected_calls)


def test_config_from_keras_auto_accum_two_calls(mocker, models):
    mocked_set_types = mocker.patch('hls4ml.utils.config.set_data_types_from_keras_model')
    mocked_accum_t = mocker.patch('hls4ml.utils.config.set_accum_from_keras_model')

    expected_calls_types = []
    expected_calls_accum_t = []

    for i, model in enumerate(models, 10):
        config = hls4ml.utils.config.config_from_keras_model(model, granularity='name', data_type_mode='auto_accum',
                                                             max_bits=i, test_inputs=[i - 1, i + 1])
        expected_calls_types.append(mocker.call(config, model, max_bits=i, test_inputs=[i - 1, i + 1]))
        expected_calls_accum_t.append(mocker.call(config, model))

    mocked_set_types.assert_has_calls(expected_calls_types)
    mocked_accum_t.assert_has_calls(expected_calls_accum_t)


def test_config_from_keras_auto_accum_only_one_call(mocker, models):
    mocked_set_types = mocker.patch('hls4ml.utils.config.set_data_types_from_keras_model')
    mocked_accum_t = mocker.patch('hls4ml.utils.config.set_accum_from_keras_model')

    expected_calls = []

    for model in models:
        config = hls4ml.utils.config.config_from_keras_model(model, granularity='name',
                                                             data_type_mode='auto_accum_only')
        expected_calls.append(mocker.call(config, model))
        mocked_set_types.assert_not_called()

    mocked_accum_t.assert_has_calls(expected_calls)


def test_config_from_keras_default_no_name_granularity_required(models):
    # No exceptions = this test passes
    for model in models:
        hls4ml.utils.config.config_from_keras_model(model, granularity='model')
        hls4ml.utils.config.config_from_keras_model(model, granularity='type')


def test_config_from_keras_auto_name_granularity_required(mocker, models):
    mocked_set_types = mocker.patch('hls4ml.utils.config.set_data_types_from_keras_model')

    for model in models:
        with pytest.raises(Exception, match=r'.*"name".*'):
            hls4ml.utils.config.config_from_keras_model(model, granularity='type', data_type_mode='auto')

        with pytest.raises(Exception, match=r'.*"name".*'):
            hls4ml.utils.config.config_from_keras_model(model, granularity='model', data_type_mode='auto')

        mocked_set_types.assert_not_called()


def test_config_from_keras_auto_accum_name_granularity_required(mocker, models):
    mocked_set_types = mocker.patch('hls4ml.utils.config.set_data_types_from_keras_model')
    mocked_accum_t = mocker.patch('hls4ml.utils.config.set_accum_from_keras_model')

    for model in models:
        with pytest.raises(Exception, match=r'.*"name".*'):
            hls4ml.utils.config.config_from_keras_model(model, granularity='type', data_type_mode='auto_accum')

        with pytest.raises(Exception, match=r'.*"name".*'):
            hls4ml.utils.config.config_from_keras_model(model, granularity='model', data_type_mode='auto_accum')

        mocked_set_types.assert_not_called()
        mocked_accum_t.assert_not_called()


def test_config_from_keras_auto_accum_only_name_granularity_required(mocker, models):
    mocked_accum_t = mocker.patch('hls4ml.utils.config.set_accum_from_keras_model')

    for model in models:
        with pytest.raises(Exception, match=r'.*"name".*'):
            hls4ml.utils.config.config_from_keras_model(model, granularity='type', data_type_mode='auto_accum_only')

        with pytest.raises(Exception, match=r'.*"name".*'):
            hls4ml.utils.config.config_from_keras_model(model, granularity='model', data_type_mode='auto_accum_only')

        mocked_accum_t.assert_not_called()


def test_set_data_types_from_keras_max_bits_complied_with(models):
    for i, model in enumerate(models, 10):
        input_shape = (5,) + model.layers[0].input_shape[1:]
        config = hls4ml.utils.config.config_from_keras_model(model, granularity='name', default_precision='dummy')
        hls4ml.utils.config.set_data_types_from_keras_model(config, model, max_bits=i,
                                                            test_inputs=np.random.rand(*input_shape))

        for value in config['LayerName'].values():
            qkeras_inferred = value['QKerasInferred'] if 'QKerasInferred' in value else []

            if isinstance(value['Precision'], dict):
                for key, precision in value['Precision'].items():
                    if key not in qkeras_inferred and precision != 'dummy':
                        match = re.search(r'(\d+)', precision)
                        assert match
                        assert int(match.group(1)) <= i + 1
            elif value['Precision'] != 'dummy':
                match = re.search(r'(\d+)', value['Precision'])
                assert match
                assert int(match.group(1)) <= i + 1
