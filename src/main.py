import typing

import numpy as np
import Config
import tensorflow as tf
import matplotlib.pyplot as plt

__licence__ = 'Comune di Val Sesia, agenzia federale (associazione a delinquere)'
__version__ = '0.1'
__dependencies__ = ['numpy 1.20', 'tensorflow .', 'matplotlib .']

METEO_SHAPE = 1


class SimpleLTMSModel(tf.keras.Model):

    def __init__(self):
        super().__init__()
        self.make_rep_layer = tf.keras.layers.LSTM(
            units=250,
            input_shape=(7, 24 + METEO_SHAPE),
            activation='tanh',
            recurrent_activation='sigmoid',
            use_bias=True,
            kernel_initializer='glorot_uniform',
            recurrent_initializer='orthogonal',
            bias_initializer='zeros',
            unit_forget_bias=True,
            recurrent_dropout=0.0)
        self.make_pred_layer = tf.keras.layers.LSTM(
            units=250,
            input_shape=(7, METEO_SHAPE),
            activation='tanh',
            recurrent_activation='sigmoid',
            use_bias=True,
            kernel_initializer='glorot_uniform',
            recurrent_initializer='orthogonal',
            bias_initializer='zeros',
            unit_forget_bias=True,
            recurrent_dropout=0.0,
            return_sequences=True,
            return_state=False,
            unroll=True)
        self.final_layer = tf.keras.layers.Dense(168)

    def __call__(self, inputs, training=False):
        x = self.make_rep_layer(inputs[0])
        y = self.make_rep_layer(inputs[1], initial_state=x)
        z = self.final_layer(y)
        return z


class CustomCallback(tf.keras.callbacks.ProgbarLogger):

    def __init__(self, test_metrics: list[tf.Tensor, tf.Tensor], log_list: list, model):
        super().__init__()
        self.test_data = test_metrics
        self.logger = log_list
        self.attached_model = model

    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)
        x = self.test_data[0]
        y = self.test_data[1]
        test_loss = self.attached_model.evaluate(x, y, batch_size=32)
        # Append only mean squared error.
        self.logger.append(test_loss[1])

def real_trig_to_signal(trig_coeffs: typing.Union[list, np.array]):
    """

    :param trig_coeffs:
    :return:
    """
    exp_coeffs = real_trig_to_exp(trig_coeffs)
    return real_exp_to_signal(exp_coeffs)


def real_exp_to_signal(exp_coeffs: typing.Union[list, np.array]):
    """

    :param exp_coeffs:
    :return:
    """
    signal = np.fft.irfft(exp_coeffs)
    return signal


def real_exp_to_trig(exp_coeffs: typing.Union[list, np.array]):
    """

    :param exp_coeffs:
    :return:
    """
    positive_frequencies = exp_coeffs[1:]
    # Symmetric Hermitian implies that the negative
    # frequencies reduce to the complex conjugate of the positive
    # frequencies.
    negative_frequencies = np.conjugate(positive_frequencies)
    complex_offset = exp_coeffs[0]

    # Again, real input implies that offset is also real.
    real_offset = np.real(2 * complex_offset)
    cosine_coefficients = np.real((positive_frequencies + negative_frequencies))
    sine_coefficients = np.real((0 + 1j) * (positive_frequencies - negative_frequencies))

    format_packed = (real_offset, cosine_coefficients, sine_coefficients)
    return np.hstack(format_packed)


def real_trig_to_exp(trig_coeffs: typing.Union[list, np.array]):
    """

    :param trig_coeffs:
    :return:
    """
    half = int(len(trig_coeffs) // 2 + 1)
    trig_offset = trig_coeffs[0]
    cosine_coefficients = trig_coeffs[1:half]
    sine_coefficients = trig_coeffs[half:]

    complex_offset = 0.5 * trig_offset
    complex_coeffs = 0.5 * (cosine_coefficients - sine_coefficients * (0 + 1j))

    packed = (complex_offset, complex_coeffs)
    return np.hstack(packed)


def real_sig_to_trig(signal):
    exps = np.fft.rfft(signal) / len(signal)
    return real_exp_to_trig(exps)


def _parse_dataset(dset):
    """ Parse the input dataset as a time series.
    :param dset: the target dataset
    :return: the output and input resulting dsets.
    """

    input = []
    output = []
    dataset = dset['x']
    for i in range(0, int(len(dataset))):
        # v = i-th row of dataset
        v = list(dset['x'][i])
        # f = part of row containing river data
        f = v[36 * 7:]
        tot_days = []
        for day in range(0, 7):
            # split weather data according to window size ( 6+1=7 )
            meteo_tot = [v[i * 7 + day] for i in range(0, 36)]
            # append each daily river height separately into different windows.
            fiume_daily = f[24 * day:24 * day + 24]
            # fourier_river = list(real_sig_to_trig(fiume_daily))
            # join them together
            tot_days.append(fiume_daily + meteo_tot)

        v_r = list(dset['y'][i])
        fourier_f = None
        for day in range(0, 7):
            meteo_tot = [v_r[i * 7 + day] for i in range(0, 36)]

            tot_days[day] += meteo_tot
            f = v_r[37 * 7:]
            # fourier_f = real_sig_to_trig(f)

        # No need to separate final river height as its the target already.
        target_fiumi = f
        output.append(np.array(target_fiumi))
        input.append(np.array(tot_days))

    return output, input


if __name__ == '__main__':

    with np.load(Config.EXAMPLESROOT + '/River Height/RnnTestData.npz') as test:
        with np.load(Config.EXAMPLESROOT + '/River Height/RnnTestDataTest.npz') as test_ifile:
            print("> Begin treatment...")

            TEST_OUTPUT, TEST_INPUT = _parse_dataset(test_ifile)
            print("Test result sizes: ", np.array(TEST_OUTPUT).shape, np.array(TEST_INPUT).shape)
            OUTPUT, INPUT = _parse_dataset(test)
            print("Training result sizes: ", np.array(OUTPUT).shape, np.array(INPUT).shape)
            input_ds = tf.convert_to_tensor(INPUT)
            output_ds = tf.convert_to_tensor(OUTPUT)

            # Make dataset.
            input_test_ds = tf.convert_to_tensor(TEST_INPUT)
            output_test_ds = tf.convert_to_tensor(TEST_OUTPUT)

            train_ds = tf.data.Dataset.from_tensor_slices(
                (input_ds, output_ds)).batch(32).shuffle(500)

            testing_ds = tf.data.Dataset.from_tensor_slices(
                (input_test_ds, output_test_ds)).batch(32).shuffle(500)

            model: tf.keras.Model = tf.keras.saving.load_model(Config.EXAMPLESROOT+'/River Height/Rnnmodel4')

            test_error = np.sqrt(model.evaluate(input_test_ds, output_test_ds)[0])*34.20

            p_2 = []
            real_train = []
            for i in range(0, 4):
                x_train = np.expand_dims(input_ds[+i * 8], 0)
                p_2.append(model.predict(x_train)[0] * 34.19282210002337 + 29.672148346052342
                           )
                real_train.append(output_ds[i * 8]  * 34.19282210002337 + 29.672148346052342
                                  )

            plt.plot(np.hstack(p_2), label='Predicted', linewidth=0.3)
            plt.plot(np.hstack(real_train), label='Ground truth', linewidth=0.3)
            plt.legend()
            plt.title(f'RNN training reconstructed result with final test error:{test_error}.')
            plt.show()

            p_2 = []
            real_train = []
            for i in range(0, 15):
                x_train = np.expand_dims(input_test_ds[+i * 8], 0)
                p_2.append(model.predict(x_train)[0]  * 34.19282210002337 + 29.672148346052342
                           )
                real_train.append(output_test_ds[i * 8]  * 34.19282210002337 + 29.672148346052342
                                  )

            plt.plot(np.hstack(p_2), label='Predicted', linewidth=0.3)
            plt.plot(np.hstack(real_train), label='Ground truth', linewidth=0.3)
            plt.legend()
            plt.title(f'RNN test reconstructed result with final test error: {test_error}.')
            plt.show()


            """
            make_rep_layer = tf.keras.layers.LSTM(
                units=351,
                input_shape=(7, 96),
                return_sequences=True,
                recurrent_dropout=0.28,
                unroll=True)

            make_pred_layer = tf.keras.layers.LSTM(
                units=350,
                input_shape=(7, METEO_SHAPE),
                return_sequences=True)
            model = tf.keras.Sequential(
                [make_rep_layer,  # Rnn
                 tf.keras.layers.Flatten(),  # Flatten all 7 days.
                 tf.keras.layers.Dense(1024, activation='relu'),
                 tf.keras.layers.Dropout(0.3),
                 tf.keras.layers.Dense(1024, activation='tanh'),
                 tf.keras.layers.Dropout(0.3),
                 tf.keras.layers.Dense(1024, activation='tanh'),
                 tf.keras.layers.Dense(168)]
            )

            model.summary()
            opt = tf.keras.optimizers.experimental.SGD(
                learning_rate=0.0115,
                momentum=0.019,
                nesterov=True,
                weight_decay=0.0013,
                clipnorm=None,
                clipvalue=None,
                global_clipnorm=None,
                use_ema=False,
                ema_momentum=0.99,
                ema_overwrite_frequency=None,
                jit_compile=True,
                name='SGD',
            )
            model.compile(optimizer=opt, loss=tf.keras.losses.MeanSquaredError(),
                          metrics=['mean_squared_error'])
            print("Model creation completed.")

            test_losses = []
            hist = model.fit(train_ds, epochs=500,
                             callbacks=CustomCallback(
                                 [input_test_ds, output_test_ds], test_losses, model)
                             )

            p_2 = []
            real_train = []
            for i in range(0, 100):
                x_train = np.expand_dims(input_ds[+i * 8], 0)
                p_2.append(model.predict(x_train)[0]  # * 34.19282210002337 + 29.672148346052342
                           )
                real_train.append(output_ds[i * 8]  # * 34.19282210002337 + 29.672148346052342
                                  )

            plt.plot(np.hstack(p_2), label='Predicted', linewidth=0.3)
            plt.plot(np.hstack(real_train), label='Ground truth', linewidth=0.3)
            plt.legend()
            plt.title('RNN training result.')
            plt.show()

            p_2 = []
            real_train = []
            for i in range(0, 15):
                x_train = np.expand_dims(input_test_ds[+i * 8], 0)
                p_2.append(model.predict(x_train)[0]  # * 34.19282210002337 + 29.672148346052342
                           )

                real_train.append(output_test_ds[i * 8]  # * 34.19282210002337 + 29.672148346052342
                                  )

            plt.plot(np.hstack(p_2), label='Predicted', linewidth=0.3)
            plt.plot(np.hstack(real_train), label='Ground truth', linewidth=0.3)
            plt.legend()
            plt.title('RNN test result.')
            plt.show()

            plt.plot(np.hstack([output_ds[k] for k in range(0, len(input_ds), 7)]), label='Input dataset')
            plt.legend()
            plt.show()
            plt.plot(np.hstack([output_test_ds[k] for k in range(0, len(output_test_ds), 7)]), label='Output dataset')
            plt.legend()
            plt.show()
            tf.keras.saving.save_model(model, filepath=Config.EXAMPLESROOT
                                                       + '/River Height/Rnnmodel4')

            with open(Config.EXAMPLESROOT + '/River Height/Model_history.txt', 'a') as savehist:
                savehist.write("New rnn test" + '\n')
                savehist.write(str(hist.history['mean_squared_error']) + '\n')
                savehist.write('Losses on test:' + str(test_losses) + '\n')

            # model = tf.keras.saving.load_model(filepath=Config.EXAMPLESROOT
            #                                            +'/River Height/Rnnmodel2')
            p_2 = []
            real_train = []
            for i in range(0, 100):
                x_train = np.expand_dims(input_ds[+i * 8], 0)
                p_2.append(model.predict(x_train)[0] * 34.19282210002337 + 29.672148346052342
                           )
                real_train.append(output_ds[i * 8] * 34.19282210002337 + 29.672148346052342
                                  )

            plt.plot(np.hstack(p_2), label='Predicted (non-zscore)', linewidth=0.3)
            plt.plot(np.hstack(real_train), label='Ground truth', linewidth=0.3)
            plt.legend()
            plt.title('RNN training result.')
            plt.show()
            p_2 = []
            real_train = []
            for i in range(0, 15):
                x_train = np.expand_dims(input_test_ds[+i * 8], 0)
                p_2.append(model.predict(x_train)[0] * 34.19282210002337 + 29.672148346052342
                           )
                real_train.append(output_test_ds[i * 8] * 34.19282210002337 + 29.672148346052342
                                  )

            plt.plot(np.hstack(p_2), label='Predicted (no-zscore)', linewidth=0.3)
            plt.plot(np.hstack(real_train), label='Ground truth', linewidth=0.3)
            plt.legend()
            plt.title('RNN test result.')
            plt.show()
        """
        """    VectorRH = z_score(VectorRH)
    PNoveC = z_score(PNoveC)
    PZeroC = z_score(PZeroC)
    TempMaxC = z_score(TempMaxC)
    TempMinC = z_score(TempMinC)
    TempAvgC = z_score(TempAvgC)
    
    PNoveCa = z_score(PNoveCa)
    PZeroCa = z_score(PZeroCa)
    TempAvgCa = z_score(TempAvgCa)
    TempMaxCa = z_score(TempMaxCa)
    TempMinCa = z_score(TempMinCa)
    
    PNoveLo = z_score(PNoveLo)
    PZeroLo = z_score(PZeroLo)
    TMediaLo = z_score(TMediaLo)
    TMaxLo = z_score(TMaxLo)
    TMinLo = z_score(TMinLo)
    VelLo = z_score(VelLo)
    RafLo = z_score(RafLo)
    DurLo = z_score(DurLo)
    TempLo = z_score(TempLo)
    
    PNoveRi = z_score(PNoveRi)
    PZeroRi = z_score(PZeroRi)
    TempAvgRi = z_score(TempAvgRi)
    TempMaxRi = z_score(TempMaxRi)
    TempMinRi = z_score(TempMinRi)
    
    PNoveBo = z_score(PNoveBo)
    PZeroBo = z_score(PZeroBo)
    NeveBo = z_score(NeveBo)
    NeveSBo = z_score(NeveSBo)
    NeveAltBo = z_score(NeveAltBo)
    TempAvgBo = z_score(TempAvgBo)
    TempMaxBo = z_score(TempMaxBo)
    TempMinBo = z_score(TempMinBo)
    VelMediaBo = z_score(VelMediaBo)
    RafficaBo = z_score(RafficaBo)
    DurataBo = z_score(DurataBo)
    RadBo = z_score(RadBo)
    """
