import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import Config
import time
from mpl_toolkits.axes_grid1 import make_axes_locatable


class SeqToSeq(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.make_repr_layer = tf.keras.layers.LSTM(
            units=356,
            input_shape=(7, 60),
            return_state=True,
            recurrent_dropout=0.5,
            unroll=True
        )

        self.decoder_layer = tf.keras.layers.LSTM(
            units=356,
            input_shape=(7, 36),
            return_state=False,
            return_sequences=True,
            recurrent_dropout=0.15,
            unroll=True
        )

        self.flattener = tf.keras.layers.Flatten()
        self.flat_dropout = tf.keras.layers.Dropout(0.4)
        self.map_layer = tf.keras.layers.Dense(
            units=768, activation='relu'
        )
        self.inner_dropout = tf.keras.layers.Dropout(0.4)
        self.first_deep_layer = tf.keras.layers.Dense(
            units=768, activation='tanh'
        )
        self.deep_dropout = tf.keras.layers.Dropout(0.4)
        self.deep_layer = tf.keras.layers.Dense(
            units=768, activation='tanh'
        )
        self.output_layer = tf.keras.layers.Dense(
            units=168
        )

    @tf.function
    def call(self, inputs, train=None, mask=None):
        x_batch = inputs[0]
        m_batch = inputs[1]

        _, s_h, s_c = self.make_repr_layer(x_batch, training=train)
        states = [s_h, s_c]
        decoded = self.decoder_layer(m_batch, initial_state=states, training=train)
        f = self.flattener(decoded)
        f = self.flat_dropout(f, training=train)
        m = self.map_layer(f)
        m = self.deep_dropout(m, training=train)
        deep = self.first_deep_layer(m, training=train)
        deep = self.inner_dropout(deep, training=train)
        deep = self.deep_layer(deep, training=train)
        out = self.output_layer(deep)
        return out


class Encoder(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.make_repr_layer = tf.keras.layers.LSTM(
            units=256,
            input_shape=(7, 60),
            return_state=True,
            recurrent_dropout=0.28,
            unroll=True
        )

    @tf.function
    def call(self, inputs, train=None, mask=None):
        meaningful_rep = self.make_repr_layer(inputs)
        return meaningful_rep


class FullyConnectedChannels(tf.keras.Model):

    def __init__(self):
        super().__init__()
        self.flattener = tf.keras.layers.Flatten()
        self.flat_dropout = tf.keras.layers.Dropout(0.4)
        self.map_layer = tf.keras.layers.Dense(
            units=1024, activation='relu'
        )
        self.inner_dropout = tf.keras.layers.Dropout(0.4)
        self.deep_layer = tf.keras.layers.Dense(
            units=1024, activation='tanh'
        )
        self.output_layer = tf.keras.layers.Dense(
            units=168
        )

    @tf.function
    def call(self, inputs, train=None, mask=None):
        f = self.flattener(inputs)
        if train:
            f = self.flat_dropout(f)
        m = self.map_layer(f)
        if train:
            m = self.inner_dropout(m)
        deep = self.deep_layer(m)
        out = self.output_layer(deep)
        return out


class Decoder(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.decoder_layer = tf.keras.layers.LSTM(
            units=256,
            input_shape=(7, 36),
            return_state=False,
            return_sequences=True,
            recurrent_dropout=0.35,
            unroll=True
        )

    @tf.function
    def call(self, inputs, train=None, mask=None):
        states = [inputs[1], inputs[2]]
        m = inputs[0]
        decoded = self.decoder_layer(m, initial_state=states)
        return decoded


def _parse_dataset(dset):
    """ Parse the input dataset as a time series.
    :param dset: the target dataset
    :return: the output and input resulting dsets.
    """

    input_data = []
    expected_data = []
    meteos = []
    dataset = dset['x']
    for i in range(0, int(len(dataset))):
        # v = i-th row of dataset
        x_row = list(dset['x'][i])
        f = x_row[36 * 7:]
        tot_days = []
        for day in range(0, 7):
            # split weather data according to window size ( 6+1=7 )
            meteo_tot = [x_row[i * 7 + day] for i in range(0, 36)]
            # append each daily river height separately into different windows.
            fiume_daily = f[24 * day:24 * day + 24]
            # fourier_river = list(real_sig_to_trig(fiume_daily))
            # join them together
            tot_days.append(fiume_daily + meteo_tot)

        y_row = list(dset['y'][i])
        weekly_meteos = []
        for day in range(0, 7):
            meteo_tot = [y_row[i * 7 + day] for i in range(0, 36)]
            weekly_meteos.append(meteo_tot)
        meteos.append(weekly_meteos)
        f = y_row[37 * 7:]

        # No need to separate final river height as its the target already.
        expected_data.append(np.array(f))
        input_data.append(np.array(tot_days))

    return expected_data, meteos, input_data


load_model = True

if __name__ == '__main__' and not load_model:

    decoder = Decoder()
    encoder = Encoder()
    predictor = FullyConnectedChannels()

    seq_to_seq = SeqToSeq()

    # Load test file..
    # Load training file..
    # Attempt some data augmentation!
    with np.load(Config.EXAMPLESROOT + '/River Height/RnnTestData.npz') as training:
        print("Begin training data parsing...")
        raw_output, raw_decode, raw_xs = _parse_dataset(training)

    with np.load(Config.EXAMPLESROOT + '/River Height/RnnTestDataTest.npz') as test:
        print("Begin test data parsing...")
        raw_test_output, raw_test_decode, raw_test_xs = _parse_dataset(test)

    output = tf.convert_to_tensor(raw_output)
    test_output = tf.convert_to_tensor(raw_test_output)

    test_decode = tf.convert_to_tensor(raw_test_decode)
    decode = tf.convert_to_tensor(raw_decode)

    xs = tf.convert_to_tensor(raw_xs)
    test_xs = tf.convert_to_tensor(raw_test_xs)

    print("Training output_shape: ",
          np.array(raw_output).shape,
          np.array(raw_xs).shape,
          np.array(raw_decode).shape)

    print("Test output_shape: ",
          np.array(raw_test_output).shape,
          np.array(raw_test_xs).shape,
          np.array(raw_test_decode).shape)

    # Begin training parameters

    opt = tf.keras.optimizers.legacy.SGD(
        learning_rate=0.025,
        momentum=0.019,
        nesterov=True,
    )
    loss_func = tf.keras.losses.MeanSquaredError()
    epochs = 500

    new_opt = tf.keras.optimizers.SGD(
        learning_rate=0.025,
        momentum=0.008,
        nesterov=True,
        weight_decay=0.004
    )
    seq_to_seq.compile(optimizer=new_opt,
                       loss=loss_func,
                       metrics=['mean_squared_error']
                       )
    encoder.compile(optimizer=opt,
                    loss=loss_func,
                    metrics=['mean_squared_error'])
    decoder.compile(optimizer=opt,
                    loss=loss_func,
                    metrics=['mean_squared_error'])
    predictor.compile(optimizer=opt,
                      loss=loss_func,
                      metrics=['mean_squared_error'])

    training_dataset = tf.data.Dataset.from_tensor_slices(
        (xs, decode, output)).shuffle(1024).batch(64)

    test_losses = []
    training_losses = []
    for epoch in range(epochs):
        print(f"Beginning of epoch {epoch}: ")
        epoch_beginning = time.perf_counter()
        for b_num, (x, d, o) in enumerate(training_dataset):
            with tf.GradientTape() as tape:
                days_vector = seq_to_seq(inputs=[x, d], train=True)
                loss = loss_func(o, days_vector)

            grad_1 = tape.gradient(loss, seq_to_seq.trainable_weights)
            opt.apply_gradients(zip(grad_1, seq_to_seq.trainable_weights))

        if epoch == 100:
            new_opt.learning_rate = new_opt.learning_rate / 2
        elif epoch % 50 == 0:
            new_opt.learning_rate = new_opt.learning_rate / 1.414
            print(f"** Current learning rate: {new_opt.learning_rate}")
        elapsed = time.perf_counter() - epoch_beginning
        print(f"Terminated all batches. Elapsed seconds: {elapsed:0.4f}. "
              f"Epoch {epoch}/{epochs} completed. ")

        train_err = loss_func(seq_to_seq([xs, decode], training=False), output)
        test_err = loss_func(seq_to_seq([test_xs, test_decode]), test_output)
        training_losses.append(train_err.numpy())
        test_losses.append(test_err.numpy())

        print(f"Total error on training: {train_err}. Total error on "
              f"the test set {test_err}")

    save = True
    if save:
        architecture_path = Config.EXAMPLESROOT + '/River Height/Architecture/'
        tf.keras.saving.save_model(seq_to_seq, architecture_path + 'seq_to_seq4')

    """
    @tf.function
    def _eval_pipeline(x_p, d_p):
        o_p, h, c = encoder(x_p)
        h_d = decoder([d_p, h, c])
        a = predictor(h_d)
        return a


    test_losses = []
    training_losses = []
    for epoch in range(epochs):
        print(f"Beginning of epoch {epoch}: ")
        epoch_beginning = time.perf_counter()
        for b_num, (x, d, o) in enumerate(training_dataset):
            
            with tf.GradientTape(persistent=True) as tape:
                _, state_h, state_c = encoder(x, train=True)
                high_dim_days = decoder([d, state_h, state_c], train=True)
                days_vector = predictor(high_dim_days, train=True)

                loss = loss_func(o, days_vector)

            grad_1 = tape.gradient(loss, encoder.trainable_weights)
            grad_2 = tape.gradient(loss, decoder.trainable_weights)
            grad_3 = tape.gradient(loss, predictor.trainable_weights)

            opt.apply_gradients(zip(grad_1, encoder.trainable_weights))
            opt.apply_gradients(zip(grad_2, decoder.trainable_weights))
            opt.apply_gradients(zip(grad_3, predictor.trainable_weights))

        elapsed = time.perf_counter() - epoch_beginning
        print(f"Terminated all batches. Elapsed seconds: {elapsed:0.4f}. "
              f"Epoch {epoch}/{epochs} completed. ")

        train_err = loss_func(_eval_pipeline(xs, decode), output)
        test_err = loss_func(_eval_pipeline(test_xs, test_decode), test_output)
        training_losses.append(train_err)
        test_losses.append(test_err)

        print(f"Total error on training: {train_err}. Total error on "
              f"the test set {test_err}")

    save = True
    if save:
        w_1 = decoder.get_weights()
        print("Length of decoder weights: ", len(w_1))
        weights = {f'w_{l}': k for k, l in zip(w_1, range(len(w_1)))}

        architecture_path = Config.EXAMPLESROOT + '/River Height/Architecture/'
        np.savez(architecture_path + 'decoder_weights', **weights)
        tf.keras.saving.save_model(decoder, architecture_path + 'decoder_model')
        tf.keras.saving.save_model(predictor, architecture_path + 'predictor_model')
        tf.keras.saving.save_model(encoder, architecture_path + 'encoder_model')
    
    """

    with open(Config.EXAMPLESROOT + '/River Height/Model_history.txt', 'a') as hist_file:
        hist_file.write("\nLatest architecture results:\n")
        hist_file.write(str(training_losses) + '\n')
        hist_file.write(str(test_losses) + '\n')

if __name__ == '__main__' and load_model:
    save_path = Config.EXAMPLESROOT + '/River Height/Architecture/'

    decoder_model = tf.keras.saving.load_model(save_path + 'decoder_model')
    encoder_model = tf.keras.saving.load_model(save_path + 'encoder_model')
    predictor_model = tf.keras.saving.load_model(save_path + 'predictor_model')

    complete_model = tf.keras.saving.load_model(save_path + 'seq_to_seq4')


    @tf.function
    def _eval_overall_model(x_p, m_p):
        o_p, h, c = encoder_model(x_p)
        h_d = decoder_model([m_p, h, c])
        a = predictor_model(h_d)
        return a


    with np.load(Config.EXAMPLESROOT + '/River Height/RnnTestDataTest.npz') as training:
        print("Begin training data parsing...")
        raw_output, raw_decode, raw_xs = _parse_dataset(training)

    output = tf.convert_to_tensor(raw_output)
    decode = tf.convert_to_tensor(raw_decode)
    xs = tf.convert_to_tensor(raw_xs)

    print("Completed parsing data.")
    mse_loss = tf.keras.losses.MeanSquaredError()

    print("Begin computation of loss...")
    print("Loss on training: ", mse_loss(complete_model([xs, decode]), output))


    def _plot_prediction(pred, truth, title: str, stack=None, labels=None):

        pred, truth = (
            np.hstack(pred) if stack and stack[0] else pred,
            np.hstack(truth) if stack and stack[1] else truth
        )

        plt.plot(pred, label=labels[0] if labels else 'Predicted', linewidth=0.3)
        plt.plot(truth, label=labels[0] if labels else 'Ground truth', linewidth=0.3)
        amt = pred / 168
        plt.vlines(x=[0, 168],
                   ymin=-1, ymax=1, colors='teal', ls='--', lw=2, linewidth=0.2)
        plt.legend()
        plt.title(title)
        plt.show()


    prediction = []
    ground_training_truth = []
    for i in range(21, 27):
        x_train = np.expand_dims(xs[+i * 7], 0)
        m_train = np.expand_dims(decode[+i * 7], 0)
        prediction.append(
            complete_model([x_train, m_train])[0]  # * 34.193 + 29.672
        )
        ground_training_truth.append(
            output[i * 7]  # * 34.193 + 29.672
        )

    _plot_prediction \
        (prediction,
         ground_training_truth,
         'Training predictions for new architecture',
         [True, True]
         )


    def _make_spectrum():
        ground_truth = output
        predicted = complete_model([xs, decode])

        splits = 128
        months_truth = np.array_split(ground_truth, splits)
        months_pred = np.array_split(predicted, splits)

        months_uncertainties = []
        for m_t, m_p in zip(months_truth, months_pred):
            hour_uncertainties = []
            for hour in range(0, 168):
                same_hour_for_all_days_in_predicted_month = \
                    [
                        week[hour] for week in m_p
                    ]
                same_hour_for_all_days_in_true_month = \
                    [
                        week[hour] for week in m_t
                    ]

                loss_for_hour = mse_loss(
                    same_hour_for_all_days_in_true_month,
                    same_hour_for_all_days_in_predicted_month
                )
                hour_uncertainties.append(loss_for_hour)

            months_uncertainties.append(hour_uncertainties)

        ax = plt.subplot()
        im = ax.imshow(np.array(months_uncertainties),
                       cmap='Spectral',
                       interpolation='nearest',
                       )

        # Put Color bar ont he right
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        ax.set_xticks(range(0, 168, 24))
        ax.set_xticklabels(range(0, 168, 24))
        ax.set(title='Rolling Windowed Time Lagged Cross Correlation',
               xlim=[0, 168],
               xlabel='Offset',
               ylabel='Epochs')
        plt.colorbar(im, cax=cax)
        plt.show()

    _make_spectrum()
