import numpy as np
import Config
import tensorflow as tf
import matplotlib.pyplot as plt

__licence__ = 'Comune di Val Sesia, agenzia federale'
__version__ = '0.1'
__dependencies__ = ['numpy 1.20', 'tensorflow .', 'matplotlib .']

if __name__ == '__main__':

    make_file = Config.EXAMPLESROOT + "/River Height/savefile.npz"
    test_file = Config.EXAMPLESROOT + "/River Height/test.npz"
    with np.load(make_file) as model_data:
        with np.load(test_file) as test_data:
            test_err = []
            train_err = []
            test_err2 = []
            for ep_amount in range(10, 700, 5):

                # print("> Training set:", model_data['x'].shape, model_data['y'].shape)
                # print("> Test set:", test_data['x'].shape, test_data['y'].shape)

                x = tf.constant([[1, 2, 3], [1,2, 3]])

                var = tf.Variable([1, 2, 3])
                var.assign([1, 4, 3])

                model = tf.keras.Sequential(
                    [
                        tf.keras.layers.InputLayer(input_shape=(model_data['x'].shape[1],)),
                        tf.keras.layers.Dense(100, activation='relu'),
                        tf.keras.layers.Dropout(rate=0.1),
                        tf.keras.layers.Dense(100, activation="relu"),
                        tf.keras.layers.Dropout(rate=0.05),
                        tf.keras.layers.Dense(100, activation="relu"),
                        tf.keras.layers.Dense(model_data['y'].shape[1]),
                    ]
                )

                train_data = model_data['x']
                train_label = model_data['y']
                train_ds = tf.data.Dataset.from_tensor_slices(
                    (train_data, train_label)).shuffle(10000).batch(32)

                # print(train_ds.cardinality())
                # print(train_ds)
                model.summary()
                model.compile(optimizer='adam',
                              loss=tf.keras.losses.MeanSquaredError(),
                              metrics=['mean_squared_error'])

                hist = model.fit(train_ds, epochs=ep_amount)
                # print(hist.history.keys())

                x_test = test_data['x']
                y_test = test_data['y']
                losses = model.evaluate(x_test, y_test, batch_size=32)
                # print(losses)

                test_err.append(losses[0]-float(hist.history['mean_squared_error'][-1]))
                train_err.append(hist.history['mean_squared_error'][-1])
                test_err2.append(losses[0])
                # print(test_err)

            g_truth = []
            p = []
            for i in range(0, len(test_data['y'])):

                x_test = np.expand_dims(test_data['x'][i],0)
                p.append(model.predict(x_test)[0])
                g_truth.append(test_data['y'][i])

            plt.plot(np.arange(0, len(np.hstack(g_truth))), np.hstack(g_truth), label='ground truth')

            plt.plot(np.arange(0, len(np.hstack(p))), np.hstack(p),  label='predicted')
            plt.legend()
            plt.show()

            print(test_err)
            plt.figure()

            plt.plot( np.arange(0, len(train_err))*5, train_err,label='Error at final epoch for training set')
            plt.plot(np.arange(0, len(train_err))*5,test_err2, label='Error at final epoch for testing set')
            plt.plot( np.arange(0, len(train_err))*5,test_err,label='Difference between test and training error')
            plt.legend()
            plt.show()