from models import neuralNets.neuralNets


class Trainer:

    def __init__(self, ):
        opt = keras.optimizers.adam_v2.Adam(learning_rate = 0.001)
        model.compile(optimizer= opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.fit(dataloader.reshape_func(train_total[0], channels = 'last'), train_total[1],
                 validation_data = (dataloader.reshape_func(test_total[0], channels = 'last'), test_total[1]),
                 batch_size = 128,
                 epochs = 500)