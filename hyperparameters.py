img_size = 256

batch_size = 32

learning_rate = 0.001

num_epochs = 50

train_length = 328500
test_length = 36500
steps_per_epoch = (train_length // batch_size) // 2
validation_steps = (test_length // batch_size) // 2
# steps_per_epoch = 100
# validation_steps = 100

max_num_weights = 10
