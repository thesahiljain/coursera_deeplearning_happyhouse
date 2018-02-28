# All the imports
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import MaxPooling2D
from keras.models import Model
import keras.backend as K
K.set_image_data_format('channels_last')
from kt_utils import load_dataset

X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

# Normalize image vectors
X_train = X_train_orig/255.
X_test = X_test_orig/255.

# Reshape
Y_train = Y_train_orig.T
Y_test = Y_test_orig.T

print ("number of training examples = " + str(X_train.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))

# Create Model


def create_model(input_shape):

    # Input image place_holder
    X_input = Input(input_shape)

    # Add padding to input
    X = ZeroPadding2D((3, 3))(X_input)

    # Layer 1 - convolution -> normalization -> relu
    # Conv2D parameters (number of filters, filter dimensions, strides, name)
    X = Conv2D(32, (7, 7), strides=(1, 1), name='conv0')(X)
    X = BatchNormalization(axis=3, name='bn0')(X)
    X = Activation('relu')(X)

    # Layer 2 - MaxPool
    X = MaxPooling2D((2, 2), name='max_pool')(X)

    # Flatten X
    X = Flatten()(X)

    # Create a fully connected layer
    X = Dense(1, activation='sigmoid', name='fc')(X)

    # Create the model
    model = Model(inputs=X_input, outputs=X, name='HappyModel')

    return model

model = create_model(X_train.shape[1:])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x=X_train, y=Y_train, epochs=40, batch_size=50)

# Test set accuracy

preds = model.evaluate(x=X_test, y=Y_test)

print()
print("Loss = " + str(preds[0]))
print("Test Accuracy = " + str(preds[1]))
