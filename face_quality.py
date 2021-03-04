import tensorflow as tf
from tensorflow.keras import Model, Input, layers, losses, optimizers
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import os
import time
from tqdm import tqdm

'''constants'''

IMAGE_SIZE = 256
TRAIN_SIZE, TEST_SIZE = 3200, 800

'''loads labels'''

# TRAIN_LABELS = np.loadtxt("dataset/train/probs.csv", delimiter=",").astype(np.float32)
# TEST_LABELS = np.loadtxt("dataset/test/probs.csv", delimiter=",").astype(np.float32)

'''helper functions'''

#displays time as h:mm:ss
def format_time(seconds):
    return "{}:{:0>2}:{:0>2}".format(int(seconds//3600), int((seconds//60)%60), int(seconds%60))

#loads a batch from the dataset into memory
def get_batch(batch_size):
    indices = np.random.randint(TRAIN_SIZE, size=batch_size)

    images = np.array(Image.open("dataset/train/{}.jpg".format(indices[0])))
    images = np.expand_dims(images, axis=0)

    for index in indices[1:]:
        filename = "dataset/train/{}.jpg".format(index)
        image = Image.open(filename)
        image = np.array(image)
        image = np.expand_dims(image, axis=0)
        images = np.append(images, image, axis=0)

    return images.astype(np.float32)/255.0, TRAIN_LABELS[indices].reshape(batch_size,1)

#evaluates model
def evaluate(model):
    sum_error = 0
    for i in range(TEST_SIZE):
        image = np.expand_dims(np.array(Image.open(f"dataset/test/{i}.jpg")), axis=0)
        label = TEST_LABELS[i]
        logit = model(image)[0][0]
        error = abs(label - logit)
        sum_error += error
    print(f"Mean Absolute Error on test set: {sum_error/TEST_SIZE}\n")
    return sum_error/TEST_SIZE

'''model'''

#resnet
def residual_module(layer_in, n_filters):
    merge_input = layer_in
    # check if the number of filters needs to be increase, assumes channels last format
    if layer_in.shape[-1] != n_filters:
      merge_input = layers.Conv2D(n_filters, (1,1), padding='same', activation='relu', kernel_initializer='he_normal')(layer_in)
    # conv1
    conv1 = layers.Conv2D(n_filters, (3,3), padding='same', activation='relu', kernel_initializer='he_normal')(layer_in)
    # conv2
    conv2 = layers.Conv2D(n_filters, (3,3), padding='same', activation='linear', kernel_initializer='he_normal')(conv1)
    # add filters, assumes filters/channels last
    layer_out = layers.add([conv2, merge_input])
    # activation function
    layer_out = layers.Activation('relu')(layer_out)
    return layer_out

def build_model(base_num_filters=4):
    x = Input([256, 256, 3])
    y = residual_module(x, base_num_filters)
    y = layers.AveragePooling2D()(y) #128

    y = residual_module(y, 2*base_num_filters)
    y = layers.AveragePooling2D()(y) #64

    y = residual_module(y, 3*base_num_filters)
    y = layers.AveragePooling2D()(y) #32

    y = residual_module(y, 4*base_num_filters)
    y = layers.AveragePooling2D()(y) #16

    y = residual_module(y, 6*base_num_filters)
    y = layers.AveragePooling2D()(y) #8

    y = residual_module(y, 8*base_num_filters)
    y = layers.AveragePooling2D()(y) #4

    y = layers.Flatten()(y)
    # y = layers.Dense(8*base_num_filters, kernel_initializer = 'he_normal')(y)
    # y = layers.LeakyReLU(0.2)(y)
    y = layers.Dense(1, activation=tf.nn.sigmoid, kernel_initializer = 'he_normal')(y)
    return Model(inputs=x, outputs=y)

'''training'''

#train step
@tf.function
def train_step(model, optimizer, images, labels):
    with tf.GradientTape() as tape:
        logits = model(images)
        loss = losses.mean_squared_error(y_true=labels, y_pred=logits)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

#training loop (num_iterations has to be a multiple of steps or it will be truncated)
def train(model, batch_size=32, num_epochs=20, steps=100, learning_rate=5e-5):
    num_iterations = int(TRAIN_SIZE * num_epochs / batch_size)
    optimizer = optimizers.Adam(lr=learning_rate)
    loss_history = []
    prev_time = time.time()
    time_elapsed = 0
    print("Training...")

    for i in range(0, num_iterations, steps):
        for j in tqdm(range(steps)):
            x, y = get_batch(batch_size)
    
            loss = train_step(model, optimizer, x, y)
            loss_history.append(loss.numpy().mean())
    
            time_elapsed += time.time() - prev_time
            prev_time = time.time()

        print("\nIteration {}/{}. Time elapsed: {}. Loss: {}\n".format(
            i + steps, num_iterations, format_time(time_elapsed), loss.numpy().mean()))
        evaluate(model)

        #save checkpoints
        model.save_weights(f"./models/{i + steps}.h5")
        
        # plot a graph that will show how our loss varied with time
        plt.plot(loss_history)
        plt.title("Training Progress")
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.savefig(os.path.join("./plots", "Training Progress"))
        #plt.show()
        plt.close()

# model = build_model()
# train(model)

#converts model to tensorflow lite
def convert_to_tflite(model, filename_to_write):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    output = converter.convert()
    open("{}.tflite".format(filename_to_write), "wb").write(output)

#gets an inference from the tensorflow lite model
def infer_tflite(lite_model_path, inputs):
    time_elapsed = time.time()
    interpreter = tf.lite.Interpreter(model_path=lite_model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    for index,item in enumerate(inputs):
        interpreter.set_tensor(input_details[index]['index'], item)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    time_elapsed = time.time() - time_elapsed
    print(time_elapsed)
    return output

if __name__ == "__main__":
    # resnet = build_model()
    # resnet.load_weights("models/resnet.h5")
    # convert_to_tflite(resnet, "resnet-opt")
    infer_tflite("resnet.tflite", [np.zeros([1,256,256,3], dtype=np.float32)])
