import tensorflow as tf
from tensorflow.keras import Model, Input, Sequential, layers, optimizers, backend as K
import numpy as np
from matplotlib import pyplot as plt
import cv2
from PIL import Image
from tqdm import tqdm
import os
import time
from random import random, randint
import uuid


'''constants'''

DATASET_PATH = "dataset/celeba/"
WEIGHTS_PATH = "models/celeba/48 filters/230k"
IMAGE_SIZE = 256
LATENT_DIM = 512
DATASET_SIZE = 30000
NUM_BLOCKS = 7
BASE_NUM_FILTERS = 48
MIXED_PROB = 0.9

'''helper functions'''

#displays time as h:mm:ss
def format_time(seconds):
    return "{}:{:0>2}:{:0>2}".format(int(seconds//3600), int((seconds//60)%60), int(seconds%60))

#loads a batch from the dataset into memory
def get_batch(batch_size):
    indices = np.random.randint(DATASET_SIZE, size=batch_size)

    batch = np.array(Image.open("{}{:0>5}.jpg".format(DATASET_PATH, indices[0])))
    batch = np.expand_dims(batch, axis=0)

    for index in indices[1:]:
        filename = "{}{:0>5}.jpg".format(DATASET_PATH, index)
        image = Image.open(filename)
        image = np.array(image)
        image = np.expand_dims(image, axis=0)
        batch = np.append(batch, image, axis=0)

    return batch.astype(np.float32)/255.0  #normalize

#generates latent vectors to be applied to blocks in the generator
#returns list of ndarrays of shape [batch_size,LATENT_DIM]
#each numpy array in the list is applied to a generator block
def create_latent_vectors(batch_size):
    return [np.random.normal(0.0, 1.0, size = [batch_size, LATENT_DIM]).astype('float32')] * NUM_BLOCKS

#mixing regularization - combines two latent vectors
def mix_latent_vectors(first, second, split_point):
    assert 0 <= split_point <= NUM_BLOCKS
    return first[:split_point] + second[split_point:]

#gradient penalty loss
def gradient_penalty(inputs, preds, weights):
    gradients = K.gradients(preds, inputs)[0]   #first layer gradients
    gradients_sqr = K.square(gradients)
    gradients_sqr_norm = K.sum(gradients_sqr, axis=np.arange(1, len(gradients_sqr.shape)))    #add up gradients of each image (every axis other than 0)
    return K.mean(gradients_sqr_norm * weights) #average over batch

#adaptive instance normalization (pass arguments in single array/tuple)
def AdaIN(image_representation_and_scale_and_bias):
    image_representation, scale, bias = image_representation_and_scale_and_bias

    #normalize data to mean of 0 and SD 1
    mean = tf.reduce_mean(image_representation, axis=(-2, -3), keepdims=True)   #average each channel, (256,256,3) -> (1,1,3)
    sd = tf.math.reduce_std(image_representation, axis=(-2, -3), keepdims=True) + 1e-7  #'''''''''''''''
    y = (image_representation - mean) / sd   #subtract mean, divide by sd

    #reshape scale and bias
    pool_shape = [-1, 1, 1, y.shape[-1]]
    scale = tf.reshape(scale, pool_shape)
    bias = tf.reshape(bias, pool_shape)

    #apply new mean and SD
    return y * scale + bias

#creates random noise to apply to image representation
def create_noise(batch_size):
    return np.random.uniform(0.0, 1.0, size = [batch_size, IMAGE_SIZE, IMAGE_SIZE, 1]).astype('float32')

#trims noise to match the image representation at a given point in the generator (pass arguments in single array/tuple)
def crop_noise(noise_and_image_representation):
    noise, image_representation = noise_and_image_representation
    height = width = image_representation.shape[-2] #width is as big as the 2nd last dimension. height and width are same anyways
    return noise[:, :height, :width, :]

'''models'''

#builds generator modules
def g_block(x, num_filters, style_vector, noise, upsample=True):
    y = layers.UpSampling2D(interpolation='bilinear')(x) if upsample else layers.Activation('linear')(x)

    scale = layers.Dense(num_filters)(style_vector) #learns scale for each filter/channel from style vector
    bias = layers.Dense(num_filters)(style_vector)  #learns bias for each filter/channel from style vector

    noise = layers.Lambda(crop_noise)([noise,y])    #trims noise to fit image representation
    noise = layers.Dense(num_filters, kernel_initializer ='zeros')(noise)   #gives each filter its own noise

    y = layers.Conv2D(filters=num_filters, kernel_size=3, padding='same', kernel_initializer='he_normal')(y)
    y = layers.add([y, noise])                      #apply noise
    y = layers.Lambda(AdaIN)([y, scale, bias])      #adapt data to scale and bias learned from style vector
    y = layers.LeakyReLU(0.2)(y)

    return y

#builds discriminator modules
def d_block(x, num_filters, downsample=True):
    y = layers.Conv2D(filters=num_filters, kernel_size=3, padding='same', kernel_initializer='he_normal')(x)
    y = layers.LeakyReLU(0.2)(y)
    y = layers.AveragePooling2D()(y) if downsample else y
    return y

#builds the actual generator
def build_generator(lite=False):
    latent_input = []
    for i in range(NUM_BLOCKS):
        latent_input.append(Input([LATENT_DIM]))

    if lite:   #take flattened buffer as input if model is configured for tensorflow lite
        noise_input = Input([IMAGE_SIZE * IMAGE_SIZE])
        noise = layers.Reshape([IMAGE_SIZE, IMAGE_SIZE, 1])(noise_input)
    else:
        noise_input = Input([IMAGE_SIZE, IMAGE_SIZE, 1])
        noise = noise_input

    mapping_network = Sequential([
        layers.Dense(LATENT_DIM, input_shape=[LATENT_DIM]),
        layers.LeakyReLU(0.2),
        layers.Dense(LATENT_DIM),
        layers.LeakyReLU(0.2),
        layers.Dense(LATENT_DIM),
        layers.LeakyReLU(0.2),
        layers.Dense(LATENT_DIM),
        layers.LeakyReLU(0.2),
    ])  #maps the latent space to an intermediate latent space, what we've been referring to as the 'style vector'

    constant_input = layers.Dense(1, trainable=False)(latent_input[0])      #we don't need to train these weights as we'll set the output to a constant value anyways
    constant_input = layers.Lambda(lambda x: x * 0 + 1)(constant_input)     #we set this to a constant, in this case 1

    #most data has constants
    #say you want to generate faces - every face has eyes, a nose, a mouth, etc...
    #the weights that transform the constant input to the first convolutional layer are learned
    #this makes it such that the generator knows the features consistent in the given data (faces have two eyes, a nose below, ...) before it starts the generation process

    y = layers.Dense(4*4*BASE_NUM_FILTERS*4, activation='relu', kernel_initializer='he_normal')(constant_input)
    y = layers.Reshape([4, 4, BASE_NUM_FILTERS*4])(y)

    y = g_block(y, BASE_NUM_FILTERS*16, mapping_network(latent_input[0]), noise, upsample=False)    #4x4
    y = g_block(y, BASE_NUM_FILTERS*8, mapping_network(latent_input[1]), noise)   #8x8
    y = g_block(y, BASE_NUM_FILTERS*6, mapping_network(latent_input[2]), noise)   #16x16
    y = g_block(y, BASE_NUM_FILTERS*4, mapping_network(latent_input[3]), noise)   #32x32
    y = g_block(y, BASE_NUM_FILTERS*3, mapping_network(latent_input[4]), noise)   #64x64
    y = g_block(y, BASE_NUM_FILTERS*2, mapping_network(latent_input[5]), noise)   #128x128
    y = g_block(y, BASE_NUM_FILTERS, mapping_network(latent_input[6]), noise)     #256x256

    final_image = layers.Conv2D(filters=3, kernel_size=1, padding='same', kernel_initializer='he_normal')(y)
    if lite: #return int32 tensor normalized between 0 and 255 if model is configured for tensorflow lite
        final_image = layers.Lambda(lambda x:tf.clip_by_value(x, 0.0, 1.0))(final_image)
        #final_image = layers.Lambda(lambda x: tf.cast(tf.floor(x * 255), dtype=tf.int32))(final_image)
    return Model(inputs=latent_input + [noise_input], outputs=final_image)

#builds the actual discriminator
def build_discriminator():
    image = Input([IMAGE_SIZE, IMAGE_SIZE, 3])

    y = d_block(image, BASE_NUM_FILTERS)#128x128
    y = d_block(y, 2*BASE_NUM_FILTERS)  #64x64
    y = d_block(y, 3*BASE_NUM_FILTERS)  #32x32
    y = d_block(y, 4*BASE_NUM_FILTERS)  #16x16
    y = d_block(y, 6*BASE_NUM_FILTERS)  #8x8
    y = d_block(y, 8*BASE_NUM_FILTERS)  #4x4
    y = d_block(y, 16*BASE_NUM_FILTERS, downsample=False)  #4x4

    y = layers.Flatten()(y)
    y = layers.Dense(16*BASE_NUM_FILTERS, kernel_initializer = 'he_normal')(y)
    y = layers.LeakyReLU(0.2)(y)
    y = layers.Dense(1, kernel_initializer = 'he_normal')(y)
    return Model(inputs=image, outputs=y)

'''training'''

#train step
@tf.function
def train_step(generator, discriminator, gen_optimizer, disc_optimizer, batch, latent_input, noise_input, gp_weights):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(latent_input + [noise_input])
        real_output = discriminator(batch)
        fake_output = discriminator(generated_images)

        gen_loss = K.mean(fake_output)
        divergence = K.mean(K.relu(1 + real_output) + K.relu(1 - fake_output))
        disc_loss = divergence + gradient_penalty(batch, real_output, gp_weights)

    gen_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
    disc_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    gen_optimizer.apply_gradients(zip(gen_gradients, generator.trainable_variables))
    disc_optimizer.apply_gradients(zip(disc_gradients, discriminator.trainable_variables))

    return gen_loss, disc_loss, divergence

#training loop (num_iterations has to be a multiple of steps or it will be truncated)
def train(generator, discriminator, batch_size=12, num_iterations=int(1e7), steps=1000, learning_rate=1e-4):
    gen_optimizer = optimizers.Adam(lr=learning_rate, beta_1=0, beta_2=0.9)
    disc_optimizer = optimizers.Adam(lr=learning_rate*4, beta_1=0, beta_2=0.9)
    gp_weights = np.load("models/gp_weights.npy") if os.path.isfile("models/gp_weights.npy") else np.array([10.0] * batch_size).astype('float32')
    gen_loss_history = []
    disc_loss_history = []
    prev_time = time.time()
    time_elapsed = 0

    #load saved models or create if they don't exist
    if os.path.isfile("models/generator.h5"):
        generator.load_weights("models/generator.h5")
    else:
        generator.save_weights("models/generator.h5")

    if os.path.isfile("models/discriminator.h5"):
        discriminator.load_weights("models/discriminator.h5")
    else:
        discriminator.save_weights("models/discriminator.h5")

    print("Training...")

    for i in range(0, num_iterations, steps):
        gen_loss, disc_loss = 0, 0
        for j in tqdm(range(steps)):
            batch = get_batch(batch_size)
    
            if random() > MIXED_PROB:
                latent_vectors = create_latent_vectors(batch_size)
            else:
                vector1, vector2 = create_latent_vectors(batch_size), create_latent_vectors(batch_size)
                latent_vectors = mix_latent_vectors(vector1, vector2, randint(0, NUM_BLOCKS))
    
            noise = create_noise(batch_size)
            gen_loss, disc_loss, divergence = train_step(
                generator, discriminator, gen_optimizer, disc_optimizer, batch, latent_vectors, noise, gp_weights
            )
            gen_loss_history.append(np.array(gen_loss).mean())
            disc_loss_history.append(np.array(disc_loss).mean())
    
            # update gradient penalty weights
            new_gp_weights = 5 / (np.array(divergence) + 1e-7)
            gp_weights = gp_weights[0] * 0.9 + new_gp_weights * 0.1
            gp_weights = np.clip([gp_weights] * batch_size, 0.01, 10000.0).astype('float32')
    
            time_elapsed += time.time() - prev_time
            prev_time = time.time()

        print("Iteration {}/{}. Generator loss: {}. Discriminator loss: {}. Time elapsed: {}\n".format(
            i + steps, num_iterations, np.array(gen_loss).mean(), np.array(disc_loss).mean(), format_time(time_elapsed)))

        #save checkpoints
        generator.save_weights("models/generator.h5")
        discriminator.save_weights("models/discriminator.h5")
        np.save("models/gp_weights.npy", gp_weights)
        
        # plot a graph that will show how our loss varied with time
        plt.plot(gen_loss_history, label="generator")
        plt.plot(disc_loss_history, label="discriminator")
        plt.legend(loc="upper right")
        plt.title("Training Progress")
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.savefig(os.path.join("./plots", "Training Progress"))
        #plt.show()
        plt.close()

'''inference'''

#gets an inference from the model
def infer(generator, batch_size=8):
    vector = create_latent_vectors(batch_size)
    noise = create_noise(batch_size)
    inference_time = time.time()
    output = generator(vector + [noise]).numpy()
    #print(f"call has taken {time.time()-inference_time} seconds")

    for image in output:
        plt.figure(figsize=(5.12,5.12))
        plt.axis('off')
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1)
        plt.imshow(image, aspect='equal', interpolation='bicubic')

    plt.show()
    return output

#generates and saves images along with their latent codes
def infer_and_save(generator, folder, size=100):
    #folder = str(uuid.uuid1())
    os.system(f"mkdir saved\{folder}\ ")
    vectors = np.array([[]])
    for i in range(size):
        vector = create_latent_vectors(1)
        noise = create_noise(1)
        output = generator(vector + [noise]).numpy()[0]
        vectors = np.append(vectors, vector[0])
        plt.figure(figsize=(5.12, 5.12))
        plt.axis('off')
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1)
        plt.imshow(output, aspect='equal', interpolation='bicubic')
        plt.savefig(f"./saved/{folder}/{i}.jpg")
    np.savetxt(f"saved/{folder}/vectors.csv", vectors, delimiter=",")

#creates video of images from vectors interpolated through a batch
def interpolate_vectors(generator, batch_of_vectors, fps=30, period=2):
    filename = uuid.uuid1()
    batch_of_vectors = batch_of_vectors[0]    #we pick the vector to be applied to the first block - it doesn't matter as the same vector is applied to all blocks
    num_frames = period * fps
    video = cv2.VideoWriter("interpolations/{}.avi".format(filename), cv2.VideoWriter.fourcc(*'DIVX'), fps, (IMAGE_SIZE,IMAGE_SIZE))
    noise = create_noise(1)

    for first, second in zip(batch_of_vectors, np.vstack([batch_of_vectors[1:], batch_of_vectors[0]])):    #loops through adjacent vectors in the batch in this manner, [v0,v1,v2,v3,v4] -> (v0,v1) (v1,v2) (v2,v3) (v3,v4) (v4,v0)
        for frame in tqdm(range(num_frames)):
            lerp = first + (frame / num_frames) * (second - first)
            lerp = [np.expand_dims(lerp, 0)] * NUM_BLOCKS   #add batch dimension then make copies for all blocks
            output = generator.predict(lerp + [noise])[0]
            output = np.uint8(np.clip(output, 0.0, 1.0) * 255)  #normalize, scale to 0...255, then convert to unsigned int
            Image.fromarray(output, "RGB").save("image.jpg")
            image = cv2.imread("image.jpg")
            video.write(image)

    #write last image skipped in for loop
    output = generator.predict([np.expand_dims(batch_of_vectors[-1], 0)] * NUM_BLOCKS + [noise])[0]
    output = np.uint8(np.clip(output, 0.0, 1.0) * 255)
    Image.fromarray(output, "RGB").save("image.jpg")
    image = cv2.imread("image.jpg")
    video.write(image)

    os.system("del image.jpg")
    video.release()
    print("{}.avi written".format(filename))

#generates images from transitions between two style vectors (from coarse to fine)
def interpolate_styles(generator, batch_size=8):
    outputs = []
    for block in range(NUM_BLOCKS+1):
        vector1, vector2 = create_latent_vectors(batch_size), create_latent_vectors(batch_size)
        mixed_vector = mix_latent_vectors(vector1, vector2, block)
        noise = create_noise(batch_size)
        outputs.append(generator.predict(mixed_vector + [noise]))

    for batch_num in range(batch_size):
        plt.figure(figsize=(NUM_BLOCKS+1, 1))
        fig, ax = plt.subplots(1, NUM_BLOCKS+1)
        for block_num in range(NUM_BLOCKS+1):
            ax[block_num].imshow(outputs[block_num][batch_num])

    plt.show()

#perturbs a specified latent variable
def vary_latent_variable(generator, style, speed = 1, folder_name = str(uuid.uuid1()), index = 0, fps=15, duration=6):
    num_frames = duration * fps
    video = cv2.VideoWriter(f"saved/{folder_name}/{index}.avi", cv2.VideoWriter.fourcc(*'DIVX'), fps, (IMAGE_SIZE, IMAGE_SIZE))
    noise = create_noise(1)
    os.system(f"mkdir saved\{folder_name}\ ")
    for num_frame in range(num_frames):
        new_style = np.copy(style)
        new_style[index] = style[index] + (-speed + 2*speed*num_frame/num_frames)
        output = generator.predict([np.expand_dims(new_style, 0)] * NUM_BLOCKS + [noise])[0]
        output = np.uint8(np.clip(output, 0.0, 1.0) * 255)
        Image.fromarray(output, "RGB").save("image.jpg")
        image = cv2.imread("image.jpg")
        video.write(image)

    os.system("del image.jpg")
    video.release()
    print(f"{folder_name}/{index}.avi written")

#mixes coarse styles from one image and fine styles from another image
def mix_styles(generator, coarse_styles, fine_styles, filepath):
    fig, ax = plt.subplots(len(coarse_styles)+1, len(fine_styles)+1, gridspec_kw={'wspace':0, 'hspace':0})
    fig.subplots_adjust(hspace=0, wspace=0, left=0, bottom=0, right=1, top=1)
    ax[0, 0].axis("off")
    noise = create_noise(1)
    for index, coarse_style in enumerate(coarse_styles):
        output = generator.predict(coarse_style + [noise])[0]
        ax[index+1, 0].imshow(output)
        ax[index + 1, 0].axis("off")
    for index, fine_style in enumerate(fine_styles):
        output = generator.predict(fine_style + [noise])[0]
        ax[0, index+1].imshow(output)
        ax[0, index + 1].axis("off")
    for row, coarse_style in enumerate(coarse_styles):
        for column, fine_style in enumerate(fine_styles):
            style = mix_latent_vectors(coarse_style, fine_style, 2)
            output = generator.predict(style + [noise])[0]
            ax[row+1, column+1].imshow(output)
            ax[row+1, column+1].axis("off")
    plt.savefig(filepath)

#converts model to tensorflow lite
def convert_to_tflite(model, filename_to_write):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # converter.target_spec.supported_types = [tf.float16]
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

'''entry point'''

if __name__ == "__main__":
    generator = build_generator()
    # generator.summary()
    # discriminator = build_discriminator()
    # discriminator.summary()
    # train(generator, discriminator, batch_size=4, num_iterations=100, steps=5)
    generator.load_weights(f"{WEIGHTS_PATH}/generator.h5")
    infer(generator)
    # infer_and_save(generator, "230k")
    # interpolate_styles(generator)
    # for i in range(4):
    #     interpolate_vectors(generator, create_latent_vectors(8), fps=30)

    # styles = np.loadtxt("saved/afe9d069-57e0-11eb-8545-801934d4c19f/vectors.csv", delimiter=",").astype(np.float32)
    # styles = styles.reshape((8, 256))
    # coarse_styles = []
    # fine_styles = []
    # for i in range(4):
    #     coarse_styles.append([np.expand_dims(styles[i], 0)] * NUM_BLOCKS)
    # for i in range(4, 8):
    #     fine_styles.append([np.expand_dims(styles[37], 0)] * NUM_BLOCKS)
    # mix_styles(generator, coarse_styles, fine_styles, "saved/20f40a2d-533b-11eb-b6b4-801934d4c19f/result.jpg")

    # convert_to_tflite(generator, "gen48")
    # infer_tflite("gen48.tflite", create_latent_vectors(1) + [create_noise(1).reshape([1, IMAGE_SIZE*IMAGE_SIZE])])
