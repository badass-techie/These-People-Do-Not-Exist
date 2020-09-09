'''© 2020 Moses Odhiambo'''
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from datasetloader import DatasetLoader as Loader
import bicubicinterpolation
import os
import time
import uuid

'''processing the dataset'''

try:
    dataset_path = tf.keras.utils.get_file('train_face.h5', 'https://www.dropbox.com/s/l5iqduhe0gwxumq/train_face.h5?dl=1')
    loader = Loader(dataset_path)
    num_training_examples = loader.get_train_size()
    print("The dataset has {} training examples".format(num_training_examples))
except:
    print("Dataset could not be fetched!")
#images are of 64x64 pixels
#the pixel values have already been normalized/constrained within 0 and 1
#shape of images is num_training_examples, 64, 64, 3
#the 3 is for the red, green, and blue channel
#shape of labels is num_training_examples, 1

'''setting up the discriminator'''

def create_discriminator(base_num_filters, num_outputs=1, in_shape=(64,64,3)):
    return tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters=base_num_filters, kernel_size=5, strides=2, padding='same', input_shape=in_shape),
        tf.keras.layers.LeakyReLU(alpha=0.2),   #leaky relu activation
        tf.keras.layers.Dropout(rate=0.4),      #sets random neurons to zero to prevent overfitting
        tf.keras.layers.BatchNormalization(),   #normalize activations from previous layer (maintain mean at 0 and sd at 1)
        tf.keras.layers.Conv2D(filters=base_num_filters*2, kernel_size=5, strides=2, padding='same'),
        tf.keras.layers.LeakyReLU(alpha=0.2),
        tf.keras.layers.Dropout(rate=0.4),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(filters=base_num_filters*4, kernel_size=3, strides=2, padding='same'),
        tf.keras.layers.LeakyReLU(alpha=0.2),
        tf.keras.layers.Dropout(rate=0.4),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(filters=base_num_filters*6, kernel_size=3, strides=2, padding='same'),
        tf.keras.layers.LeakyReLU(alpha=0.2),
        tf.keras.layers.Dropout(rate=0.4),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Flatten(),              #reshape to vector
        tf.keras.layers.Dense(units=512),
        tf.keras.layers.LeakyReLU(alpha=0.2),
        tf.keras.layers.Dense(units=num_outputs, activation=None)
    ])

discriminator = create_discriminator(12)    #binary classification CNN with 12 filters
#print(discriminator.summary())

'''training the discriminator'''

@tf.function    #decorator to turn this function into a tensorflow computation graph
def train_step(optimizer, x, y):
    with tf.GradientTape() as tape:
        logits = discriminator(x) #feed-forward
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits) #compute loss

    grads = tape.gradient(loss, discriminator.trainable_variables)    #compute gradients
    optimizer.apply_gradients(zip(grads, discriminator.trainable_variables))  #apply gradients
    return loss

def train_discriminator(batch_size=32, num_epochs=2, learning_rate=5e-4):
    optimizer = tf.keras.optimizers.Adam(learning_rate)
    loss_history = [] #to keep track of loss
    prev_time = time.time()
    time_elapsed = 0
    print("Training discriminator...")

    for epoch in range(num_epochs):
        for idx in range(num_training_examples//batch_size):
            x, y = loader.get_batch(batch_size)
            loss = train_step(optimizer, x, y)
            loss_history.append(loss.numpy().mean())

            time_elapsed += time.time() - prev_time
            prev_time = time.time()
            print("Batch {} of {}. Loss: {}. Time elapsed: {} seconds.".format(idx+1, num_training_examples//batch_size, loss.numpy().mean(), time_elapsed))
            
        print("Epoch {} of {} complete".format(epoch+1, num_epochs))
        discriminator.save_weights(os.path.join("./training_checkpoints", "dis_ckpt"))

    #plot a graph that will show how our loss varied with time
    plt.plot(range(len(loss_history)), loss_history)
    plt.title("Discriminator Training Progress")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.savefig(os.path.join("./plots", "Discriminator Training Progress"))
    plt.close()

def test_discriminator(batch_size=100):
    x, y = loader.get_batch(batch_size)
    y_hat = tf.round(tf.nn.sigmoid(discriminator(x)))
    print("{}% of images classified correctly".format(tf.reduce_mean(tf.cast(tf.equal(y, y_hat), tf.float32))*100))

#train_discriminator()
#test_discriminator(1000)

'''setting up the generator'''

#the generator samples points from the latent space as input and returns a square image with 3 channels(red, green, and blue) as output
#the latent space is an arbitrarily defined vector space of values from a gaussian distribution, i.e. values with a mean of 0 and standard deviation of 1
#the latent space has no meaning, but by drawing points from this space randomly and providing them to the generator, 
 #the generator assigns meaning to the latent points and, in turn, the latent space, until, at the end of training, 
 #the latent vector space represents a compressed representation of the output space, that only the generator knows how to turn into faces

def create_generator(base_num_filters, latent_dim=200):
    return tf.keras.Sequential([
        # Transform to pre-convolutional generation
        tf.keras.layers.Dense(units=4*4*6*base_num_filters, input_dim=latent_dim),  #4x4 feature maps (with 6N occurances)
        tf.keras.layers.LeakyReLU(alpha=0.2),   #leaky relu activation
        tf.keras.layers.BatchNormalization(),   #normalize activations from previous layer (maintain mean at 0 and sd at 1)
        tf.keras.layers.Reshape(target_shape=(4, 4, 6*base_num_filters)),  #inverse of tf.keras.layers.Flatten

        # Upscaling convolutions (inverse of encoder)
        tf.keras.layers.Conv2DTranspose(filters=4*base_num_filters, kernel_size=3,  strides=2, padding='same'), #8x8
        tf.keras.layers.LeakyReLU(alpha=0.2),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2DTranspose(filters=2*base_num_filters, kernel_size=3,  strides=2, padding='same'), #16x16
        tf.keras.layers.LeakyReLU(alpha=0.2),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2DTranspose(filters=1*base_num_filters, kernel_size=5,  strides=2, padding='same'), #32x32
        tf.keras.layers.LeakyReLU(alpha=0.2),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2DTranspose(filters=3, kernel_size=5,  strides=2, padding='same', activation=tf.nn.sigmoid),#final resolution of 64x64
                                                                                                                        #3 filters for the red, green, and blue channels
    ])

generator = create_generator(12)
#print(generator.summary())

#generate points in latent space as input for the generator
def generate_latent_points(latent_dim, batch_size):
    x_input = np.random.randn(latent_dim * batch_size)  #draws the given number of points from a gaussian distribution
    x_input = x_input.reshape(batch_size, latent_dim)   #reshapes points into a batch of inputs for the network
    return x_input

#use the generator to generate n fake examples, with labels of 0, showing that they're not faces
def generate_fake_samples(g_model, latent_dim, n_samples):
    x_input = generate_latent_points(latent_dim, n_samples)
    x_hat = g_model(x_input)
    y = np.zeros((n_samples, 1), dtype=np.float32)  #create class labels of 0
    return x_hat, y

'''training the composite Generator Adversarial Network (GAN) made up of the generator and discriminator'''

#the discriminator is trained prior to creating the GAN
#the generator from the GAN will try to generate an image from the latent space, then the discriminator will classify it
#at first, the pre-trained discriminator will confidently flag the output as false (output of near 0) as the generator won't have been trained yet
#we want the generator to create samples as realistic as possible, so when we calculate the loss, we mark its output as real
#this will lead to a larger loss if the generator's output is fake, and hence the weights will be updated to a larger extent
#as the generator starts to learn, we want to train the discriminator even further by providing it the fake samples from the generator, 
 #along with real ones, then updating its weights based on the error of its output
#this creates the 'adversarial' or competitive aspect of GANs - as the generator learns how to fool the discriminator,
 #the discriminator learns how to detect fake samples even better than before

@tf.function
def discriminator_train_step(optimizer, x, y):
    with tf.GradientTape() as tape:
        logits = discriminator(x) #feed-forward
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits) #compute loss

    grads = tape.gradient(loss, discriminator.trainable_variables)    #compute gradients
    optimizer.apply_gradients(zip(grads, discriminator.trainable_variables))  #apply gradients

@tf.function
def generator_train_step(optimizer, x_gen):
    with tf.GradientTape() as tape:
        y_gen = generator(x_gen)                                                            #generator output
        dis_logits = discriminator(y_gen)                                                   #discriminator's classification of generator's output
        dis_labels = np.ones(dis_logits.shape, dtype=np.float32)                                              #labels
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=dis_labels, logits=dis_logits)#compute loss

    grads = tape.gradient(loss, generator.trainable_variables)                              #compute gradients
    optimizer.apply_gradients(zip(grads, generator.trainable_variables))                    #apply gradients
    return loss

def train_gan(latent_dim=200, batch_size=32, num_epochs=24, dis_learning_rate=5e-4, gen_learning_rate=5e-4):
    num_batches = num_training_examples//batch_size
    dis_optimizer = tf.keras.optimizers.Adam(dis_learning_rate)
    gen_optimizer = tf.keras.optimizers.Adam(gen_learning_rate)
    loss_history = []
    prev_time = time.time()
    time_elapsed = 0
    print("Training GAN...")

    for epoch in range(num_epochs):
        for idx in range(num_batches):
            half_batch_size = batch_size//2

            #training the discriminator
            x_real, y_real = loader.get_batch(half_batch_size, only_faces=True)                     #fetch real samples
            x_fake, y_fake = generate_fake_samples(generator, latent_dim, half_batch_size)          #create fake samples from generator
            x_dis, y_dis = np.vstack((x_real, x_fake)), np.vstack((y_real, y_fake))                     #merge real and fake samples
            discriminator_train_step(dis_optimizer, x_dis, y_dis)                                   #train discriminator on the samples

            #training the generator
            x_gen = generate_latent_points(latent_dim, batch_size)                                  #generator input
            loss = generator_train_step(gen_optimizer, x_gen)

            loss_history.append(loss.numpy().mean())
            time_elapsed += time.time() - prev_time
            prev_time = time.time()
            print("Batch {} of {}. Loss: {}. Time elapsed: {} seconds.".format(idx+1, num_training_examples//batch_size, loss.numpy().mean(), time_elapsed))

        print("Epoch {} of {} complete".format(epoch+1, num_epochs))
        discriminator.save_weights(os.path.join("./training_checkpoints", "dis_ckpt"))
        generator.save_weights(os.path.join("./training_checkpoints", "gen_ckpt"))

    #plot a graph that will show how our loss varied with time
    plt.plot(range(len(loss_history)), loss_history)
    plt.title("GAN Training Progress")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.savefig(os.path.join("./plots", "GAN Training Progress"))
    plt.close()

#train_gan()

'''restoring the last checkpoint'''

#restore the model weights for the last checkpoint after training
generator.load_weights(tf.train.latest_checkpoint("./training_checkpoints"))
#generator.summary()

'''time to create faces😁'''

#the network trained on images of 64x64 dimensions, which are quite low-resolution
#let's fix this by upscaling them
#we'll use a popular algorithm in image processing called bicubic interpolation

#saves images generated by the GAN
def save_plot(image, filename, confirm_if_saved=True, show_instead=True):
    plt.figure(figsize=(5.12,5.12))
    plt.axis('off')
    plt.subplots_adjust(left=0,bottom=0,right=1,top=1)
    plt.imshow(image, aspect='auto', interpolation='bicubic')
    if show_instead:
        plt.show()
    else:
        plt.savefig(filename)
        plt.close()
    if confirm_if_saved and not show_instead:
        print("plot '{}' saved on disk".format(filename))

def create_faces(num_faces=10):
    for i in range(num_faces):
        latent_points = generate_latent_points(200, 1)
        img = generator(latent_points)[0]
        name = os.path.join("./generated_images", str(uuid.uuid1()) + ".png")
        save_plot(img, name)

create_faces(int(input("Enter the number of faces you want the AI to create>")))
