from __future__ import print_function, division

from keras.datasets import cifar10
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, GaussianNoise
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.layers import Concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import adam_v2
from keras import losses
from tensorflow.keras.utils import to_categorical
import keras.backend as K
import scipy
import matplotlib.pyplot as plt
from skimage.transform import resize
import numpy as np
import getopt
import os
import sys
import time

class CCGAN():
    def __init__(self):
        self.img_rows = 32
        self.img_cols = 32
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.mask_height = 10
        self.mask_width = 10
        self.num_classes = 10

        # Number of filters in first layer of generator and discriminator
        self.gf = 32
        self.df = 32

        optimizer = adam_v2.Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=['mse', 'categorical_crossentropy'],
            loss_weights=[0.5, 0.5],
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generates imgs
        masked_img = Input(shape=self.img_shape)
        gen_img = self.generator(masked_img)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The valid takes generated images as input and determines validity
        valid, _ = self.discriminator(gen_img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(masked_img , valid)
        self.combined.compile(loss=['mse'],
            optimizer=optimizer)

    # Calculate neuron activation for an input
    def activate(weights, inputs):
	    activation = weights[-1]
	    for i in range(len(weights)-1):
		    activation += weights[i] * inputs[i]
	    return activation
 
    # Transfer neuron activation
    def transfer(activation):
	    return 1.0 / (1.0 + exp(-activation))
 
    # Calculate the derivative of an neuron output
    def transfer_derivative(output):
	    return output * (1.0 - output)
 
    # Backpropagate error and store in neurons
    def backward_propagate_error(network, expected):
	    for i in reversed(range(len(network))):
		    layer = network[i]
		    errors = list()
		    if i != len(network)-1:
			    for j in range(len(layer)):
				    error = 0.0
				    for neuron in network[i + 1]:
					    error += (neuron['weights'][j] * neuron['delta'])
				    errors.append(error)
		    else:
			    for j in range(len(layer)):
				    neuron = layer[j]
				    errors.append(neuron['output'] - expected[j])
		    for j in range(len(layer)):
			    neuron = layer[j]
			    neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])
 
    # Update network weights with error
    def update_weights(network, row, l_rate):
	    for i in range(len(network)):
		    inputs = row[:-1]
		    if i != 0:
			    inputs = [neuron['output'] for neuron in network[i - 1]]
		    for neuron in network[i]:
			    for j in range(len(inputs)):
				    neuron['weights'][j] -= l_rate * neuron['delta'] * inputs[j]
			    neuron['weights'][-1] -= l_rate * neuron['delta']
       
    def build_generator(self):
        """U-Net Generator"""

        def conv2d(layer_input, filters, f_size=4, bn=True):
            """Layers used during downsampling"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
            """Layers used during upsampling"""
            u = UpSampling2D(size=2)(layer_input)
            u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
            if dropout_rate:
                u = Dropout(dropout_rate)(u)
            u = BatchNormalization(momentum=0.8)(u)
            u = Concatenate()([u, skip_input])
            return u

        img = Input(shape=self.img_shape)

        # Downsampling
        d1 = conv2d(img, self.gf, bn=False)
        d2 = conv2d(d1, self.gf*2)
        d3 = conv2d(d2, self.gf*4)
        d4 = conv2d(d3, self.gf*8)

        # Upsampling
        u1 = deconv2d(d4, d3, self.gf*4)
        u2 = deconv2d(u1, d2, self.gf*2)
        u3 = deconv2d(u2, d1, self.gf)

        u4 = UpSampling2D(size=2)(u3)
        output_img = Conv2D(self.channels, kernel_size=4, strides=1, padding='same', activation='tanh')(u4)

        return Model(img, output_img)

    
    
    def build_discriminator(self):

        img = Input(shape=self.img_shape)

        model = Sequential()
        model.add(Conv2D(64, kernel_size=4, strides=2, padding='same', input_shape=self.img_shape))
        model.add(LeakyReLU(alpha=0.8))
        model.add(Conv2D(128, kernel_size=4, strides=2, padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(InstanceNormalization())
        model.add(Conv2D(256, kernel_size=4, strides=2, padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(InstanceNormalization())

        model.summary()

        img = Input(shape=self.img_shape)
        features = model(img)

        validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(features)

        label = Flatten()(features)
        label = Dense(self.num_classes+1, activation="softmax")(label)

        return Model(img, [validity, label])

    def mask_randomly(self, imgs):
        y1 = np.random.randint(0, self.img_rows - self.mask_height, imgs.shape[0])
        y2 = y1 + self.mask_height
        x1 = np.random.randint(0, self.img_rows - self.mask_width, imgs.shape[0])
        x2 = x1 + self.mask_width

        masked_imgs = np.empty_like(imgs)
        for i, img in enumerate(imgs):
            masked_img = img.copy()
            _y1, _y2, _x1, _x2 = y1[i], y2[i], x1[i], x2[i],
            masked_img[_y1:_y2, _x1:_x2, :] = 0
            masked_imgs[i] = masked_img

        return masked_imgs


    def train(self, epochs, batch_size=128, sample_interval=50):

        # Load the dataset
        (X_train, y_train), (_, _) = cifar10.load_data()

        
        # Adversarial ground truths
        valid = np.ones((batch_size, 4, 4, 1))
        fake = np.zeros((batch_size, 4, 4, 1))

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Sample half batch of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]
            labels = y_train[idx]

            masked_imgs = self.mask_randomly(imgs)

            # Generate a half batch of new images
            gen_imgs = self.generator.predict(masked_imgs)

            # One-hot encoding of labels
            labels = to_categorical(labels, num_classes=self.num_classes+1)
            fake_labels = to_categorical(np.full((batch_size, 1), self.num_classes), num_classes=self.num_classes+1)

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(imgs, [valid, labels])
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, [fake, fake_labels])
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Train the generator
            g_loss = self.combined.train_on_batch(masked_imgs, valid)

            # Plot the progress
            print ("%d [D loss: %f, op_acc: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[4], g_loss))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                # Select a random half batch of images
                idx = np.random.randint(0, X_train.shape[0], 6)
                imgs = X_train[idx]
                self.sample_images(epoch, imgs)
                self.save_model()

    def sample_images(self, epoch, imgs):
        r, c = 3, 6

        masked_imgs = self.mask_randomly(imgs)
        gen_imgs = self.generator.predict(masked_imgs)

        imgs = (imgs + 1.0) * 0.5
        masked_imgs = (masked_imgs + 1.0) * 0.5
        gen_imgs = (gen_imgs + 1.0) * 0.5

        gen_imgs = np.where(gen_imgs < 0, 0, gen_imgs)

        fig, axs = plt.subplots(r, c)
        for i in range(c):
            axs[0,i].imshow(imgs[i, :, :, 0])
            axs[0,i].axis('off')
            axs[1,i].imshow(masked_imgs[i, :, :, 0])
            axs[1,i].axis('off')
            axs[2,i].imshow(gen_imgs[i, :, :, 0])
            axs[2,i].axis('off')
        fig.savefig("ccgan/images/%d.png" % epoch)
        plt.close()

    def save_model(self):

        def save(model, model_name):
            model_path = "ccgan/saved_model/%s.json" % model_name
            weights_path = "ccgan/saved_model/%s_weights.hdf5" % model_name
            options = {"file_arch": model_path,
                        "file_weight": weights_path}
            json_string = model.to_json()
            open(options['file_arch'], 'w').write(json_string)
            model.save_weights(options['file_weight'])

        save(self.generator, "ccgan_generator")
        save(self.discriminator, "ccgan_discriminator")

    #AMF
    def process(image, size, window=1, threshold=0., spam=False):
        ## set filter window and image dimensions
        filter_window = 2*window + 1
        xlength, ylength = size
        vlength = filter_window*filter_window
        if spam:
            print('Image length in X direction: {}'.format(xlength))
            print('Image length in Y direction: {}'.format(ylength))
            print('Filter window size: {0} x {0}'.format(filter_window))

        ## create 2-D image array and initialize window
        image_array = np.reshape(np.array(image, dtype=np.uint8), (ylength, xlength))
        filter_window = np.array(np.zeros((filter_window, filter_window)))
        target_vector = np.array(np.zeros(vlength))
        pixel_count = 0

        try:
            ## loop over image with specified window filter_window
            for y in range(window, ylength-(window+1)):
                for x in range(window, xlength-(window+1)):
                ## populate window, sort, find median
                    filter_window = image_array[y-window:y+window+1, x-window:x+window+1]
                    target_vector = np.reshape(filter_window, ((vlength),))
                    ## numpy sort
                    median = median_demo(target_vector, vlength)
                    ## C median library
                    # median = medians_1D.quick_select(target_vector, vlength)
                ## check for threshold
                    if not threshold > 0:
                        image_array[y, x] = median
                        pixel_count += 1
                    else:
                        scale = np.zeros(vlength)
                        for n in range(vlength):
                            scale[n] = abs(int(target_vector[n]) - int(median))
                        scale = np.sort(scale)
                        Sk = 1.4826 * (scale[vlength//2])
                        if abs(int(image_array[y, x]) - int(median)) > (threshold * Sk):
                            image_array[y, x] = median
                            pixel_count += 1

        except TypeError as err:
            print('Error in processing function:'.format(err))
            sys.exit(2)
            ## ,NameError,ArithmeticError,LookupError

        print('{} pixel(s) filtered out of {}'.format(pixel_count, xlength*ylength))
        ## convert array back to sequence and return
        return np.reshape(image_array, (xlength*ylength)).tolist()


    def median_demo(target_array, array_length):
        sorted_array = np.sort(target_array)
        median = sorted_array[array_length//2]
        return median

    class Timer(object):
        def __init__(self, verbose=False):
            self.verbose = verbose

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.secs = self.end - self.start
        self.msecs = self.secs * 1000  # millisecs
        if self.verbose:
            print('elapsed time: {} ms'.format(self.msecs))


    def main(argv):
        filenames = None
        try:
            args, filenames = getopt.getopt(argv[1:],'hvwt',['help', 'verbose', 'window=', 'threshold='])
        except getopt.error as msg:
            args = "dummy"
            print(msg)
            print('Usage: {} [-h|v|--window=[1..5]|--threshold=[0..N]] <filename>'.format(argv[0]))
            print('Demonstrates adaptive median filtering on gray-scale images.')
            sys.exit(2)

        # Obligatory spam variable; controls verbosity of the output
        spam = False

        # window = ws, where the filter_window = 2*ws + 1,
        # ie, ws = 1 is a 3x3 window (filter_window=3)
        window = 1
        threshold = 0.

        for o, a in args:
            if o in ("-h", "--help"):
                print(__doc__)
                sys.exit(0)
            if o in ("-v", "--verbose"):
                spam = True

        if spam:
            print('options = {}'.format(args))
            print('filenames = {}'.format(filenames))

        try:
            for o in args[:]:
                if o[0] == '--threshold' and o[1] != '':
                    threshold = float(o[1])
                    args.remove(o)
                if o[0] == '--threshold' and o[1] == '':
                    print('The --threshold option requires an argument.')
                    sys.exit(2)
            for o in args[:]:
                if o[0] == '--window' and o[1] != '':
                    window = int(o[1])
                    args.remove(o)
                if o[0] == '--window' and o[1] == '':
                    print('The --window option requires an argument.')
                    sys.exit(2)
        except ValueError as err:
            print('Parameter error: {}\nOption must be a number.'.format(err))
            sys.exit(2)
        except TypeError as err:
            print('Parameter error: {}'.format(err))
            sys.exit(2)

        if threshold < 0.:
            print('The threshold must be a non-negative real value (default=0).')
            sys.exit(2)

        if not 1 <= window <= 5:
            print('The window size must be an integer between 1 and 5 (default=1).')
            sys.exit(2)

        if not filenames:
            print('Please enter one or more image filenames.')
            sys.exit(2)

        if spam:
            print('window = {}'.format(window))
            print('threshold = {}'.format(threshold))

        image_count = 0
        filter_time = 0.

        for filename in filenames:
            try:
                infile = open(filename, "rb")
            except IOError as err:
                print('Input file error: {}'.format(err))
                if spam:
                    print('Please check the name(s) of your input file(s).')
                os.close(sys.stderr.fileno())
                sys.exit(2)

            try:
                pil_image = Image.open(infile)
                if pil_image.mode == 'P':
                    if spam:
                        print('Original image mode: {}'.format(pil_image.mode))
                    pil_image = pil_image.convert('L')
            except IOError:
                print('Cannot parse input image format: {}'.format(pil_image))
            if spam:
                print('Input image format: {}'.format(pil_image.format))
                print('Input image size: {}'.format(pil_image.size))
                print('Working image mode: {}'.format(pil_image.mode))

            ## Convert the PIL image object to a python sequence (list)
            input_sequence = list(pil_image.getdata())

            try:
                ## filter input image sequence
                with Timer(spam) as ttimer:
                    output_sequence = process(input_sequence, pil_image.size, window, threshold, spam)

                ## init output image
                file, ext = os.path.splitext(filename)
                outfile = "new_" + file + ext
                try:
                    output_image = Image.new(pil_image.mode, pil_image.size, None)
                    output_image.putdata(output_sequence)
                    output_image.save(outfile, pil_image.format)
                    if spam:
                        print('Output image name: {}'.format(outfile))

                except IOError as err:
                    print('Output file error: {}'.format(err))
                    if spam:
                        print('Cannot create output image for {}.'.format(input_image))
                        print('  Continuing with next available file...')
                    pass

            except MemoryError as err:
                sys.stderr.write(err)
                if spam:
                    print('Not enough memory to create output image for {}.'.format(input_image))
                    print('  Continuing with next available file...')
                pass

            infile.close()
            image_count += 1
if __name__ == '__main__':
    ccgan = CCGAN()
    ccgan.train(epochs=20000, batch_size=32, sample_interval=200)
