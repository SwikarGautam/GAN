import tensorflow as tf
from ConvoNet import *
import matplotlib.pyplot as plt

def vec(w):
    e = np.zeros((10, 1))
    e[w] = 1.0
    return e



mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

x_train = x_train.reshape((60000, 784))
x_test = x_test.reshape((10000, 784))
x_train = np.concatenate((x_train, x_test), axis=0)
y_train.reshape(60000, 1)
y_test.reshape(10000, 1)
y_train = [vec(y) for y in y_train]
y_test = [vec(y) for y in y_test]
train = zip(x_train, y_train) 
train_set = list(train)
test = zip(x_test, y_test)
test_set = list(test)  

Discriminator = ConvoNet((28, 28, 1), BinaryCrossEntropyLoss())

Discriminator.add(Dense(784, 1024, LeakyReLu(0.2) ))
Discriminator.add(Dense(1024, 512, LeakyReLu(0.2))) 
Discriminator.add(Dense(512, 256, LeakyReLu(0.2)))
Discriminator.add(Dense(256, 1, Sigmoid()))


Generator = ConvoNet((100, 1, 1), NoLoss())

Generator.add(Dense(100, 256, LeakyReLu(0.2), batch_norm_momentum=0.8))
Generator.add(Dense(256, 512, LeakyReLu(0.2),batch_norm_momentum=0.8))
Generator.add(Dense(512, 1024, LeakyReLu(0.2),batch_norm_momentum=0.8))
Generator.add(Dense(1024, 784, Tanh()))

Generator.set_adam_parameters(0.5, 0.9)
Discriminator.set_adam_parameters(0.5, 0.9)

GAN = ConvoNet((100, 1, 1), CrossEntropyLoss(classification=False))

GAN.sequence.extend(Generator.sequence)
GAN.sequence.extend(Discriminator.sequence)
GAN.set_regularization(0.0001)


def train_GAN(real_data, noise, mini_batch_size, lr1, lr2):
    Discriminator.set_update(True)

    fake_img = Generator.predict(noise)

    real_label = np.full((mini_batch_size, 1), 0.9)
    fake_label = np.full((mini_batch_size, 1), 0)
    real = list(zip(real_data, real_label))
    fake = list(zip(fake_img, fake_label))
    real.extend(fake)
    all_data = real
    random.shuffle(all_data)
    Discriminator.train(all_data, 1, lr1, 256, False)

    Discriminator.set_update(False)

    fake_data = list(zip(noise, real_label))
    GAN.train(fake_data,2, lr2, 128, False)


tnoise = np.random.randn(1, 100)

timg2 = Generator.predict(np.random.randn(1, 100))
mini_batch_size = 256

lr1 = 0.0015
lr2 = 0.003
for j in range(1):
    
    for i in range(0, 70000, mini_batch_size):
        if lr2 > 0.001:
            lr2 *= 0.9985
        if lr1 > 0.0005:
            lr1*=0.9985
        noise = np.random.randn(256, 100)

        train_GAN(x_train[i:i+256], noise, 256, lr1, lr2)
    print("epoch:\t{}".format(j+1))

print("evaluating...")
fig, ax = plt.subplots(2, 7)

for i in range(2):
    for j in range(7):
        noise = np.random.randn(1, 100)
        img = Generator.predict(noise)
        img = np.where(img<0,0,img)
        ax[i, j].imshow(img.reshape(28, 28))
        ax[i, j].set_title('%.4f' % (GAN.predict(noise)[0][0]))
        ax[i, j].axis('off')
ax[0, 0].imshow(x_train[random.randint(0, 1000)].reshape(28, 28))
ax[0, 0].axis('off')
ax[0, 0].set_title('%.3f' % (Discriminator.predict(x_train[8])[0][0]))
plt.show()