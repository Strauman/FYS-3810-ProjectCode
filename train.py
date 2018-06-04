from model import build_model, random_dim
import numpy as np
from prettytable import PrettyTable
from tensorflow.examples.tutorials.mnist import input_data
import generators
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
PLOTOUT_PATH="plots/{current_epoch}.png"
batch_size=128
num_epochs=5000
table_interval=20
plot_interval=100
generator_architecture=generators.conv_batchnorm()

def onehot(num_classes,hots):
    """ Generate onehot vectors from num_class classes """
    oh=np.zeros((hots.shape[0], num_classes))
    rows=np.arange(oh.shape[0])
    oh[rows,hots]=1
    return oh

generator, adversarial, discriminator=build_model(generator_architecture)

def train():
    global current_epoch
    #------ Load the data ------#
    mnist = input_data.read_data_sets('../mnist', one_hot=True,reshape=False)
    def get_noise(n):
        """ Generates n batches of noise with random mnist_labels """
        Zr=np.random.normal(0,1,size=(n,random_dim))
        Cz=onehot(10,np.random.randint(0,9,size=(n,)))
        return [Zr,Cz]
    def get_fake_images(n):
        """ Generates n images with the generator using random
        mnist_labels """
        Zr,Cz=get_noise(n)
        return generator.predict([Zr, Cz]), Cz

    def get_real_images(n):
        """ Load images with corresponding labels from the MNIST dataset """
        X_next,C_next=mnist.train.next_batch(n)
        return X_next,C_next

    def get_discriminator_data(n):
        """ Generate n batches for sending in to the discriminator """
        X_real,C_real=get_real_images(n)
        Y_real=np.ones(n)
        X_fake,C_fake=get_fake_images(n)
        Y_fake=np.zeros(n)
        X=np.concatenate([X_real, X_fake])
        Y=np.concatenate([Y_real, Y_fake])
        C=np.concatenate([C_real,C_fake])
        return X,C,Y

    for current_epoch in range(num_epochs):
        #------ Train discriminator ------#
        X,C,Y=get_discriminator_data(batch_size)
        stat_dis=discriminator.train_on_batch([X,C],Y)
        #------ Train adversarial model ------#
        Z_gen,C_gen=get_noise(batch_size)
        Y_gen=np.ones(Z_gen.shape[0])
        stat_adv=adversarial.train_on_batch([Z_gen, C_gen],Y_gen)
        if current_epoch%table_interval==0:
            tbl=PrettyTable()
            tbl.field_names=["Model",*discriminator.metrics_names]
            tbl.add_row(["Discriminator", *stat_dis])
            tbl.add_row(["Adversarial", *stat_dis])
            print(tbl)
        if current_epoch%plot_interval==0:
            plotout(current_epoch)

#------ Plotting ------#
def plotout(current_epoch,ncol=5,suffix="", relpath="images"):
    example_dim=10
    Z=np.random.normal(0, 1, size=[example_dim, random_dim])
    C=np.arange(example_dim)
    C=np.random.randint(0,9,size=Z.shape[0])
    C=onehot(10,C)
    dim=(example_dim//ncol, ncol)

    samples=generator.predict([Z,C])
    for i in range(samples.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(samples[i].reshape(28,28), interpolation='nearest', cmap='gray')
        plt.axis('off')
    plt.savefig(PLOTOUT_PATH.format(current_epoch=current_epoch,suffix=suffix,relpath=relpath))
    plt.pause(0.1)
if __name__ == '__main__':
    train()
