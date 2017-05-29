from train_gan import MnistGanModelTrain
from mnist_gan import MnistGanModel
from dataset import get_data

data_dir = "/Users/ahmetkucuk/Documents/Research/Medical/sample_patches/"

gan_model = MnistGanModel()
dataset = get_data(dataset_dir=data_dir, size=28)
print(dataset.size())
train = MnistGanModelTrain(gan_model, dataset)
train.train(n_of_epochs=4000, n_of_samples=160, should_plot=True, batch_size=50)

