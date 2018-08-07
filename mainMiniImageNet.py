from datasets import miniImagenetOneShot
from option import Options
from experiments.OneShotMiniImageNetBuilder import miniImageNetBuilder

'''
:param batch_size: Experiment batch_size
:param classes_per_set: Integer indicating the number of classes per set
:param samples_per_class: Integer indicating samples per class
        e.g. For a 20-way, 1-shot learning task, use classes_per_set=20 and samples_per_class=1
             For a 5-way, 10-shot learning task, use classes_per_set=5 and samples_per_class=10
'''

# Experiment Setup
batch_size = 10
fce = True
classes_per_set = 5
samples_per_class = 5
channels = 3
# Training setup
total_epochs = 500
total_train_batches = 100
total_val_batches = 100
total_test_batches = 250
# Parse other options
args = Options().parse()

args.dataroot = './utils/data/miniImagenet'
dataTrain = miniImagenetOneShot.miniImagenetOneShotDataset(
    dataroot=args.dataroot,
    type='train',
    nEpisodes=total_train_batches * batch_size,
    classes_per_set=classes_per_set,
    samples_per_class=samples_per_class)

dataVal = miniImagenetOneShot.miniImagenetOneShotDataset(
    dataroot=args.dataroot,
    type='val',
    nEpisodes=total_val_batches * batch_size,
    classes_per_set=classes_per_set,
    samples_per_class=samples_per_class)

dataTest = miniImagenetOneShot.miniImagenetOneShotDataset(
    dataroot=args.dataroot,
    type='test',
    nEpisodes=total_test_batches * batch_size,
    classes_per_set=classes_per_set,
    samples_per_class=samples_per_class)

obj_oneShotBuilder = miniImageNetBuilder(dataTrain, dataVal, dataTest)
obj_oneShotBuilder.build_experiment(batch_size, classes_per_set,
                                    samples_per_class, channels, fce)

best_val = 0.
for e in range(0, total_epochs):
    total_c_loss, total_accuracy = obj_oneShotBuilder.run_training_epoch()
    print("Epoch {}: train_loss: {}, train_accuracy: {}".format(
        e, total_c_loss, total_accuracy))

    total_val_c_loss, total_val_accuracy = obj_oneShotBuilder.run_validation_epoch(
    )
    print("Epoch {}: val_loss: {}, val_accuracy: {}".format(
        e, total_val_c_loss, total_val_accuracy))

    if total_val_accuracy >= best_val:  # if new best val accuracy -> produce test statistics
        best_val = total_val_accuracy
        total_test_c_loss, total_test_accuracy = obj_oneShotBuilder.run_testing_epoch(
        )
        print("Epoch {}: test_loss: {}, test_accuracy: {}".format(
            e, total_test_c_loss, total_test_accuracy))

    else:
        total_test_c_loss = -1
        total_test_accuracy = -1
