from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np
from sklearn import preprocessing
import matplotlib as plt
import pandas as pd
import os
import cv2

def load_images(path):
    """
    Loads images from specified path.
    Args:
        path: Path to load images from.
    Returns:
        images (numpy.array): numpy array of loaded images.
    """
    images = []
    for filename in os.listdir(path):
        img = cv2.imread(os.path.join(path, filename), cv2.IMREAD_COLOR)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)
    return np.array(images)

def normalize_data(data, width, height, negative=False):
    """
    Normalizes images in interval [0 1] or [-1 1] and returns batches.
    Args:
        data (numpy.array): Array of images to process.
        width (int): Width dimension of image in pixels.
        height (int): Height dimension of image in pixels.
        negative (bool, optional): Flag that determines interval (True: [-1 1], False: [0 1]). Defaults to True.
    Returns:
        data (tf.data.Dataset): Kvasir-SEG dataset sliced w/ batch_size and normalized.
    """
    normalized_data = []
    for image in data:
        resized_image = cv2.resize(image, (width, height)).astype('float32')
        if negative:
            image = (resized_image / 127.5) - 1
        else:
            image = (resized_image / 255.0)
        normalized_data.append(image)
    return normalized_data

lda = LinearDiscriminantAnalysis()
X = np.load('features.npy', allow_pickle=True)
# X = np.array([X])
print(X)
print("shape is ", X.shape)
labels = []
values = []
for x in X :
    print(x[0])
    labels.append(x[0])
    print(x[1])
    values.append(x[1])

print("===========")
# print(X[1])

# a = []
# for key, value in X.iteritems():
#     temp = [key,value]
#     a.append(temp)
# print(a)
# print(np.dtype(X))
# print(X.fields)
#convert dataset to pandas DataFrame
# df = pd.DataFrame(data = np.c_[iris['data'], iris['target']],
#                  columns = iris['feature_names'] + ['target'])
# df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
# df.columns = ['s_length', 's_width', 'p_length', 'p_width', 'target', 'species']
#
# #view first six rows of DataFrame
# df.head()

le = preprocessing.LabelEncoder()
# labels = ["Cheetah", "Jaguar", "Leopard", "Lion", "Tiger"]
y = le.fit_transform(labels)
print("labels ", y)
X = values
print(len(values))
print(values[0].shape)
print(values[1].shape)
print(values[2].shape)
print(values[3].shape)
print(values[4].shape)

exit()
X_lda = lda.fit_transform(X, y)


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# %matplotlib inline

n_train = 20  # samples for training
n_test = 200  # samples for testing
n_averages = 50  # how often to repeat classification
n_features_max = 75  # maximum number of features
step = 4  # step size for the calculation


def generate_data(n_samples, n_features):
    """Generate random blob-ish data with noisy features.

    This returns an array of input data with shape `(n_samples, n_features)`
    and an array of `n_samples` target labels.

    Only one feature contains discriminative information, the other features
    contain only noise.
    """
    X, y = make_blobs(n_samples=n_samples, n_features=1, centers=[[-2], [2]])

    # add non-discriminative features
    if n_features > 1:
        X = np.hstack([X, np.random.randn(n_samples, n_features - 1)])
    return X, y

acc_clf1, acc_clf2 = [], []
n_features_range = range(1, n_features_max + 1, step)
for n_features in n_features_range:
    score_clf1, score_clf2 = 0, 0
    for _ in range(n_averages):
        X, y = generate_data(n_train, n_features)

        clf1 = LinearDiscriminantAnalysis(solver='lsqr', shrinkage=0.5).fit(X, y)
        clf2 = LinearDiscriminantAnalysis(solver='lsqr', shrinkage=None).fit(X, y)

        X, y = generate_data(n_test, n_features)
        score_clf1 += clf1.score(X, y)
        score_clf2 += clf2.score(X, y)

    acc_clf1.append(score_clf1 / n_averages)
    acc_clf2.append(score_clf2 / n_averages)

features_samples_ratio = np.array(n_features_range) / n_train

with plt.style.context('seaborn-talk'):
    plt.plot(features_samples_ratio, acc_clf1, linewidth=2,
             label="LDA with shrinkage", color='navy')
    plt.plot(features_samples_ratio, acc_clf2, linewidth=2,
             label="LDA", color='gold')

    plt.xlabel('n_features / n_samples')
    plt.ylabel('Classification accuracy')
    plt.legend(prop={'size': 18})
    plt.tight_layout()

lda_variances =  lda.explained_variance_ratio_

plt.xlabel('LD1')
plt.ylabel('LD2')
plt.scatter(
    X_lda[:,0],
    X_lda[:,1],
    c=y,
    cmap='rainbow',
    alpha=0.7,
    edgecolors='b'
)
