

# Compare simple use of ML on our data
import numpy as np

from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from scipy.stats import mode
import matplotlib.pyplot as plt
from scipy.io import loadmat
from mpl_toolkits.mplot3d import Axes3D
from sklearn import tree
import graphviz

# load data from saved matlab file
D = loadmat("./tf_e_features.mat")
print(D.keys())

dataC = D['E_chest']
dataR = D['E_foot']
dataTR = D['E_thigh']
L = D['L']
des = D['des']
t = D['t']
num = des.shape[0]

# train and validate brute force KNN and SVM methods
def predict_knn(feature_train, label_train, k):
    # find nearest neighbors
    neigh = KNeighborsClassifier(n_neighbors=k).fit(feature_train, label_train)

    return neigh


def classify_knn(feature_train, label_train, feature_test, label_test, n):
    # predict label KNN
    neigh = predict_knn(feature_train, label_train, 2)

    label_test_pred = neigh.predict(feature_test)
    label_test_pred = label_test_pred.astype('int64')

    #label_classes = ['Bend','Sit','Lie down','Stand','Turn','Walk','Fallfwd']
    #confusion, accuracy = compute_confusion_matrix(label_test_pred, label_test, n)
    #visualize_confusion_matrix(confusion, accuracy, label_classes[:n], 'KNN Confusion Matrix')

    return label_test_pred

def classify_svm(x_train, y_train, x_test, y_test, n):
    # train label SVM
    clf = SVC(kernel='rbf', class_weight='balanced', C=1e3, gamma=0.1)
    clf = clf.fit(x_train, y_train)

    # predict using svm
    y_pred = clf.predict(x_test)

    #confusion, accuracy = compute_confusion_matrix(y_pred, y_test, n)
    #label_classes = ['Bend','Sit','Lie down','Stand','Turn','Walk','Fallfwd']
    #visualize_confusion_matrix(confusion, accuracy, label_classes[:n], 'SVM Confusion Matrix')
    return y_pred

def classify_nn(x_train, y_train, x_test, y_test, n):

    # correct int classes start from 0 to 6
    y_train = y_train - 1
    y_test = y_test - 1

    # define the keras model
    model = Sequential()
    model.add(Dense(20, input_dim=x_train.shape[1], activation='relu'))
    model.add(Dense(8))
    model.add(Dense(7, activation='sigmoid'))

    # compile the keras model
    model.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=['accuracy'])

    # fit the keras model on the dataset
    y_binary = to_categorical(y_train) # minus 1 to bring int start from 0 instead of 1
    model.fit(x_train, y_binary, epochs=100, batch_size=10)

    # predict using svm
    y_pred = model.predict_classes(x_test)

    #confusion, accuracy = compute_confusion_matrix(y_pred, y_test, n)
    #label_classes = ['Bend','Sit','Lie down','Stand','Turn','Walk','Fallfwd']
    #visualize_confusion_matrix(confusion, accuracy, label_classes[:n], 'NN Confusion Matrix')

    # return start from 1 to 7 classes
    #y_pred = y_pred + 1
    return y_pred

def compute_confusion_matrix(label_pred, label_test, n):
    label_test_pred = label_pred.astype('int64')
    label_test = label_test.astype('int64')
    num_test = label_test.shape[0]
    acc = 0
    confusion = np.zeros((n, n))
    for i in range(num_test):
        confusion[label_test_pred[i] - 1, label_test[i] - 1] += 1
        if label_test_pred[i] == label_test[i]:
            acc = acc + 1
    accuracy = acc / num_test
    for i in range(n):
        if np.sum(confusion[:, i]) != 0:
            confusion[:, i] = confusion[:, i] / np.sum(confusion[:, i])
    return confusion, accuracy


def visualize_confusion_matrix(confusion, accuracy, label_classes, name):
    plt.title("{}, accuracy = {:.3f}".format(name, accuracy))
    plt.imshow(confusion)
    ax, fig = plt.gca(), plt.gcf()
    plt.xticks(np.arange(len(label_classes)), label_classes)
    plt.yticks(np.arange(len(label_classes)), label_classes)
    ax.set_xticks(np.arange(len(label_classes) + 1) - .5, minor=True)
    ax.set_yticks(np.arange(len(label_classes) + 1) - .5, minor=True)
    ax.tick_params(which="minor", bottom=False, left=False)
    plt.show()


def visualize_activity_recognition(t, label_true, label_pred_mode, label_classes,name):
    plt.title("Activity recognition {}".format(name))
    plt.plot(t, label_true, linewidth=3.0)
    plt.plot(t.reshape(-1), label_pred_mode.reshape(-1), '--')
    plt.yticks(np.arange(len(label_classes)) + 1, label_classes)
    plt.xlabel("time (s)")
    plt.ylabel("Activity")
    plt.legend(["True", "Predict"])
    plt.show()


# cut data to smaller database
dataC_cut = np.reshape(dataC[1,:],(1,-1))
des_cut = np.reshape(des[1,:],(1,-1))
for i in range(7):
    dataC_cut = np.append(dataC_cut,dataC[i*1999+500:i*1999+700,:],axis=0)
    des_cut = np.append(des_cut,des[i*1999+500:i*1999+700,:],axis=0)

num_cut = des_cut.shape[0]
t_cut = np.arange(0,num_cut*0.01,0.01)

label_knn_pred = np.zeros((6, num_cut))
label_knn = des_cut
label_svm_pred = np.zeros((6, num_cut))
label_svm = label_knn
label_cnn_pred = np.zeros((6, num_cut))
label_cnn = label_knn
num_classes = 7

label_classes = ['Bend','Sit','Lie down','Stand','Turn','Walk','Fallfwd']

# PCA analysis
X = dataC_cut
Y = des_cut.reshape(-1)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=42)
X_test = X
Y_test = Y

# use PCA and choose suitable signals
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.decomposition import PCA

pca = PCA()
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

explained_variance = pca.explained_variance_ratio_

pca = PCA(n_components=4)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

# predict label KNN
label_knn_pred = classify_knn(X_train, Y_train, X_test, Y_test, num_classes)
label_svm_pred = classify_svm(X_train, Y_train, X_test, Y_test, num_classes)
label_cnn_pred = classify_nn(X_train, Y_train, X_test, Y_test, num_classes)
label_cnn_pred = label_cnn_pred + 1

'''
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

cm = confusion_matrix(Y_test, label_svm_pred)
print(cm)
print(accuracy_score(Y_test, label_svm_pred))
'''

confusion_knn, accuracy_knn = compute_confusion_matrix(label_knn_pred.reshape(-1), label_knn, num_classes)
confusion_svm, accuracy_svm = compute_confusion_matrix(label_svm_pred.reshape(-1), label_svm, num_classes)
confusion_cnn, accuracy_cnn = compute_confusion_matrix(label_cnn_pred.reshape(-1), label_cnn, num_classes)

visualize_confusion_matrix(confusion_knn, accuracy_knn, label_classes[:num_classes], 'KNN Mode Confusion Matrix')
visualize_confusion_matrix(confusion_svm, accuracy_svm, label_classes[:num_classes], 'SVM Mode Confusion Matrix')
visualize_confusion_matrix(confusion_cnn, accuracy_cnn, label_classes[:num_classes], 'NN Mode Confusion Matrix')

visualize_activity_recognition(t_cut, Y, label_knn_pred, label_classes,"KNN")
visualize_activity_recognition(t_cut, Y, label_svm_pred, label_classes,"SVM")
visualize_activity_recognition(t_cut, Y, label_cnn_pred, label_classes,"NN")

plt.title('Scatter plot of first two transformed signals')
scatter = plt.scatter(X_train[:, 0], X_train[:, 1], c=Y_train, marker='o',cmap='Accent')#jet, Accent
plt.xlabel("z1")
plt.ylabel("z2")
plt.legend(handles=scatter.legend_elements()[0], labels=label_classes)
plt.show()



# decision Tree
clf = tree.DecisionTreeClassifier(max_depth=4)
clf = clf.fit(X_train, Y_train)

y_pred = clf.predict(X_test)

confusion_DT, accuracy_DT = compute_confusion_matrix(y_pred.reshape(-1), label_knn, num_classes)
visualize_confusion_matrix(confusion_DT, accuracy_DT, label_classes[:num_classes], 'DT Confusion Matrix')

tree.plot_tree(clf)
plt.show()
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'
os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/dot.exe'
dot_data = tree.export_graphviz(clf, out_file=None,
                     filled=True, rounded=True,
                     special_characters=True)
graph = graphviz.Source(dot_data)
graph.render("activity")

'''
# Plot data after 3D PCA
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plt.title('Signals After 3D PCA')
ax.scatter(X_train[:, 0], X_train[:, 1], X_train[:, 2], c=Y_train, marker='o')
ax.set_zscale('symlog')
ax.set_yscale('symlog')
ax.set_xscale('symlog')
plt.show()
'''