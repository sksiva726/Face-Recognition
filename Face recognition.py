# Import all required libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
#Visualization
import matplotlib.pyplot as plt
import seaborn as sns # for advance data visualization
#Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import metrics

#System

face_data = np.load(r"C:\Users\DELL\OneDrive\Desktop\PREP0106\face recognitionPCA\archive (1)\olivetti_faces.npy")
target = np.load(r"C:\Users\DELL\OneDrive\Desktop\PREP0106\face recognitionPCA\archive (1)\olivetti_faces_target.npy")
print(face_data[0][:].shape) 
print(target)

#check the total number of data points and also unique 
print("There are {} images in the dataset".format(len(face_data)))
print("There are {} unique targets in the dataset".format(len(np.unique(target))))
print("Size of each image is {}x{}".format(face_data.shape[1],face_data.shape[2]))
print("Pixel values were scaled to [0,1] interval. e.g:{}".format(face_data[0][0,:4]))
face_data.shape
print("unique target number:",np.unique(target))

def show_40_distinct_people(images, unique_ids):
    #Creating 4X10 subplots in  18x9 figure size
    fig, axarr = plt.subplots(nrows = 4, ncols = 10, figsize = (18, 9))
    #For easy iteration flattened 4X10 subplots matrix to 40 array
    axarr=axarr.flatten()
    
    #iterating over user ids
    for unique_id in unique_ids:
        image_index = unique_id*10

        axarr[unique_id].imshow(images[image_index], cmap = 'gray')
        axarr[unique_id].set_xticks([])
        axarr[unique_id].set_yticks([])
        axarr[unique_id].set_title("face id:{}".format(unique_id))
    plt.suptitle("There are 40 distinct people in the dataset")
    

show_40_distinct_people(face_data, np.unique(target))
plt.show()

def show_10_faces_of_n_subject(images, subject_ids):
    cols=10# each subject has 10 distinct face images
    rows=(len(subject_ids)*10)/cols #
    rows=int(rows)
    
    fig, axarr=plt.subplots(nrows=rows, ncols=cols, figsize=(18,9))
    #axarr=axarr.flatten()
    
    for i, subject_id in enumerate(subject_ids):
        for j in range(cols):
            image_index=subject_id*10 + j
            axarr[i,j].imshow(images[image_index], cmap="gray")
            axarr[i,j].set_xticks([])
            axarr[i,j].set_yticks([])
            axarr[i,j].set_title("face id:{}".format(subject_id))
    
#You can playaround subject_ids to see other people faces
show_10_faces_of_n_subject(images = face_data, subject_ids=[0, 5, 21, 24, 36])
plt.show()

#We reshape images for machine learnig  model
X=face_data.reshape((face_data.shape[0],face_data.shape[1]*face_data.shape[2]))
print("X shape:",X.shape)


X_train, X_test, y_train, y_test = train_test_split(X, target, test_size=0.3, stratify = target, random_state = 0)
print("X_train shape:",X_train.shape)
print("y_train shape:{}".format(y_train.shape))

y_frame = pd.DataFrame()
y_frame['subject ids'] = y_train
y_frame.groupby(['subject ids']).size().plot.bar(figsize = (15,8),title = "Number of Samples for Each Classes", color = 'red')
plt.show()
 
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
pca.fit(X)
X_pca=pca.transform(X)
X.shape
X.size

number_of_people = 10
index_range = number_of_people*10
fig = plt.figure(figsize = (10,8))
ax = fig.add_subplot(1,1,1)
scatter = ax.scatter(X_pca[:index_range,0],
            X_pca[:index_range,1], 
            c = target[:index_range],
            s = 10,
           cmap = plt.get_cmap('jet', number_of_people)
          )

ax.set_xlabel("First Principle Component")
ax.set_ylabel("Second Principle Component")
ax.set_title("PCA projection of {} people".format(number_of_people))

fig.colorbar(scatter)
plt.show()

# Finding Optimum Number of Principle Component

pca = PCA()
pca.fit(X)

plt.figure(1, figsize = (12,8))

plt.plot(pca.explained_variance_, linewidth = 2)
 
plt.xlabel('Components')
plt.ylabel('Explained Variaces')
plt.show()

n_components = 90
pca=PCA(n_components=n_components, whiten=True)
pca.fit(X_train)

fig,ax=plt.subplots(1,1,figsize=(8,8))
ax.imshow(pca.mean_.reshape((64,64)), cmap="gray")
ax.set_xticks([])
ax.set_yticks([])
ax.set_title('Average Face')
plt.show()

number_of_eigenfaces = len(pca.components_)
eigen_faces = pca.components_.reshape((number_of_eigenfaces, face_data.shape[1], face_data.shape[2]))

cols = 10
rows = int(number_of_eigenfaces/cols)
fig, axarr = plt.subplots(nrows = rows, ncols = cols, figsize = (15,15))
axarr = axarr.flatten()
for i in range(number_of_eigenfaces):
    axarr[i].imshow(eigen_faces[i],cmap = "gray")
    axarr[i].set_xticks([])
    axarr[i].set_yticks([])
    axarr[i].set_title("eigen id:{}".format(i))
plt.suptitle("All Eigen Faces".format(10*"=", 10*"="))
plt.show()

X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
# apply support vector classifier
clf = SVC()
clf.fit(X_train_pca, y_train)
y_pred = clf.predict(X_test_pca)
print("accuracy score:{:.2f}".format(metrics.accuracy_score(y_test, y_pred)))

import seaborn as sns
plt.figure(1, figsize=(12,8))
sns.heatmap(metrics.confusion_matrix(y_test, y_pred))
plt.show()

print(metrics.classification_report(y_test, y_pred))

models=[]
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(("LR",LogisticRegression()))
models.append(("NB",GaussianNB()))
models.append(("KNN",KNeighborsClassifier(n_neighbors=5)))
models.append(("DT",DecisionTreeClassifier()))
models.append(("SVM",SVC()))

for name, model in models:
    
    clf = model

    clf.fit(X_train_pca, y_train)

    y_pred = clf.predict(X_test_pca)
    print(10*"=","{} Result".format(name).upper(),10*"=")
    print("Accuracy score:{:0.2f}".format(metrics.accuracy_score(y_test, y_pred)))
    print()
#According to the above results, Linear Discriminant Analysis and Logistic Regression seems to have the best performances.

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
pca = PCA(n_components = n_components, whiten = True)
pca.fit(X)
X_pca = pca.transform(X)
for name, model in models:
    kfold = KFold(n_splits = 5, shuffle = True, random_state = 0)
    
    cv_scores = cross_val_score(model, X_pca, target, cv = kfold)
    print("{} mean cross validations score:{:.2f}".format(name, cv_scores.mean()))
# According to the cross validation scores Linear Discriminant Analysis and Logistic Regression still have best performance

lr = LinearDiscriminantAnalysis()
lr.fit(X_train_pca, y_train)
y_pred = lr.predict(X_test_pca)
print("Accuracy score:{:.2f}".format(metrics.accuracy_score(y_test, y_pred)))
 
 # confusion_matrix
cm = metrics.confusion_matrix(y_test, y_pred)

plt.subplots(1, figsize = (12,12))
sns.heatmap(cm)
plt.show()

print("Classification Results:\n{}".format(metrics.classification_report(y_test, y_pred)))

#  Leave One Out vross-validation
from sklearn.model_selection import LeaveOneOut
loo_cv = LeaveOneOut()
clf = LogisticRegression()
cv_scores = cross_val_score(clf,
                         X_pca,
                         target,
                         cv = loo_cv)
print("{} Leave One Out cross-validation mean accuracy score:{:.2f}".format(clf.__class__.__name__, cv_scores.mean()))


from sklearn.model_selection import LeaveOneOut
loo_cv = LeaveOneOut()
clf = LinearDiscriminantAnalysis()
cv_scores = cross_val_score(clf,
                         X_pca,
                         target,
                         cv=loo_cv)
print("{} Leave One Out cross-validation mean accuracy score:{:.2f}".format(clf.__class__.__name__, cv_scores.mean()))

from sklearn.model_selection import LeaveOneOut, GridSearchCV

#This process takes long time. You can use parameter:{'C': 1.0, 'penalty': 'l2'} 
#grid search cross validation score:0.93
"""
params = {'penalty':['l1', 'l2'],
                'C':np.logspace(0, 4, 10)
                }
clf = LogisticRegression()
#kfold=KFold(n_splits=3, shuffle=True, random_state=0)
loo_cv = LeaveOneOut()
gridSearchCV = GridSearchCV(clf, params, cv=loo_cv)
gridSearchCV.fit(X_train_pca, y_train)
print("Grid search fitted..")
print(gridSearchCV.best_params_)
print(gridSearchCV.best_score_)
print("grid search cross validation score:{:.2f}".format(gridSearchCV.score(X_test_pca, y_test)))
"""
lr = LogisticRegression(C = 1.0, penalty = "l2")
lr.fit(X_train_pca, y_train)
print("lr score:{:.2f}".format(lr.score(X_test_pca, y_test)))

# precision recall curves
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier

Target = label_binarize(target, classes = range(40))
print(Target.shape)
print(Target[0])

n_classes = Target.shape[1]
X_train_multiclass, X_test_multiclass, y_train_multiclass, y_test_multiclass = train_test_split(X, 
                                                                                              Target,
                                                                                             test_size = 0.3,
                                                                                             stratify = Target,
                                                                                     random_state = 0)
pca = PCA(n_components = n_components, whiten=True)
pca.fit(X_train_multiclass)

X_train_multiclass_pca = pca.transform(X_train_multiclass)
X_test_multiclass_pca = pca.transform(X_test_multiclass)

X_train_multiclass_pca.shape
X_test_multiclass_pca.shape

oneRestClassifier = OneVsRestClassifier(lr)

oneRestClassifier.fit(X_train_multiclass_pca, y_train_multiclass)
y_score = oneRestClassifier.decision_function(X_test_multiclass_pca)

# For each class
precision = dict()
recall = dict()
average_precision = dict()
for i in range(n_classes):
    precision[i], recall[i], _ = metrics.precision_recall_curve(y_test_multiclass[:, i],
                                                        y_score[:, i])
    average_precision[i] = metrics.average_precision_score(y_test_multiclass[:, i], y_score[:, i])

# A "micro-average": quantifying score on all classes jointly
precision["micro"], recall["micro"], _ = metrics.precision_recall_curve(y_test_multiclass.ravel(),
    y_score.ravel())
average_precision["micro"] = metrics.average_precision_score(y_test_multiclass, y_score,
                                                     average="micro")
print('Average precision score, micro-averaged over all classes: {0:0.2f}'
      .format(average_precision["micro"]))

from sklearn.utils.fixes import signature

step_kwargs = ({'step': 'post'}
               if 'step' in signature(plt.fill_between).parameters
               else {})
plt.figure(1, figsize=(12,8))
plt.step(recall['micro'], precision['micro'], color='b', alpha=0.2,
         where='post')
plt.fill_between(recall["micro"], precision["micro"], alpha=0.2, color='b',
                 **step_kwargs)

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title(
    'Average precision score, micro-averaged over all classes: AP={0:0.2f}'
    .format(average_precision["micro"]))
plt.show()


 

