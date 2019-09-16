import fire
import pickle
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler

def unpickle(file):
    with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='latin1')
    return data

def load_cifar():
    train_data, train_labels, test_data, test_labels = [], [], [], []
    for i in range(1):
        batchName = "./data/data_batch_"+str(i+1)
        unpickled = unpickle(batchName)
        train_data.extend(unpickled['data'])
        train_labels.extend(unpickled['labels'])
    unpickled = unpickle('./data/test_batch')
    test_data.extend(unpickled['data'])
    test_labels.extend(unpickled['labels'])
    return train_data, train_labels, test_data, test_labels

def preprocess(train_data, test_data):
    min_max_scaler = MinMaxScaler()
    train_data_minmax = min_max_scaler.fit_transform(train_data)
    test_data_minmax = min_max_scaler.transform(test_data)
    return train_data_minmax, test_data_minmax

def reduce_dim(train_data, train_labels, test_data, data_representation, principal_components):
    if data_representation == 'raw-data':
        reduced_train_data = train_data
        reduced_test_data = test_data
    elif data_representation == 'pca':
        reduction_model = PCA(n_components=principal_components)
        reduced_train_data = reduction_model.fit_transform(train_data)
        reduced_test_data = reduction_model.transform(test_data)
    elif data_representation == 'lda':
        reduction_model = LDA()
        reduced_train_data = reduction_model.fit_transform(train_data, train_labels)
        reduced_test_data = reduction_model.transform(test_data)
    else:
        print(data_representation + ": Data representation not implemented")
        exit(0)
    return reduced_train_data, reduced_test_data

def classify(X, y, classifier_type, C, solver, log_gamma, log_C, max_depth):
    if classifier_type == 'linear-svm':
        model = SVC(kernel='linear',C=C)
    elif classifier_type == 'logistic-regression':
        model = LogisticRegression(solver=solver,multi_class='auto')
    elif classifier_type == 'rbf-svm':
        gamma = 2**log_gamma
        C = 2**log_C
        model = SVC(kernel='rbf', gamma=gamma, C=C)
    elif classifier_type == 'decision-tree':
        model = DecisionTreeClassifier(max_depth=max_depth)
    else:
        print(classifier_type + ": Classifier type not implemented")
        exit(0)
    model.fit(X,y)
    return model

def evaluate(target, predicted):
    f1 = f1_score(target, predicted, average='micro')
    acc = accuracy_score(target, predicted)
    return f1, acc

def main(num_batches=10, data_representation='pca', num_components=209,
    classifier_type='linear-svm', C=1.0, solver='sag', log_gamma=0, log_C=0, max_depth=1):
    full_train_data, full_train_labels, full_test_data, full_test_labels = load_cifar()
    print("loaded")
    if classifier_type=='rbf-svm':
        full_train_data, full_test_data = preprocess(full_train_data, full_test_data)
        print("preprocessed")
    full_train_data, full_test_data = reduce_dim(full_train_data, full_train_labels, full_test_data, data_representation, num_components)
    print("dimension reduction completed")
    accuracy_sum = 0.0
    batch_size = len(full_train_data)//num_batches
    for i in range(num_batches):
        print("-----------------------Training on Batch " + str(i+1) + "-------------------------")
        train_data = full_train_data[i*batch_size:(i+1)*batch_size]
        train_labels = full_train_labels[i*batch_size:(i+1)*batch_size]
        model = classify(train_data,train_labels,classifier_type,C,solver,log_gamma,log_C,max_depth)
        print("model created")
        pred = model.predict(full_test_data)
        print("predictions completed")
        f1, acc = evaluate(full_test_labels,pred)
        accuracy_sum += acc
        print("Results - F1 score: {}, Accuracy: {}".format(f1, acc))
    print("--------------------Average accuracy----------------------")
    print(accuracy_sum/num_batches)

if __name__ == '__main__':
    fire.Fire(main)
