import numpy as np


#data= length of train data
def  split_train_test(data,K):
    index=[]
    set=[]
    shuffled_indices = np.random.permutation(data)
    fold_size = int(data//K)
    reminder=data%K
    for i in range(reminder):
        fold= shuffled_indices[i*fold_size+i:(i+1)*fold_size+i+1]
        set.append(fold)
    for i in range(reminder,K):
        fold = shuffled_indices[i * fold_size + reminder:(i + 1) * fold_size + reminder ]
        set.append(fold)
    for j in range(len(set)):
        train_indicis = []
        test_indicies = []
        for a in range(len(set)):
            if a != j:
                for k in set[a]:
                    train_indicis.append(k)
            if a == j:
                for k in set[a]:
                    test_indicies.append(k)
        index.append([train_indicis, test_indicies])
    return index







def compute_error(test_y_hat,test_y_CV):
    import pandas as pd
    df=pd.DataFrame()
    df["y"]=test_y_CV
    df["y_hat"]=test_y_hat
    return df.query('y!=y_hat').shape[0]/test_y_CV.shape[0]

def find_error(model,train_x_CV,train_y_CV,test_x_CV,test_y_CV):
    model.fit(train_x_CV, train_y_CV)
    test_y_hat=model.predict(test_x_CV)
    return compute_error(test_y_hat,test_y_CV)

def cal_average(num):
    sum_num = 0
    for t in num:
        sum_num = sum_num + t

    avg = sum_num / len(num)
    return avg


def find_best(models, train_x, train_y,k):
    lst=[]
    for model in models:
        List=[]
        for train_index, test_index in split_train_test(len(train_x),k):
            train_x_CV, train_y_CV,test_x_CV,test_y_CV = train_x[train_index], train_y[train_index],train_x[test_index],train_y[test_index]
            error=find_error(model, train_x_CV, train_y_CV, test_x_CV, test_y_CV)
            List.append(error)
        average_error=cal_average(List)
        lst.append(average_error)

    for j in range(0,len(models)):
        if lst[j]==min(lst):
            print(models[j])
