import numpy as np
import pandas as pd

path = 'D:\credit 2017\crx.data'
def data_manipulate(x):
    df = pd.read_csv(x, names=range(16))
    cate_col = [0, 3, 4, 5, 6, 8, 9, 11, 12]
    num_col = [1, 2, 7, 10, 13, 14]
    # replace '?' with mean of the column ignoring missing value.
    for i in num_col:
        df[i] = pd.to_numeric(df[i], errors='coerce')
        mean_num = df[i].mean()
        df[i] = df[i].fillna(mean_num)
    # replace '?' with midian of the column ignoring miss value
    for j in cate_col:
        midian_cat = df[j].mode()
        df[j] = df[j].fillna(midian_cat)
    data_list = np.array(df)
    return data_list


def isnumber(str):
    try:
        float(str);
        return True;
    except ValueError:
        return False;

def data_scaling(data_list):
    aver_list = []
    std_list = []
    n =0
    for i in range(len(np.transpose(data_list))):
        if isnumber(data_list[0][i]):
            s = 0
            std = []
            for j in range(len(data_list)):
                number = float(data_list[j][i])
# count numbers by columns
                s += number
# put numbers by columns in to a list.
                std.append(number)
# record standard derivation and average of each column
            std_list.append(np.std(std))
            aver_list.append(s/len(data_list))
    for i in range(len(np.transpose(data_list))):
        # scaling all numeric features
        if isnumber(data_list[0][i]):
            for j in range(len(data_list)):
                data_list[j][i] = (float(data_list[j][i])-aver_list[n])/std_list[n]
            n += 1
    # put data after scaling into data frame
    dataFrame = pd.DataFrame(data_list)
    cate_col = [0, 3, 4, 5, 6, 8, 9, 11, 12]
    num_col = [1, 2, 7, 10, 13, 14]
    for i in num_col:

# change data type of numeric columns to numeric.

        dataFrame[i] = pd.to_numeric(dataFrame[i])
    # replace categories with vectors which expand dimension with numbers of categories.
    feature_col =pd.get_dummies(dataFrame[cate_col])
    data_list = feature_col.join(dataFrame[num_col])
    # divide feature and label into to array

    data_list_feature = np.array(data_list)
    data_list_label = np.array(dataFrame[15])

    return data_list_feature, data_list_label

def KNN_algorithm(data_list_feature, data_list_label, k):

# divide feature and label lists into training and testing.

    Test_Lebel_list = []
    data_list_feature_train = data_list_feature[:552]
    data_list_feature_test = data_list_feature[552:]
    data_list_label_train = data_list_label[:552]
    for index in range(len(data_list_feature_test)):

 # calculate the distance betwwen test data and each raw of training data.

        diff = np.tile(data_list_feature_test[index],(data_list_feature_train.shape[0],1))-data_list_feature_train
        squared_diff = diff**2
        squared_distance = np.sum(squared_diff,axis=1)
        distance = np.sqrt(squared_distance)
# sort distances and return a list of index
        sortDistance = np.argsort(distance)
        Counter = {}
# count the most common label appears in k nearest samples.
        for i in range(k):
            voteLebel = data_list_label_train[sortDistance[i]]
            Counter[voteLebel] = 1 + Counter.get(voteLebel, 0)
        for item, value in Counter.items():
            if value>= k/2:
                Test_Lebel_list.append(item)
    return Test_Lebel_list


# predict labels of test dataset, and calculate the rate of right prediction.
data = data_manipulate(path)
(data_list_feature, data_list_label) = data_scaling(data)
Test_Lebel = KNN_algorithm(data_list_feature, data_list_label, 7)
Counter = 0
data_list_label_test = data_list_label[552:]
n = len(Test_Lebel)
for i in range(n):
    if data_list_label_test[i] == Test_Lebel[i]:
        Counter += 1
Rate = Counter/n
print(Rate)



