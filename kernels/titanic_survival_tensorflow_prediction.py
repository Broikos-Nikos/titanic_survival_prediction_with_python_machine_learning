import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from collections import namedtuple
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

# ==================== Data Insertion and Editing ===============
# ========== Insert Data Files ==========
data_directory="../data/"
train_data = pd.read_csv(data_directory+"train.csv")
test_data = pd.read_csv(data_directory+"test.csv")
data = train_data.append(test_data)
test_passenger_id=test_data["PassengerId"]
# ========== Insert Data Files ==========

data['HasCabin'] = data["Cabin"].apply(lambda x: 0 if type(x) == float else 1)
data['FamilySize'] = data['Parch'] + data['SibSp']

data['Last_Name'] = data['Name'].apply(lambda x: str.split(x, ",")[0])
data['Fare'].fillna(data['Fare'].mean(), inplace=True)
DEFAULT_SURVIVAL_VALUE = 0.5
data['Family_Survival'] = DEFAULT_SURVIVAL_VALUE


for grp, grp_df in data[['Survived','Name', 'Last_Name', 'Fare', 'Ticket', 'PassengerId',
                           'SibSp', 'Parch', 'Age', 'Cabin']].groupby(['Last_Name', 'Fare']):   
    if (len(grp_df) != 1):
        for ind, row in grp_df.iterrows():
            smax = grp_df.drop(ind)['Survived'].max()
            smin = grp_df.drop(ind)['Survived'].min()
            passID = row['PassengerId']
            if (smax == 1.0):
                data.loc[data['PassengerId'] == passID, 'Family_Survival'] = 1
            elif (smin==0.0):
                data.loc[data['PassengerId'] == passID, 'Family_Survival'] = 0
print("Number of passengers with family survival information:", 
      data.loc[data['Family_Survival']!=0.5].shape[0])



for _, grp_df in data.groupby('Ticket'):
    if (len(grp_df) != 1):
        for ind, row in grp_df.iterrows():
            if (row['Family_Survival'] == 0) | (row['Family_Survival']== 0.5):
                smax = grp_df.drop(ind)['Survived'].max()
                smin = grp_df.drop(ind)['Survived'].min()
                passID = row['PassengerId']
                if (smax == 1.0):
                    data.loc[data['PassengerId'] == passID, 'Family_Survival'] = 1
                elif (smin==0.0):
                    data.loc[data['PassengerId'] == passID, 'Family_Survival'] = 0                      
print("Number of passenger with family/group survival information: " 
      +str(data[data['Family_Survival']!=0.5].shape[0]))
train_data['Family_Survival'] = data['Family_Survival'][:891]
test_data['Family_Survival'] = data['Family_Survival'][891:]
train_data['HasCabin'] = data['HasCabin'][:891]
test_data['HasCabin'] = data['HasCabin'][891:]
train_data['FamilySize'] = data['FamilySize'][:891]
test_data['FamilySize'] = data['FamilySize'][891:]






# ========== Clearing Data Functions ==========
def drop_not_concerned(dataframe, columns):
    return dataframe.drop(columns, axis=1)
def clean_concerned(dataframe, columns):
    return dataframe[columns].replace(np.nan, dataframe[columns].mean())
# ========== Clearing Data Functions ==========



# ========== Editing and Normalizing Data ==========
concerned_columns = ["Survived","Age", "Sex", "Fare", "Pclass", 'Family_Survival', 'FamilySize']
train_data = clean_concerned(train_data, concerned_columns)
concerned_columns.remove("Survived")
test_data = clean_concerned(test_data, concerned_columns)



# ========== Binarizing Sex field ==========
def sex_to_int(data):
    le = LabelEncoder()
    le.fit(["male","female"])
    data["Sex"]=le.transform(data["Sex"]) 
    return data

train_data = sex_to_int(train_data)
test_data = sex_to_int(test_data)
# ========== Binarizing Sex field ==========



# ========== Dummy categories cration ==========
def dummy_data(data, columns):
    for column in columns:
        data = pd.concat([data, pd.get_dummies(data[column], prefix=column)], axis=1)
        data = data.drop(column, axis=1)
    return data

dummy_columns = ["Pclass", "Sex"]
train_data=dummy_data(train_data, dummy_columns)
test_data=dummy_data(test_data, dummy_columns)
# ========== Dummy categories cration ==========



# ========== Editing and Normalizing Data  ==========
def normalize_data(dataframe, columns):
    scaler = MinMaxScaler()
    dataframe[columns] = scaler.fit_transform(dataframe[columns].values.reshape(-1,1))
    return dataframe
train_data = normalize_data(train_data, "Fare")
test_data = normalize_data(test_data, "Fare")
train_data = normalize_data(train_data, "Age")
test_data = normalize_data(test_data, "Age")
# ========== Editing and Normalizing Data  ==========



# ========== Ploting a heatmap for relations ==========
from matplotlib.pyplot import figure
figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
g = sns.heatmap(train_data.corr(),annot=True, fmt = ".2f", cmap = "coolwarm")
plt.show()
# ========== Ploting a heatmap for relations ==========







# ========== Splitting Data ==========
def split_valid_test_data(data, fraction=(1 - 0.8)):
    data_y = data["Survived"]
    lb = LabelBinarizer()
    data_y = lb.fit_transform(data_y)
    data_x = data.drop(["Survived"], axis=1)
    train_x, valid_x, train_y, valid_y = train_test_split(data_x, data_y, test_size=fraction)
    return train_x.values, train_y, valid_x, valid_y

train_x, train_y, valid_x, valid_y = split_valid_test_data(train_data)
# ========== Splitting Data ==========
# ==================== Data Insertion and Editing ===============



# ==================== Algorythm ====================
# ========== Build Neural Network ==========
def build_neural_network(hidden_units=10):
    tf.reset_default_graph()
    inputs = tf.placeholder(tf.float32, shape=[None, train_x.shape[1]])
    labels = tf.placeholder(tf.float32, shape=[None, 1])
    learning_rate = tf.placeholder(tf.float32)
    is_training=tf.Variable(True,dtype=tf.bool)
    
    initializer = tf.contrib.layers.xavier_initializer()
    fc = tf.layers.dense(inputs, hidden_units, activation=None,kernel_initializer=initializer)
    fc=tf.layers.batch_normalization(fc, training=is_training)
    fc=tf.nn.relu(fc)
    
    logits = tf.layers.dense(fc, 1, activation=None)
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
    cost = tf.reduce_mean(cross_entropy)
    
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    predicted = tf.nn.sigmoid(logits)
    correct_pred = tf.equal(tf.round(predicted), labels)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Export the nodes 
    export_nodes = ['inputs', 'labels', 'learning_rate','is_training', 'logits',
                    'cost', 'optimizer', 'predicted', 'accuracy']
    Graph = namedtuple('Graph', export_nodes)
    local_dict = locals()
    graph = Graph(*[local_dict[each] for each in export_nodes])

    return graph

model = build_neural_network()



def get_batch(data_x,data_y,batch_size=32):
    batch_n=len(data_x)//batch_size
    for i in range(batch_n):
        batch_x=data_x[i*batch_size:(i+1)*batch_size]
        batch_y=data_y[i*batch_size:(i+1)*batch_size]
        
        yield batch_x,batch_y



epochs = 50
train_collect = 50
train_print=train_collect*10

learning_rate_value = 0.001
batch_size=16

x_collect = []
train_loss_collect = []
train_acc_collect = []
valid_loss_collect = []
valid_acc_collect = []
# ========== Build Neural Network ==========

# ========== Start Trainning of Neural Network ==========
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    iteration=0
    for e in range(epochs):
        for batch_x,batch_y in get_batch(train_x,train_y,batch_size):
            iteration+=1
            feed = {model.inputs: train_x, model.labels: train_y, model.learning_rate: learning_rate_value, model.is_training:True }

            train_loss, _, train_acc = sess.run([model.cost, model.optimizer, model.accuracy], feed_dict=feed)
            
            if iteration % train_collect == 0:
                x_collect.append(e)
                train_loss_collect.append(train_loss)
                train_acc_collect.append(train_acc)

                if iteration % train_print==0:
                     print("Epoch: {}/{}".format(e + 1, epochs), "Train Loss: {:.4f}".format(train_loss), "Train Acc: {:.4f}".format(train_acc))
                        
                feed = {model.inputs: valid_x, model.labels: valid_y, model.is_training:False}
                val_loss, val_acc = sess.run([model.cost, model.accuracy], feed_dict=feed)
                valid_loss_collect.append(val_loss)
                valid_acc_collect.append(val_acc)
                
                if iteration % train_print==0:
                    print("Epoch: {}/{}".format(e + 1, epochs), "Validation Loss: {:.4f}".format(val_loss), "Validation Acc: {:.4f}".format(val_acc))
                
    print("Training end with validation accuracy: ", val_acc, " trainning accuracy: ", train_acc)
    saver.save(sess, "../tensorflow_data/titanic.ckpt")
# ========== End Trainning of Neural Network ==========
# ==================== Algorythm ====================


fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
fig.suptitle('Training Results')

axes[0].set_ylabel("Accuracy", fontsize=14)
axes[1].set_xlabel("Epoch", fontsize=14)
axes[0].plot(train_acc_collect)
axes[0].plot(valid_acc_collect)

axes[1].set_ylabel("Loss", fontsize=14)
axes[1].set_xlabel("Epoch", fontsize=14)
axes[1].plot(train_loss_collect)
axes[1].plot(valid_loss_collect)
plt.show()



# ==================== Evaluation ====================
# ========== Load the trainned neural network ==========
model=build_neural_network()
restorer=tf.train.Saver()
with tf.Session() as sess:
    restorer.restore(sess,"../tensorflow_data/titanic.ckpt")
    feed={model.inputs:test_data, model.is_training:False}
    test_predict=sess.run(model.predicted,feed_dict=feed)
test_predict=pd.DataFrame(np.float_(test_predict))
test_predict.fillna(test_predict.mean(), inplace=True)
from sklearn.preprocessing import Binarizer
binarizer=Binarizer(0.5)
test_predict_result=binarizer.fit_transform(test_predict)
test_predict_result=test_predict_result.astype(np.int32)
# ========== Load the trainned neural network ==========



# ========== Create file for evaluation ==========
passenger_id=test_passenger_id.copy()
prediction=passenger_id.to_frame()
prediction["Survived"]=test_predict_result
prediction.to_csv(data_directory+"prediction_submission.csv",index=False)
# ========== Create file for evaluation ==========
# ==================== Evaluation ====================


