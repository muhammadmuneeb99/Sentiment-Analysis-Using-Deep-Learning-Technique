# Imports
import re
from nltk.tokenize.treebank import TreebankWordDetokenizer
import gensim
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
import numpy as np
import pandas as pd

print('Importing Done')
##

## Imports ##
train = pd.read_csv('SAUDL - Corpus/train.csv')
##
print('\n')

## Data exploration ##
desired_width = 320
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns', 10)
print(train.head(15))
##
print('\n')

## Dataset Length ##
print(len(train))
##
print('\n')

## finding other different value than neutral, negative and positive?
print(train['sentiment'].unique())
##
print('\n')

## Checking How's distributed the dataset? Is it biased?
print(train.groupby('sentiment').nunique())
##
print('\n')

## Data Cleaning ##
train = train[['selected_text', 'sentiment']]
print(train.head())
##
print('\n')

## Checking for null value ##
print(train["selected_text"].isnull().sum())
##
print('\n')

## Filling the null value ##
print(train["selected_text"].fillna("No content", inplace=True))
##
print('\n')

## clean data method remove unnecessary Items ##
def clean_data(data):
    # Removing URLs with a regular expression
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    data = url_pattern.sub(r'', data)
    # Remove Emails
    data = re.sub('\S*@\S*\s?', '', data)
    # Remove new line characters
    data = re.sub('\s+', ' ', data)
    # Remove distracting single quotes
    data = re.sub("\'", "", data)
    return data

##
print('\n')

## Splitting of Data ##
temp = []
# Splitting pd.Series to list
data_to_list = train['selected_text'].values.tolist()
for i in range(len(data_to_list)):
    temp.append(clean_data(data_to_list[i]))
print(list(temp[:5]))
##
print('\n')

## sent_to_words converts to sentence to word ##
def sent_to_words(sentences):
    for sentence in sentences:
        yield (gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

data_words = list(sent_to_words(temp))
print(data_words[:10])
##
print('\n')

## checking the length of data_words ##
print(len(data_words))
##
print('\n')

## detokenize method convert words to sentence ##
def detokenize(text):
    return TreebankWordDetokenizer().detokenize(text)

##
print('\n')

## printing converted sentences ##
data = []
for i in range(len(data_words)):
    data.append(detokenize(data_words[i]))
print(data[:5])
##
print('\n')

## Converting to numpy array ##
data = np.array(data)
##
print('\n')

## Label encoding ##
labels = np.array(train['sentiment'])
y = []
for i in range(len(labels)):
    if labels[i] == 'neutral':
        y.append(0)
    if labels[i] == 'negative':
        y.append(1)
    if labels[i] == 'positive':
        y.append(2)
y = np.array(y)
labels = tf.keras.utils.to_categorical(y, 3, dtype="float32")
del y
##
print('\n')

## checking the length of labels ##
print(len(labels))
##
print('\n')

## Data sequencing and splitting ##
from keras.models import Sequential
from keras import layers
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.callbacks import ModelCheckpoint

max_words = 5000
max_len = 200

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(data)
sequences = tokenizer.texts_to_sequences(data)
tweets = pad_sequences(sequences, maxlen=max_len)
print(tweets)
##
print('\n')

## printing the labels
print(labels)
##
print('\n')

## Splitting the data ##
X_train, X_test, y_train, y_test = train_test_split(tweets, labels, random_state=0)
print(len(X_train), len(X_test), len(y_train), len(y_test))
##
print('\n')

## no of epochs ##
no_epoch = 70
##

## Model building ##

## Single LSTM layer model (Start)
model1 = Sequential()
model1.add(layers.Embedding(max_words, 20))
model1.add(layers.LSTM(15, dropout=0.5))
model1.add(layers.Dense(3, activation='softmax'))

model1.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
# Implementing model checkpoins to save the best metric and do not lose it on training.
checkpoint1 = ModelCheckpoint("best_model1.hdf5", monitor='val_accuracy', verbose=1, save_best_only=True,
                              save_weights_only=False, mode='auto', save_freq='epoch')
history = model1.fit(X_train, y_train, epochs=no_epoch, validation_data=(X_test, y_test), callbacks=[checkpoint1])
print(history)
## Single LSTM layer model (End)
print('\n')

## Bidirectional LTSM model (Start)
model2 = Sequential()
model2.add(layers.Embedding(max_words, 40, input_length=max_len))
model2.add(layers.Bidirectional(layers.LSTM(20, dropout=0.6)))
model2.add(layers.Dense(3, activation='softmax'))
model2.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
# Implementing model checkpoins to save the best metric and do not lose it on training.
checkpoint2 = ModelCheckpoint("best_model2.hdf5", monitor='val_accuracy', verbose=1, save_best_only=True,
                              save_weights_only=False, mode='auto', save_freq='epoch')
history = model2.fit(X_train, y_train, epochs=no_epoch, validation_data=(X_test, y_test), callbacks=[checkpoint2])
print(history)
## Bidirectional LTSM model (End)
print('\n')

## 1D Convolutional model (Start)
from keras import regularizers

model3 = Sequential()
model3.add(layers.Embedding(max_words, 40, input_length=max_len))
model3.add(layers.Conv1D(20, 6, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=2e-3, l2=2e-3),
                         bias_regularizer=regularizers.l2(2e-3)))
model3.add(layers.MaxPooling1D(5))
model3.add(layers.Conv1D(20, 6, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=2e-3, l2=2e-3),
                         bias_regularizer=regularizers.l2(2e-3)))
model3.add(layers.GlobalMaxPooling1D())
model3.add(layers.Dense(3, activation='softmax'))
model3.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
checkpoint3 = ModelCheckpoint("best_model3.hdf5", monitor='val_accuracy', verbose=1, save_best_only=True,
                              save_weights_only=False, mode='auto', save_freq='epoch')
history = model3.fit(X_train, y_train, epochs=no_epoch, validation_data=(X_test, y_test), callbacks=[checkpoint3])
print(history)
## 1D Convolutional model (End)
print('\n')

## Best model validation ##
best_model = keras.models.load_model("best_model2.hdf5")
test_loss, test_acc = best_model.evaluate(X_test, y_test, verbose=2)
print('Model accuracy: ', test_acc)
predictions = best_model.predict(X_test)
print(predictions)
##
print('\n')

## Confusion matrix ##
from sklearn.metrics import confusion_matrix

matrix = confusion_matrix(y_test.argmax(axis=1), np.around(predictions, decimals=0).argmax(axis=1))
import seaborn as sns

conf_matrix = pd.DataFrame(matrix, index=['Neutral', 'Negative', 'Positive'],
                           columns=['Neutral', 'Negative', 'Positive'])
# Normalizing
conf_matrix = conf_matrix.to_numpy() / conf_matrix.to_numpy().sum(axis=1)[:, np.newaxis]
plt.figure(figsize=(15, 15))
sns.heatmap(conf_matrix, annot=True, annot_kws={"size": 15})
plt.show()
##
print('\n')

## Performance on Some Testing Text
sentiment = ['Neutral', 'Negative', 'Positive']

sequence = tokenizer.texts_to_sequences(['this experience has been the worst , want my money back'])
test = pad_sequences(sequence, maxlen=max_len)
print(sentiment[np.around(best_model.predict(test), decimals=0).argmax(axis=1)[0]])
print('\n')

sequence = tokenizer.texts_to_sequences(['this data science article is the best ever'])
test = pad_sequences(sequence, maxlen=max_len)
print(sentiment[np.around(best_model.predict(test), decimals=0).argmax(axis=1)[0]])
print('\n')

sequence = tokenizer.texts_to_sequences(['i hate youtube ads, they are annoying'])
test = pad_sequences(sequence, maxlen=max_len)
print(sentiment[np.around(best_model.predict(test), decimals=0).argmax(axis=1)[0]])
print('\n')

sequence = tokenizer.texts_to_sequences(['i really loved how the technician helped me with the issue that i had'])
test = pad_sequences(sequence, maxlen=max_len)
print(sentiment[np.around(best_model.predict(test), decimals=0).argmax(axis=1)[0]])
##

print('\n')
