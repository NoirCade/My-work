import pandas as pd
import numpy as np
import ktrain
from ktrain import text
from sklearn.model_selection import train_test_split
import time


whole_data = pd.read_csv('train.csv', encoding='utf-8')
data_train, data_valid = train_test_split(whole_data, test_size=0.1, random_state=99)


X_train = data_train.text.tolist()
X_test = data_valid.text.tolist()

y_train = data_train.target.tolist()
y_test = data_valid.target.tolist()

data = data_train.append(data_valid, ignore_index=True)

class_names = ['disaster', 'non-disaster']

print('size of training set: %s' % (len(data_train['text'])))
print('size of validation set: %s' % (len(data_valid['text'])))
print(data.target.value_counts())

# print(data.head(10))

encoding = {
    'non-disaster': 0,
    'disaster': 1
}

# Integer values for each class
# y_train = [encoding[x] for x in y_train]
# y_test = [encoding[x] for x in y_test]

(x_train,  y_train), (x_test, y_test), preproc = text.texts_from_array(x_train=X_train, y_train=y_train,
                                                                       x_test=X_test, y_test=y_test,
                                                                       class_names=class_names,
                                                                       preprocess_mode='bert',
                                                                       maxlen=350,
                                                                       max_features=35000)

model = text.text_classifier('bert', train_data=(x_train, y_train), preproc=preproc)

learner = ktrain.get_learner(model, train_data=(x_train, y_train),
                             val_data=(x_test, y_test),
                             batch_size=6)

learner.fit_onecycle(2e-5, 3)
learner.validate(val_data=(x_test, y_test), class_names=class_names)

predictor = ktrain.get_predictor(learner.model, preproc)
predictor.get_classes()

message = 'I just broke up with my boyfriend'

start_time = time.time()
prediction = predictor.predict(message)

print('predicted: {} ({:.2f})'.format(prediction, (time.time() - start_time)))

predictor.save("models/bert_model")
