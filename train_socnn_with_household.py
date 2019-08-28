import pandas as pd
import os
import numpy as np
from socnn import build_socnn
from sklearn.model_selection import train_test_split
from scipy import stats

limit=np.inf
X = pd.read_csv(os.path.join('household_power_consumption.txt'), sep=';',
                parse_dates={'datetime': [0, 1]}, dtype=np.float32,
                na_values=['?'], nrows=10000)
                #limit if (limit < np.inf) else None)
X['time'] = X['datetime'].apply(lambda x: x.hour * 60 + x.minute)

#data preprocessing
X_nonan = X.interpolate(limit_direction='both')
X_nonan = X_nonan.iloc[:,1:8].apply(stats.zscore, axis=0)
X_nonan['time'] = X['datetime'].apply(lambda x: x.hour * 60 + x.minute)
X = X_nonan

len_time = 128
dim = 8 # inculded duration


### make datasets###
input_data = []
output_data = []

for i in range(len(X)-128):
    buff = []
    for j in range(len_time):
        buff.append([X['Global_active_power'][i+j],
                     X['Global_reactive_power'][i+j],
                     X['Voltage'][i+j],
                     X['Global_intensity'][i+j],
                     X['Sub_metering_1'][i+j],
                     X['Sub_metering_2'][i+j],
                     X['Sub_metering_3'][i+j],
                     X['time'][i+j]])
    input_data.append(buff)
    output_data.append([X['Global_active_power'][i+len_time],
                  X['Global_reactive_power'][i+len_time],
                  X['Voltage'][i+len_time],
                  X['Global_intensity'][i+len_time],
                  X['Sub_metering_1'][i+len_time],
                  X['Sub_metering_2'][i+len_time],
                  X['Sub_metering_3'][i+len_time]])

input_data = np.array(input_data)
output_data = np.array(output_data)

output_data = np.reshape(output_data, (-1, 1, dim-1))

train_input_data, test_input_data, train_output_data, test_output_data = train_test_split(input_data, output_data,
                                                                                          test_size=0.2, shuffle=False)
#exclude time
train_v_input = np.delete(train_input_data,dim-1 ,2)
test_v_input = np.delete(test_input_data,dim-1 ,2)


#train and test socnn

model = build_socnn(input_shape_sig=(len_time, dim), input_shape_off=(len_time, dim-1), dim=dim)
history = model.fit([train_input_data, train_v_input], [train_output_data, train_v_input],
                     verbose=1, epochs=100)
score = model.evaluate([test_input_data, test_v_input], [test_output_data, test_v_input],
                       verbose=0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])