import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# 데이터 입력 과정
file = pd.read_excel('D:/development/HEA/20212nd/data/beforeCovid.xlsx', header=0)
file2 = pd.read_excel('D:/development/HEA/20212nd/data/afterCovid.xlsx', header=0)

# 코로나 이전
dlspi_sa = file['dlspi_sa']
dlcbd_sa = file['dlcbd_sa']
dlcpi = file['dlcpi_sa']
dlm2 = file['dlm2']
dlmmi = file['dlmmi']

train_input = pd.DataFrame({
    'dlcpd': dlcbd_sa, 'dlcpi': dlcpi, 'dlm2': dlm2, 'dlmmi': dlmmi
})

train_output = pd.DataFrame({
    'dlspi': dlspi_sa
})

# 데이터 확인
print("training input data")
print( train_input)

print()
print()
print("training output data")
print(train_output)

train_input = train_input.to_numpy()
train_output = train_output.to_numpy()

# 코로나 이후
dlspi_sa = file2['dlspi_sa']
dlcbd_sa = file2['dlcbd_sa']
dlcpi = file2['dlcpi_sa']
dlm2 = file2['dlm2']
dlmmi = file2['dlmmi']

test_input = pd.DataFrame({
    'dlcpd': dlcbd_sa, 'dlcpi': dlcpi, 'dlm2': dlm2, 'dlmmi': dlmmi
})

test_output = pd.DataFrame({
    'dlspi': dlspi_sa
})

print()
print()
print("test input data")
print(test_input)
print()
print()
print("test output data")
print(test_output)

test_input = test_input.to_numpy()
test_output = test_output.to_numpy()
print()
print()
# 데이터 표준화 평균 0 표준편차 1
scalerX = StandardScaler()
scalerX.fit(train_input)
train_input_scaled = scalerX.transform(train_input)
test_input_scaled = scalerX.transform(test_input)

scalerY = StandardScaler()
scalerY.fit(train_output)
train_output_scaled = scalerY.transform(train_output)
test_output_scaled = scalerY.transform(test_output)

# 적절한 kernel 확인
C_ep_ga_data = pd.DataFrame(columns=('Kernel', 'C', 'epsilon', 'gamma', 'training r2', 'test r2', 'r2_sum'))

# 커널 방식 비교
training_r2score = []
test_r2score = []
Kernel_settings = ['linear', 'poly', 'rbf']
C_settings = [1, 100]
epsilon_settings = [0.1]
gamma_settings = [0.1]

for Kernel in Kernel_settings:
    for C in C_settings:
        for epsilon in epsilon_settings:
            for gamma in gamma_settings:
                # build the model
                reg = SVR(C=C, kernel=Kernel, epsilon=epsilon, gamma=gamma)
                reg.fit(train_input_scaled, train_output_scaled.ravel())

                # r2 on the training set
                train_output_pred = scalerY.inverse_transform(reg.predict(train_input_scaled).reshape(-1, 1))
                train_score = r2_score(train_output, train_output_pred)
                training_r2score.append(train_score)

                # r2 on the test set
                test_output_pred = scalerY.inverse_transform(reg.predict(test_input_scaled).reshape(-1, 1))
                test_score = r2_score(test_output, test_output_pred)
                test_r2score.append(test_score)

                total_score = train_score + test_score

                i = [Kernel, C, epsilon, gamma, train_score, test_score, total_score]
                C_ep_ga_data.loc[len(C_ep_ga_data)] = i

print()
print()
print("커널 비교 과정")
print(C_ep_ga_data)
print()

# 커널 확정, 정규화의 parameter 비교
C_ep_ga_data = pd.DataFrame(columns=('C', 'epsilon', 'gamma', 'training r2', 'test r2', 'r2_sum'))
training_r2score = []
test_r2score = []
C_settings = [1, 5, 90, 95, 100, 105, 110]
epsilon_settings = [0.001, 0.01, 0.1]
gamma_settings = [0.01, 0.1]
maximum = 0
max_para = []

for C in C_settings:
    for epsilon in epsilon_settings:
        for gamma in gamma_settings:
            # build the model
            reg = SVR(C=C, kernel=Kernel, epsilon=epsilon, gamma=gamma)
            reg.fit(train_input_scaled, train_output_scaled.ravel())

            # r2 on the training set
            train_output_pred = scalerY.inverse_transform(reg.predict(train_input_scaled).reshape(-1, 1))
            train_score = r2_score(train_output, train_output_pred)
            training_r2score.append(train_score)

            # r2 on the test set
            test_output_pred = scalerY.inverse_transform(reg.predict(test_input_scaled).reshape(-1, 1))
            test_score = r2_score(test_output, test_output_pred)
            test_r2score.append(test_score)
            total_score = train_score + test_score

            i = [C, epsilon, gamma, train_score, test_score, total_score]
            if total_score > maximum:
                maximum = total_score
                max_para = reg.get_params()
            C_ep_ga_data.loc[len(C_ep_ga_data)] = i
print(C_ep_ga_data)
print(max_para)
print()

final_svr = SVR(C=95,gamma=0.1,epsilon=0.1,kernel='rbf')
final_svr.fit(train_input_scaled, train_output_scaled.ravel())

final_result_pred_scaled = final_svr.predict(test_input_scaled)
final_result_pred = scalerY.inverse_transform(final_result_pred_scaled.reshape(-1, 1))

rmse = mean_squared_error(test_output, final_result_pred)
print(final_result_pred)
print()

mae = mean_absolute_error(test_output, final_result_pred)
print("rmse: ", rmse)
print("mae: ", mae)

print()
print()
print()
# 커널 확정, 정규화 안하고 구하기
training_r2score = []
test_r2score = []
C_settings = [1, 10, 50, 80, 90, 100]
epsilon_settings = [0.001, 0.01, 0.1]
gamma_settings = [0.01, 0.1]
maximum = 0
max_para = []

for C in C_settings:
    for epsilon in epsilon_settings:
        for gamma in gamma_settings:
            # build the model
            reg = SVR(C=C, kernel=Kernel, epsilon=epsilon, gamma=gamma)
            reg.fit(train_input, train_output.ravel())

            # r2 on the training set
            train_output_pred = reg.predict(train_input)
            train_score = r2_score(train_output, train_output_pred)
            training_r2score.append(train_score)

            # r2 on the test set
            test_output_pred = reg.predict(test_input)
            test_score = r2_score(test_output, test_output_pred)
            test_r2score.append(test_score)

            total_score = train_score + test_score

            i = [C, epsilon, gamma, train_score, test_score, total_score]
            if total_score > maximum:
                maximum = total_score
                max_para = reg.get_params()

            C_ep_ga_data.loc[len(C_ep_ga_data)] = i
print(C_ep_ga_data)
print(max_para)
print()


final_svr = SVR(C=90,gamma=0.1,epsilon=0.01,kernel='rbf')
final_svr.fit(train_input, train_output.ravel())

final_result_pred = final_svr.predict(test_input)

rmse = mean_squared_error(test_output, final_result_pred)
print(final_result_pred)
print()
mae = mean_absolute_error(test_output, final_result_pred)
print("rmse: ", rmse)
print("mae: ", mae)
