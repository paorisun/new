from math import sqrt
from os import listdir, path
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from numpy.random import Generator, PCG64

def vec_len(v):
    sum = 0
    for component in v:
        sum += component ** 2
    return sqrt(sum)

def load_dataset(filename):
    #csv 데이터 로드
    df = pd.read_csv(filename, index_col=0, usecols=range(4))
    #시간 데이터 소숫점 버리기
    df.index = df.index.map(lambda x: int(x))
    #10초까지 자르기
    df = df.loc[df.index < 10]
    #합력 계산
    df = df.apply(lambda d: vec_len(d), axis="columns", result_type="expand")
    #평균 groupby
    average = df.groupby(df.index.name).mean()
    #numpy 배열로 변환
    return np.array(average)

def load_datasets_in_folder(folder):
    datasets = np.empty((0, 10))
    file_names = []
    for f in listdir(folder):
        full_path = path.join(folder, f)
        if not path.isfile(full_path) or path.splitext(full_path)[1] != ".csv":
            continue
        dataset = load_dataset(full_path)
        if len(dataset) != 10:
            print(f'파일 로드 오류: {full_path}, dataset의 크기가 10이 아니고 {len(dataset)} 입니다. 무시합니다.')
            continue
        print(f'데이터 로드 성공: {full_path}')
        datasets = np.vstack((datasets, dataset))
        file_names.append(f)
    return (datasets, file_names)


def load_training_data():
    #데이터셋 정의
    X = np.empty((0, 10))
    Y = np.empty((0, 3))
    #뜀 데이터 로드
    print("뜀 데이터를 로드합니다.")
    run_data = load_datasets_in_folder("./dataset_run")[0]
    X = np.vstack((X, run_data))
    Y = np.vstack((Y, np.repeat([[1, 0, 0]], len(run_data), axis=0)))
    #걸음 데이터 로드
    print("걸음 데이터를 로드합니다.")
    walk_data = load_datasets_in_folder("./dataset_walk")[0]
    X = np.vstack((X, walk_data))
    Y = np.vstack((Y, np.repeat([[0, 1, 0]], len(walk_data), axis=0)))
    print("가만히 데이터를 로드합니다.")
    still_data = load_datasets_in_folder("./dataset_still")[0]
    X = np.vstack((X, still_data))
    Y = np.vstack((Y, np.repeat([[0, 0, 1]], len(still_data), axis=0)))
    return (X, Y)

# 입력할 데이터
# X = np.array([[1.029815, 1.51113, 1.890595, 2.36834, 2.395795, 2.003316583, 2.18814, 1.843235, 1.59085, 1.21185],
#               [0.029815, 0.51113, 0.890595, 0.36834, 0.395795, 0.003316583, 0.18814, 0.843235, 0.59085, 0.21185]])
# Y = np.array([[0], [1]])


#StandardScaler 설정
scaler = StandardScaler()

#csv로부터 데이터 로드
X, Y = load_training_data()
#전처리
scaler = scaler.fit(X)
X = scaler.transform(X)
# X = np.round(X,2)

print(X)
print(Y)

input_size = 10
hidden_size_1 = 30
hidden_size_2 = 30
hidden_size_3 = 30
output_size = 3

rng = Generator(PCG64(seed=425028234))
# 가중치 랜덤 1, 2, 3
W1 = rng.random((input_size, hidden_size_1))
W2 = rng.random((hidden_size_1, hidden_size_2))
W3 = rng.random((hidden_size_2, hidden_size_3))
W4 = rng.random((hidden_size_3, output_size))

b1 = np.zeros((1, hidden_size_1))
b2 = np.zeros((1, hidden_size_2))
b3 = np.zeros((1, hidden_size_3))
b4 = np.zeros((1, output_size))

# W1 = np.array([[0.1,0.2],
#                [0.3,0.4]])

# W2 = np.array([[0.5,0.6],
#                [0.7,0.8]])
# W3 = np.array([[0.9],
#                [0.95]])

# print("---------------------------------")
# print(W1)
# print("---------------------------------")
# print(W2)
# print("---------------------------------")
# print(W3)
# print("---------------------------------")

np.set_printoptions(precision=20, suppress=True)

# 학습률과 반복수
learning_rate = 0.01
num_iterations = 100000
error_list = []

error_10000 = []
# error_list = []
# 학습시작
for i in range(num_iterations):
    # 순전파
    z1 = np.dot(X, W1) + b1
    a1 = 1 / (1 + np.exp(-z1))
    z2 = np.dot(a1, W2) + b2
    a2 = 1 / (1 + np.exp(-z2))
    z3 = np.dot(a2, W3) + b3
    a3 = 1 / (1 + np.exp(-z3))
    z4 = np.dot(a3, W4) + b4
    y_hat = 1 / (1 + np.exp(-z4))


    #역전파
    error = Y - y_hat
    
    delta_output = error * y_hat * (1 - y_hat)
    delta_hidden_3 = delta_output.dot(W4.T) * a3 * (1 - a3)
    delta_hidden_2 = delta_hidden_3.dot(W3.T) * a2 * (1 - a2)
    delta_hidden_1 = delta_hidden_2.dot(W2.T) * a1 * (1 - a1)
    # 가중치와 bias 업데이트
    W4 += learning_rate * a3.T.dot(delta_output)
    b4 += learning_rate * np.sum(delta_output, axis=0, keepdims=True)
    W3 += learning_rate * a2.T.dot(delta_hidden_3)
    b3 += learning_rate * np.sum(delta_hidden_3, axis=0, keepdims=True)
    W2 += learning_rate * a1.T.dot(delta_hidden_2)
    b2 += learning_rate * np.sum(delta_hidden_2, axis=0, keepdims=True)
    W1 += learning_rate * X.T.dot(delta_hidden_1)
    b1 += learning_rate * np.sum(delta_hidden_1, axis=0, keepdims=True)
    if i % 10000 == 0:
        print(i,"회 반복 : ",error)
        error_list.append(abs(error))

# error_list = np.ravel(error_list)
# plt.plot(error_list)
# plt.title('Training Loss')
# plt.xlabel('Iterations')
# plt.ylabel('Error')
# plt.show()

def interpret_output(output):
    output = np.ndarray.flatten(output)
    assert(len(output) == 3)
    active = output >= 0.7
    #0.7 이상이 2개이상일 경우 판단 불가
    if np.count_nonzero(active) >= 2:
        return "판단 불가"
    if active[0]:
        return "뜀"
    elif active[1]:
        return "걸음"
    else:
        return "가만히 있기"


#테스트 데이터 로드
test_datum, file_names = load_datasets_in_folder("./test_data")

for test_data, file_name in zip(test_datum, file_names):
    print("---------------------------")
    print(f'테스트 데이터셋 이름: {file_name}')
    test_data = scaler.transform(np.reshape(test_data, (1, 10)))
    test_data = np.array([test_data])
    print(f'데이터: {test_data}')
    z1 = np.dot(test_data, W1) + b1
    a1 = 1 / (1 + np.exp(-z1))
    z2 = np.dot(a1, W2) + b2
    a2 = 1 / (1 + np.exp(-z2))
    z3 = np.dot(a2, W3) + b3
    a3 = 1 / (1 + np.exp(-z3))
    z4 = np.dot(a3, W4) + b4

    y_hat_test = 1 / (1 + np.exp(-z4))
    print(f"결과: {y_hat_test}")
    print(f"결과 해석: {interpret_output(y_hat_test)}")
