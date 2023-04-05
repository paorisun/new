from math import sqrt
from os import listdir, path
import os
import shutil
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from numpy.random import Generator, PCG64

model_folder = "./model"

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

np.set_printoptions(precision=20, suppress=True)

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
W1 = rng.random((input_size, hidden_size_1))
W2 = rng.random((hidden_size_1, hidden_size_2))
W3 = rng.random((hidden_size_2, hidden_size_3))
W4 = rng.random((hidden_size_3, output_size))

b1 = np.zeros((1, hidden_size_1))
b2 = np.zeros((1, hidden_size_2))
b3 = np.zeros((1, hidden_size_3))
b4 = np.zeros((1, output_size))

np.set_printoptions(precision=20, suppress=True)

# 학습률과 반복수
learning_rate = 0.01
num_iterations = 5000000
error_list = []

error_10000 = []
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
    if i % 100000 == 0:
        print(i,"회 반복 : ",error)
        error_list.append(abs(error))

if path.exists(model_folder):
    print("이미 존재하는 네트워크를 삭제합니다.")
    shutil.rmtree(model_folder)
os.makedirs(model_folder, exist_ok=True)

print(f"학습된 네트워크를 다음 위치에 저장합니다: {model_folder}")

joblib.dump(scaler, path.join(model_folder, "scaler.joblib"))

np.savez(path.join(model_folder, 'model.npz'), *[W1, W2, W3, W4, b1, b2, b3, b4])