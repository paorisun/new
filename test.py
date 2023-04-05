from math import sqrt
from os import listdir, path
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

np.set_printoptions(precision=20, suppress=True)

#StandardScaler 로드
scaler = joblib.load(path.join(model_folder, "scaler.joblib"))

# 가중치 랜덤 1, 2, 3
container = np.load(path.join(model_folder, 'model.npz'), 'rb')
W1, W2, W3, W4, b1, b2, b3, b4 = [container[key] for key in container]

#테스트 데이터 로드
test_datum, file_names = load_datasets_in_folder("./test_data")

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
    print("결과:", y_hat_test)
    print(f"결과 해석: {interpret_output(y_hat_test)}")
