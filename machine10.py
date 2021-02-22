from tensorflow.keras import Model # Model 임포트
from tensorflow import keras # keras 임포트
from tensorflow.keras.models import Sequential #Sequential 임포트
from tensorflow.keras.layers import Dense, Dropout, Flatten #Dense, Dropout, Flatten 임포트
from tensorflow.keras.layers import Conv2D, MaxPooling2D #Conv2D, Maxpooling2D 임포트
from tensorflow.keras.datasets import mnist # mnist 임포트
import matplotlib.pyplot as plt #matplotlib.pyplot 을 임포트하고 plt로 쓰겠다
import numpy as np #numpy를 임포트하고 np로 쓰겠다

def load_data(): # 데이터를 부르는 함수
    # 먼저 MNIST 데이터셋을 로드하겠습니다. 케라스는 `keras.datasets`에 널리 사용하는 데이터셋을 로드하기 위한 함수를 제공합니다. 이 데이터셋은 이미 훈련 세트와 테스트 세트로 나누어져 있습니다. 훈련 세트를 더 나누어 검증 세트를 만드는 것이 좋습니다:

    (X_train_full, y_train_full), (X_test, y_test) = mnist.load_data() # mnist데이터를 X에는 훈련데이터와 테스트데이터를 Y에는 그에 맞는 라벨값을 저장
    X_train_full = X_train_full.astype(np.float32) # float형태로 변환
    X_test = X_test.astype(np.float32) # float형태로 변환
    return X_train_full, y_train_full, X_test, y_test

def data_normalization(X_train_full, X_test):
    # 전체 훈련 세트를 검증 세트와 (조금 더 작은) 훈련 세트로 나누어 보죠. 또한 픽셀 강도를 255로 나누어 0~1 범위의 실수로 바꾸겠습니다.

    X_train_full = X_train_full / 255. #X의 훈련값을 0~1의 수로 하기위해 255를 나누어줌

    X_test = X_test / 255. #X의 테스트값을 0~1의 수로 하기위해 255를 나누어줌
    train_feature = np.expand_dims(X_train_full, axis=3) #tensorflow를 사용하기위해 차원수를 하나 늘린다
    test_feature = np.expand_dims(X_test, axis=3) #tensorflow를 사용하기위해 차원수를 하나 늘린다

    print(train_feature.shape, train_feature.shape)
    print(test_feature.shape, test_feature.shape)

    return train_feature,  test_feature


def draw_digit(num): #안쓰는 함수같은데 왜있는지 모르겠습니다.
    for i in num:
        for j in i:
            if j == 0:
                print('0', end='')
            else :
                print('1', end='')
        print()





def makemodel(X_train, y_train, X_valid, y_valid, weight_init):
    model = Sequential() #Sequential 모델 생성
    model.add(Conv2D(32, kernel_size=(3, 3),  activation='relu')) # 3 by 3 크기의 커널을 32개 만들고 activation은 relu
    model.add(MaxPooling2D(pool_size=2)) #이미지 크기를 줄이기 위해 맥스풀링
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu')) # 3 by 3 크기의 커널을 64개 만들고 activation은 relu
    model.add(MaxPooling2D(pool_size=2)) #이미지 크기를 줄이기 위해 맥스풀링
    model.add(Dropout(0.25)) # 0.25% 드롭아웃
    model.add(Flatten()) #이미지의 사이즈를 펴준다?
    model.add(Dense(128, activation='relu')) # 입력값과 가중치를 계산하여 128개의 출력하는 층  activation은 relu
    model.add(Dense(10, activation='softmax')) # 0~9까지의 10개가 필요하므로 10개를 출력하는 층 activation은 softmax

    model.compile(loss='sparse_categorical_crossentropy',#모델 컴파일, loss함수는 'sparse_categorical_crossentropy'사용
                  optimizer='adam', #optimizer은 'adam'
                  metrics=['accuracy'])  #정확도를 나타냄



    return model

def plot_history(histories, key='accuracy'):
    plt.figure(figsize=(16,10)) #그림을 그리기 위한 판 생성 사이즈는 16 by 10

    for name, history in histories: #name은 'baseline' history는 피팅시킨 모델을 가져온다
        val = plt.plot(history.epoch, history.history['val_'+key],
                       '--', label=name.title()+' Val') #x축은 모델의 epoch값 y축은 val(test)의 정확도값으로 하고 '--'형태로 그림을 그리고 baseline Val이라고 라벨링함
        plt.plot(history.epoch, history.history[key], color=val[0].get_color(),
                 label=name.title()+' Train')  #x축은 모델의 epoch값 y축은 train의 정확도값으로 하고  baseline Train이라고 라벨링함

    plt.xlabel('Epochs') #x축이름 Epochs
    plt.ylabel(key.replace('_',' ').title()) #y축 이름 Accuracy
    plt.legend()

    plt.xlim([0,max(history.epoch)]) #x값을 0~history.epoch의 최대값까지 설정
    plt.show()
import random


def draw_prediction(pred, k,X_test,y_test,yhat):
    samples = random.choices(population=pred, k=16) #아래 함수에서 만든 pred라는 리스트에서 16개를 랜덤하게 추출하여 sample에 저장

    count = 0
    nrows = ncols = 4
    plt.figure(figsize=(12,8)) #그림을 그리기 위해 12,8 사이즈의 판 생성

    for n in samples:
        count += 1 #count가 1이고 for문을 돌때마다 1씩 증가
        plt.subplot(nrows, ncols, count) #4 by 4서브플롯을 만들고 count에 따라 몇번째인지 결정
        plt.imshow(X_test[n].reshape(28, 28), cmap='Greys', interpolation='nearest') # 위에서 얻은 16개의 데이터를 28 by 28로 바꾸어 나타냄
        tmp = "Label:" + str(y_test[n]) + ", Prediction:" + str(yhat[n]) #Label에는 y_test이 보이게 라벨값이 Prediction에서는 X_test를 넣었을때 예측한 값을 보이게함
        plt.title(tmp) #각 그림의 타이틀을 바로위에서구한 tmp가 나타나도록 함

    plt.tight_layout()
    plt.show()

def evalmodel(X_test,y_test,model):
    yhat = model.predict(X_test) # X_test값을 넣은 y의 예측값을 yhat에 저장
    yhat = yhat.argmax(axis=1) #예측값중 가장 큰값을 저장

    print(yhat.shape)
    answer_list = [] #빈 리스트 생성  위에 draw_prediction에서 쓰임

    for n in range(0, len(y_test)):
        if yhat[n] == y_test[n]:
            answer_list.append(n) #만약 예측 라벨값과 실제 라벨값이 같다면 리스트에 저장

    draw_prediction(answer_list, 16,X_test,y_test,yhat) #위에서 만든 함수 적용

    answer_list = [] #이번에는 예측 라벨값과 실제 라벨값이 다를때를 보여주기 위해 새로운 리스트 생성

    for n in range(0, len(y_test)):
        if yhat[n] != y_test[n]:# 예측 라벨값과 실제 라벨값이 다르다면 리스트에 저장
            answer_list.append(n)

    draw_prediction(answer_list, 16,X_test,y_test,yhat) #위에서 만든 함수 적용

def main():
    X_train, y_train, X_test, y_test = load_data() #위에서 load_data()에서 리턴한 값들을 저장
    X_train, X_test = data_normalization(X_train,  X_test) #data_normalization()에서 리턴한 값들을 저장 X값들만 0~1사이의 값으로 만들어줌

    #show_oneimg(X_train)
    #show_40images(X_train, y_train)

    model= makemodel(X_train, y_train, X_test, y_test,'glorot_uniform') #makemodel에서 만든 모델을 저장장


    baseline_history = model.fit(X_train,
                                 y_train,
                                 epochs=2,
                                 batch_size=512,
                                 validation_data=(X_test, y_test),
                                 verbose=2)# 모델 피팅 x에는 X_train y에는 y_train값을 넣고 epochs는 2번만 돌도록 batch size는 512로
                                            #  validation data는 각 테스트값과 라벨값을 넣고 verbose =2 로 설정


    evalmodel(X_test, y_test, model) #evalmodel함수 적용
    plot_history([('baseline', baseline_history)])#plot_history함수 적용
main()