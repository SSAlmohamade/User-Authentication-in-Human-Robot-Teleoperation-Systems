import statistics
from typing import List, Any, Union

import tsfel
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from math import sqrt
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from tabulate import tabulate
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
#
# from sklearn.linear_model import LinearRegression
# from sklearn.cross_validation import cross_val_score
#
# scores = cross_val_score(LinearRegression(), X, y,scoring='r2')
import os
from sklearn.metrics import mean_squared_error
from scipy.stats import skew,kurtosis
from scipy import spatial
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from numpy import dot
from numpy.linalg import norm
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cdist
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.svm import OneClassSVM
from scipy.spatial import distance

from  Featuer_TSFEL import Features as FT
from  Featuer_TSFEL_seg import Features as FT_seg
pd.set_option('display.max_colwidth', None)

def Distance (x, y, t):
    dx = x[1:] - x[:-1]
    dy = y[1:] - y[:-1]

    step_size = np.sqrt(dx ** 2 + dy ** 2)
    # print('step_size:',step_size)


    cumulativeDistance = np.concatenate(([0], np.cumsum(step_size)))
    # print('cumulative Distance:',cumulativeDistance)

    dist = distance.euclidean(x, y)
    # print('euclidean_distance:', dist)

    # plt.plot(t, cumulativeDistance)
    # plt.show()
    return step_size

def Position_Velocity_similarity(self):

    # print('Attacker',Attacker)
    fileID1 = 'Data/User_' + Targeted + '_Ex2_Task1.csv'
    fileID2 = 'Training_against' + Targeted + '_Task' + str(num) + '.csv'
    dataset = pd.read_csv(fileID1)
    dataset2 = pd.read_csv(fileID2)

    #  targeted
    Pos1 = dataset[dataset["0"] == 1]
    Pos2 = dataset[dataset["0"] == 2]
    Pos3 = dataset[dataset["0"] == 3]
    # print('P2', Pos2)


    Part1_1= Pos1.iloc[:, 1].values
    Part1_2 = Pos2.iloc[:,1].values
    Part1_3 = Pos3.iloc[:,1].values
    # position
    X_T_1 = Pos1.iloc[:, 2].values
    X_T_2 = Pos2.iloc[:, 2].values
    X_T_3 = Pos3.iloc[:, 2].values

    Y_T_1 = Pos1.iloc[:, 3].values
    Y_T_2 = Pos2.iloc[:, 3].values
    Y_T_3 = Pos3.iloc[:, 3].values

    # Time targeted
    PT_1 = Part1_1[len(Part1_1) - 1] - Part1_1[0]
    PT_2 = Part1_2[len(Part1_2) - 1] - Part1_2[0]
    PT_3 = Part1_3[len(Part1_3) - 1] - Part1_3[0]
    # speed
    vm1_1 = np.mean(Pos1.iloc[:, 8].values)
    vm1_2 = np.mean(Pos2.iloc[:, 8].values)
    vm1_3 = np.mean(Pos3.iloc[:, 8].values)

    distance_T1= Distance(X_T_1, Y_T_1, Part1_1)
    # print('distance_T1 ', np.sum(distance_T1))
    distance_T2 = Distance(X_T_2, Y_T_2, Part1_2)
    # print('distance_T2 ', np.sum(distance_T2))
    distance_T3 = Distance(X_T_3, Y_T_3, Part1_3)
    # print('distance_T3 ', np.sum(distance_T3))

    #  Attacer
    Pos1_A = dataset2[dataset2["0"] == 1]
    Pos2_A = dataset2[dataset2["0"] == 2]
    Pos3_A = dataset2[dataset2["0"] == 3]
    # print('P2', Pos3_A)

    Time1_1A = Pos1_A.iloc[:, 1].values
    Time2_1A = Pos2_A.iloc[:, 1].values
    Time3_1A = Pos3_A.iloc[:, 1].values
    # position
    X_A_1 = Pos1_A.iloc[:, 2].values
    X_A_2 = Pos2_A.iloc[:, 2].values
    X_A_3 = Pos3_A.iloc[:, 2].values

    Y_A_1 = Pos1_A.iloc[:, 3].values
    Y_A_2 = Pos2_A.iloc[:, 3].values
    Y_A_3 = Pos3_A.iloc[:, 3].values

    # Time targeted
    PT_1A = Time1_1A[len(Time1_1A) - 1] - Time1_1A[0]
    PT_2A = Time2_1A[len(Time2_1A) - 1] - Time2_1A[0]
    PT_3A = Time3_1A[len(Time3_1A) - 1] - Time3_1A[0]
    # speed
    vm2_1A = np.mean(Pos1_A.iloc[:, 8].values)
    vm2_2A = np.mean(Pos2_A.iloc[:, 8].values)
    vm2_3A = np.mean(Pos3_A.iloc[:, 8].values)

    distance_A1 = Distance(X_A_1, Y_A_1, Time1_1A)
    # print('distance_A1 ', np.sum(distance_A1))
    distance_A2 = Distance(X_A_2, Y_A_2, Time2_1A)
    # print('distance_A2 ', np.sum(distance_A2))
    distance_A3 = Distance(X_A_3, Y_A_3, Time3_1A)
    # print('distance_A3 ', np.sum(distance_A3))

    # ===========================================
    DT1=round(np.sum(distance_T1),3)
    DT2 = round(np.sum(distance_T2), 3)
    DT3 = round(np.sum(distance_T3), 3)

    DA1=round(np.sum(distance_A1),3)
    DA2 = round(np.sum(distance_A2), 3)
    DA3 = round(np.sum(distance_A3), 3)

    print('cumulative Distance:')
    data = [[1,round(np.sum(distance_T1),3),  round(np.sum(distance_A1),3)],
            [2,round(np.sum(distance_T2),3), round(np.sum(distance_A2),3)],
            [3,round(np.sum(distance_T3),3), round(np.sum(distance_A3),3)]]
    print (tabulate(data, headers=["Part", "Targeted distance", "Attacker distance"]))
    # ===========================================



    # =============whole task
    P1 = dataset.iloc[:, 1].values
    P2 = dataset2.iloc[:, 1].values
    # print(P1)
    P1 = P1[len(P1) - 1] - P1[0]
    P2 = P2[len(P2) - 1] - P2[0]

    vm1 =np.mean( dataset.iloc[:, 8].values)
    vm2 = np.mean(dataset2.iloc[:, 8].values)
    similarity_V =  1- spatial.distance.cosine(vm1, vm2)
    # print('similarityV', similarity_V)

    similarity =cosine_similarity(dataset.values, dataset2.values)
    SV= np.mean(similarity[:,8])
    # ==================================================

    return PT_1,PT_2,PT_3,vm1_1,vm1_2,vm1_3,PT_1A,\
           PT_2A, PT_3A, vm2_1A, vm2_2A,vm2_3A,DT1,\
           DT2,DT3,DA1,DA2,DA3

def fill_missing_values(df):
    # Handle eventual missing data. Strategy: replace with mean.
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(df.mean(), inplace=True)
    return df

def Compute_Acceleration_features(feat):
    # The Acceleration features
    temp =[]
    Mean_Accel = np.mean(feat)
    temp.append(Mean_Accel)
    # print('Mean_Accel=',Mean_Accel)

    Minimum_Accel = min(feat)
    temp.append(Minimum_Accel)
    # print('min=',Minimum_Accel)

    Maximum_Accel = max(feat)
    temp.append(Maximum_Accel)
    # print('max=',Maximum_Accel)

    Median_Accel = np.median(feat)
    temp.append(Median_Accel)
    # print('Median=',Median_Accel)

    PeakTOpeak_Amplitude = Maximum_Accel - Minimum_Accel
    temp.append(PeakTOpeak_Amplitude)
    # print('PeakTOpeak_Amplitude=',PeakTOpeak_Amplitude)
    SKEW = skew(feat)
    temp.append(SKEW)

    KURTOSIS = kurtosis(feat)
    temp.append(KURTOSIS)

    Standard_Deviation = np.std(feat)
    temp.append(Standard_Deviation)
    # print('Standard_Deviation=',Standard_Deviation)

    # Variance = statistics.variance(feat)
    # temp.append(Variance)
    # print('Variance=',Variance)

    total_energy= tsfel.feature_extraction.features.total_energy(feat, 20)
    temp.append(total_energy)

    entropy= tsfel.feature_extraction.features.entropy(feat, prob='standard')
    temp.append(entropy)


    return temp

def Targeted_plot(Tar):

    global Targeted
    Targeted = Tar
    fileID1 = 'Data/User_'+str(Targeted)+'_Ex2_Task1.csv'
    dataset = pd.read_csv(fileID1)
    x1 = dataset.iloc[:, 2].values
    y1 = dataset.iloc[:, 3].values
    plt.plot(x1, y1,color="blue", label="Targeted User")
    xstart=[0.02]
    ystaret=[-0.8322]
    plt.plot(xstart, ystaret,marker='o',markersize=35,color='red', label="Start point")
    xend = [0.04390]
    yend = [0.29270]
    plt.plot(xend, yend, marker='o',markersize=32,color='yellow', label="End point")

    xtest1 = [0.69391]
    ytest1 = [-0.5000]
    plt.plot(xtest1, ytest1, marker='o',color='green', markersize=20, label="Test point")

    xtest2 = [0.49391]
    ytest2 = [0.02499]
    plt.plot(xtest2, ytest2, marker='o',color='green', markersize=20)



    plt.xlabel('x - axis')
    plt.xticks(rotation=90)
    # naming the y axis
    plt.ylabel('y - axis')
    plt.yticks(rotation=90)
    # giving a title to my graph
    # plt.title('Two lines on same graph!')
    plt.rcParams["figure.figsize"] = (4, 3)
    # plt.legend()
    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05))
    #       fancybox=True, shadow=True, ncol=3)
    PngFile= 'Targeted_'+str(Targeted)+'.png'
    # if os.path.isfile(PngFile):
    # os.remove(PngFile)
    plt.savefig(PngFile,orientation='portrait')
    plt.close()


def plot_similarity(self,Att,Tar,no):
    global Attacker
    global Targeted
    global num
    Attacker = Att
    Targeted = Tar
    num=no
    fileID1 = 'Data/User_'+Targeted+'_Ex2_Task1.csv'
    # fileID2 = 'Attacker_'+str(Attacker)+'_Ex2_Task1.csv'
    fileID2 = 'Training_against' + Targeted + '_Task' + str(num) + '.csv'
    dataset = pd.read_csv(fileID1)
    dataset2 = pd.read_csv(fileID2)

    # MAE = mean_absolute_error(dataset, dataset2)
    # print('\n mean_absolute_error=', MAE)
    # MSE = mean_squared_error(dataset, dataset2)
    # print('\n mean_squared_error=', MSE)
    # RMSE = sqrt(MSE)
    # print('\n root_mean_squared_error=', RMSE)
    # print('\n')

    # result = profile_similarity(self)
    probabilityAll,result,prpba,MAE,RMSE= Decision_seg()

    x1 = dataset.iloc[:, 2].values
    y1 = dataset.iloc[:, 3].values
    x2 = dataset2.iloc[:, 2].values
    y2 = dataset2.iloc[:, 3].values




    print('result', result,Targeted)
    if result == 'Accepted':
        linecolor = "green"
    else:
        linecolor = "red"

    plt.plot(x1, y1,color="blue", label="Targeted User")
    plt.plot(x2, y2,color=linecolor, label="You")



    xstart=[0.02]
    ystaret=[-0.8322]
    plt.plot(xstart, ystaret,marker='o',markersize=35,color='red', label="Start point")

    xend = [0.04390]
    yend = [0.29270]
    plt.plot(xend, yend, marker='o',markersize=32,color='yellow', label="End point")

    xtest1 = [0.69391]
    ytest1 = [-0.5000]
    plt.plot(xtest1, ytest1, marker='o',color='green', markersize=20, label="Test point")

    xtest2 = [0.49391]
    ytest2 = [0.02499]
    plt.plot(xtest2, ytest2, marker='o',color='green', markersize=20)


    # TPosition[-0.006089999806135893, 0.32499781250953674]
    # TPosition2[0.6939100027084351, -0.5000022649765015]
    # naming the x axis
    plt.xlabel('x - axis')
    plt.xticks(rotation=90)
    # naming the y axis
    plt.ylabel('y - axis')
    plt.yticks(rotation=90)
    # giving a title to my graph
    # plt.title('Two lines on same graph!')
    plt.rcParams["figure.figsize"] = (4, 3)
    # plt.legend()
    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05))
    #       fancybox=True, shadow=True, ncol=3)
    PngFile= 'Attacker_'+str(Attacker)+'.png'
    # if os.path.isfile(PngFile):
    # os.remove(PngFile)
    plt.savefig(PngFile,orientation='portrait')
    plt.close()
    print('result',result)

    # V1 = dataset.iloc[:, 7].values
    # V2 = dataset2.iloc[:, 7].values
    # t1 = dataset.iloc[:, 1].values
    # t2 = dataset2.iloc[:, 1].values
    # plt.plot(t1, V1, color="blue", label="Targeted User")
    # plt.plot(t2, V2, color=linecolor, label="You")
    # plt.show()

    return probabilityAll, result , prpba,MAE,RMSE


def Decision_seg():
    global Attacker
    global Targeted
    All = pd.read_csv("featuer_file_seg28User.csv")
    All = fill_missing_values(All)
    # print(All.head(1))
    # All.drop(All.columns[1], axis=1, inplace=True)
    # print(All.head(1))
    fileID2 = 'Training_against' + Targeted + '_Task' + str(num) + '.csv'
    print('file:',fileID2)

    tagetuser_Sample1 = All[All["User"] == int(Targeted)]
    tagetuser_Sample = tagetuser_Sample1[tagetuser_Sample1["Task"] == 1]
    tagetuser_SampleX = tagetuser_Sample.iloc[:, 1:].values
    tagetuser_Sampley = tagetuser_Sample.iloc[:, 0].values
    tagetuser_Sampley[:] = 1
    print('tagetuser_Sample', tagetuser_Sample.shape)




    TestSamples1 = All[All["User"] != int(Targeted)]
    # TestSamples = TestSamples1[TestSamples1["Task"] == 1]
    TestSamples = TestSamples1[TestSamples1["Task"] == 1].head(tagetuser_SampleX.shape[0])
    TestSamplesX = TestSamples.iloc[:, 1:].values
    TestLeabelsy = TestSamples.iloc[:, 0].values
    TestLeabelsy[:] = 0
    # X2 = All.iloc[:, 1:].values
    # y2 = All.iloc[:, 0].values

    # print('tagetuser_SampleX', tagetuser_SampleX.shape)
    # print('TestSamplesX', TestSamplesX.shape)
    # tagetuser_SampleX = Scaler.fit_transform(tagetuser_SampleX)
    # TestSamplesX = Scaler.fit_transform(TestSamplesX)
    X2 = np.concatenate((TestSamplesX, tagetuser_SampleX))
    y2 = np.concatenate((TestLeabelsy, tagetuser_Sampley))
    # print('X2', X2.shape)


    Feature_1 = FT_seg()
    U = Attacker
    task=1
    Data = pd.read_csv((fileID2))
    Data1 = Data[Data["0"] == 1]
    Data2 = Data[Data["0"] == 2]
    Data3 = Data[Data["0"] == 3]
    list_df = [Data1, Data2, Data3]

    Featuerfile = []
    seg = 0
    for D in list_df:
        profile = []
        seg = seg + 1
        time_stamp, Velocity_x, Velocity_y, Velocity_m, Accel_x, Accel_y, Accel_m = Feature_1.ImportData(D)
        # Feature_1.Create_15_profile(U,i, time_stamp, Velocity_x, Velocity_y, Velocity_m, Accel_x, Accel_y, Accel_m)

        Vx = Compute_Acceleration_features(Velocity_x)
        Vy = Compute_Acceleration_features(Velocity_y)
        Vm = Compute_Acceleration_features(Velocity_m)

        Ax = Compute_Acceleration_features(Accel_x)
        Ay = Compute_Acceleration_features(Accel_y)
        Am = Compute_Acceleration_features(Accel_m)
        T = time_stamp[len(time_stamp) - 1] - time_stamp[0]

        profile.append(1)
        profile.append(seg)

        # profile.append(1)

        profile.append(T)

        for f in [Vx, Vy, Vm, Ax, Ay, Am]:
            for fet in range(len(f)):
                profile.append(f[fet])

        Featuerfile.append(profile)
    # Allseg=pd.DataFrame(Featuerfile)
    # for i in Featuerfile:
    #    # print(i)
    count = 0
    print('Targeted user', [int(Targeted)])
    print('-----------------------------')

    RandomForest = []
    OneClassSVMAll = []
    IsolationForestAll = []
    probabilityAll = []
    Featuerfile2=pd.DataFrame(Featuerfile)
    # print('Featuerfile2',Featuerfile2)
    test = Featuerfile2.iloc[:, 1:].values



    # print('Attaker_SampleX', test.shape)
    # print('test',test)
    MAEAll=[]
    RMSEAll=[]
    similarityAll=[]

    # Feature Scaling
    # Scaler = RobustScaler(with_centering=True,with_scaling=True,copy=True,)
    # print('test', test[0])
    # Scaler = StandardScaler()
    # X2 = Scaler.fit_transform(X2)
    # test = Scaler.fit_transform(test)
    # tagetuser_SampleX = Scaler.fit_transform(tagetuser_SampleX)

    # print('test',test[0])
    for i in range (len(test)):
        MAE = mean_absolute_error(tagetuser_SampleX[i],test[i])
        MAE2 = mean_absolute_percentage_error(tagetuser_SampleX[i], test[i])
        MAEAll.append(MAE)
        print(' mean_absolute_error=', MAE)
        MSE=mean_squared_error(tagetuser_SampleX[i],test[i])

        # print('\n mean_squared_error=', MSE)
        RMSE = sqrt(MSE)
        RMSEAll.append(RMSEAll)
        print('root_mean_squared_error=', RMSE)
        euclideandist = distance.euclidean(tagetuser_SampleX[i],test[i])
        print(' euclidean dist=', euclideandist)
        euclideandist = distance.cityblock(tagetuser_SampleX[i], test[i])
        print(' cityblock dist=', euclideandist)
        result = 1 - spatial.distance.cosine(tagetuser_SampleX[i], test[i])
        similarityAll.append(result)
        print(' similarity=', result)


        print('\n')

    for A in Featuerfile:
        count += 1

        X_test = np.array(A[1:]).reshape(1, -1)
        # print('X_test',X_test)
        # y_test = [1]

        RF = RandomForestClassifier(n_estimators= 200, min_samples_split= 2,\
                                      min_samples_leaf=1, \
                                      max_depth= 10, bootstrap= False,\
                                      criterion='entropy',max_features='sqrt',\
                                      random_state=0).fit(X2, y2)
        pred = RF.predict(X_test)
        probability = RF.predict_proba(X_test)
        # print('mean_absolute_error',mean_absolute_error(y_test, pred))
        # print(mean_absolute_error(y_test, et_pred))
        # probabilityAll.append(probability[0][int(Targeted) - 1])
        probabilityAll.append(probability[0][1])
        # print("Attacker part", count)
        # print('predict:', pred[0])
        RandomForest.append(pred[0])

        print('probability', probability)

        clf = OneClassSVM(nu=0.2,max_iter=100).fit(tagetuser_SampleX)
        print('decision_function',clf.decision_function(X_test))
        OneClassSVMAll.append(clf.predict(X_test)[0])

        RF = IsolationForest(n_estimators=200, max_features=X_test.shape[1], \
                             contamination=0.3).fit(tagetuser_SampleX)
        decision_function_score_anomalies = abs(RF.decision_function(X_test))
        sklearn_score_anomalies = abs(RF.score_samples(X_test))
        # print('IsolationForest',RF.predict(X_test))
        IsolationForestAll.append(RF.predict(X_test)[0])


    print('RandomForest', RandomForest)
    print('OneClassSVM', OneClassSVMAll)
    # print('IsolationForest', IsolationForestAll)
    print('probabilityAll', probabilityAll,np.sum(probabilityAll)/len(probabilityAll))
    D= np.sum(probabilityAll) / len(probabilityAll)

    print('D',D)
    if D>=0.50:
        result = 'Accepted'
    else:
        result = 'Rejected'

    print('D', D,result )


    return probabilityAll,result , D, MAEAll,RMSEAll

def Decision_seg_time():
    global Attacker
    global Targeted
    All=pd.read_csv("featuer_file_time_seg20User.csv")
    All = fill_missing_values(All)
    fileID2 = 'Training_against' + Targeted + '_Task' + str(num) + '.csv'

    tagetuser_Sample1 = All[All["User"] == int(Targeted)]
    tagetuser_Sample = tagetuser_Sample1[tagetuser_Sample1["Task"] == 1]
    tagetuser_SampleX = tagetuser_Sample.iloc[:, 1:].values
    tagetuser_Sampley = tagetuser_Sample.iloc[:, 0].values
    tagetuser_Sampley[:] = 1


    TestSamples1= All[All["User"] != int(Targeted)]
    # TestSamples = TestSamples1[TestSamples1["Task"] == 1]
    TestSamples = TestSamples1[TestSamples1["Task"] == 1].head(tagetuser_Sampley.shape[0])
    TestSamplesX = TestSamples.iloc[:, 1:].values
    TestLeabelsy = TestSamples.iloc[:, 0].values
    TestLeabelsy[:] = 0
    # X2 = All.iloc[:, 1:].values
    # y2 = All.iloc[:, 0].values


    X2 = np.concatenate((TestSamplesX, tagetuser_SampleX))
    y2 = np.concatenate((TestLeabelsy, tagetuser_Sampley))
    print('tagetuser_SampleX', tagetuser_SampleX.shape)
    print('TestSamplesX', TestSamplesX.shape)
    print('X2', X2.shape)

    Feature_1 = FT_seg()
    U = Attacker
    task=1
    Data = pd.read_csv((fileID2))
    # Time segment
    # ==================================================
    n=50
    list_df = [Data[i:i+n] for i in range(0,Data.shape[0],n)]
    Featuerfile = []
    seg = 0
    for D in list_df:
        profile = []
        seg =seg+ 1
        time_stamp, Velocity_x, Velocity_y, Velocity_m, Accel_x, Accel_y, Accel_m = Feature_1.ImportData(D)
        # Feature_1.Create_15_profile(U,i, time_stamp, Velocity_x, Velocity_y, Velocity_m, Accel_x, Accel_y, Accel_m)

        Vx = Compute_Acceleration_features(Velocity_x)
        Vy = Compute_Acceleration_features(Velocity_y)
        Vm = Compute_Acceleration_features(Velocity_m)

        Ax = Compute_Acceleration_features(Accel_x)
        Ay = Compute_Acceleration_features(Accel_y)
        Am = Compute_Acceleration_features(Accel_m)
        T = time_stamp[len(time_stamp) - 1] - time_stamp[0]
        profile.append(1)
        profile.append(seg)
        # profile.append(1)
        profile.append(T)
        for f in [Vx, Vy, Vm, Ax, Ay, Am]:
            for fet in range(len(f)):
                profile.append(f[fet])

        Featuerfile.append(profile)
        # Allseg=pd.DataFrame(Featuerfile)

    # print('Featuer file',Featuerfile)

    # selector = VarianceThreshold(threshold=0.0)  # Remove low variance features
    # X2 = selector.fit_transform(X2)
    # tagetuser_SampleX= selector.fit_transform(tagetuser_SampleX)
    # Featuerfile = selector.fit_transform(Featuerfile)
    count = 0
    print('Targeted user', [int(Targeted)])
    print('-----------------------------')
    RandomForest = []
    OneClassSVMAll=[]
    IsolationForestAll=[]
    probabilityAll=[]

    for A in Featuerfile:
        count += 1
        X_test = np.array(A[1:]).reshape(1, -1)
        y_test = [int(Targeted)]

        RF = RandomForestClassifier(n_estimators= 200, min_samples_split= 2,\
                                      min_samples_leaf=1, \
                                      max_depth= 10, bootstrap= False,\
                                      criterion='entropy',max_features='sqrt',random_state=0).fit(X2, y2)
        pred = RF.predict(X_test)
        probability = RF.predict_proba(X_test)
        print(RF.classes_)
        print(probability)
        # probabilityAll.append(probability[0][int(Targeted) - 1])
        probabilityAll.append(probability[0][1])
        # print("Attacker part", count)
        # print('predict:', pred[0])
        RandomForest.append(pred[0])

        clf = OneClassSVM(nu=0.45,max_iter=100,).fit(tagetuser_SampleX)
        print('OneClassSVM',clf.predict(X_test)* (-1))
        OneClassSVMAll.append((clf.predict(X_test)[0]))

        RF = IsolationForest(n_estimators =200,max_features=X_test.shape[1],\
                             contamination=0.5).fit(tagetuser_SampleX)
        IsolationForestAll.append(RF.predict(X_test)[0])


        # print('\n')
    print('RandomForest',RandomForest)
    print('OneClassSVM',OneClassSVMAll,)
    # print('IsolationForest',IsolationForestAll)
    print('probabilityAll',probabilityAll,np.sum(probabilityAll)/len(probabilityAll))

    propa= np.sum(probabilityAll)/len(probabilityAll)
    if propa>=0.50:
        result = 'Accepted'
    else:
        result = 'Rejected'
    return result, propa


def main ():
    print()
    global Attacker
    global Targeted

    Attacker=1
    Targeted=2
    # plot_similarity(1,Attacker,Targeted,1)
    # Targeted_plot(Targeted)
    # # Decision()
    # Decision_seg()
    # Decision_seg_time()

if __name__ == '__main__':
    main()
