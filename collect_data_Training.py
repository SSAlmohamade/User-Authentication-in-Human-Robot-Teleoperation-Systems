# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 10:19:43 2018

@author: shurook
"""

'''importing required libraries '''
import vrep
import sys
import csv
import math
import time
from scipy import spatial
import matplotlib.pyplot as plt
import pandas as pd
import Exel_Function as EF


#import matplotlib.pyplot as plt


'''experment information'''
Experment =2 # Time=1 , Point=2
User_Name='Attacker_1'
Task_NO=1
# EXl_File='Feture_Profile.xlsx'




'''---------------------------------------------------------'''
'''--- Setting up connection with VREP---'''
'''--------------------------------------------------------'''


print( '================ Connetion to VRep ================')
vrep.simxFinish(-1) # just in case, close all opened connections

clientID=vrep.simxStart('127.0.0.1',19997,True,True,5000,5) # Connect to V-REP
if clientID!=-1:
        print ('Connected to remote API server')
else:
        print ('Connection unsuccessful ')
        sys.exit('Could not connect')
 
       
'''---------------------------------------------------------------------------------------------'''        
''' Read manipSphere position from Simulater '''
'''---------------------------------------------------------------------------------------------'''
def get_object_Position_Point_Segmant():
        print('--------------------------------------- ')
        Values_manipSphere_Position = []  #empty array for postion measurements
        Values_Shape_L_Velocity=[]       #empty array for L_Velocity measurements
        r=[]
        Time=[]
    
        Values_manipSphere_Position1 = []  #empty array for postion measurements
        Values_Shape_L_Velocity1=[]       #empty array for L_Velocity measurements
        r1=[]
        Time1=[]
        
        Values_manipSphere_Position2 = []  #empty array for postion measurements
        Values_Shape_L_Velocity2=[]       #empty array for L_Velocity measurements
        r2=[]
        Time2=[]
    
    
   
        start =time.time()
         #Target Position
        errorCodeTarget,target = vrep.simxGetObjectHandle(clientID,'Target1',vrep.simx_opmode_oneshot_wait)
        returnCode,target_P=vrep.simxGetObjectPosition(clientID,target,-1,vrep.simx_opmode_oneshot_wait)        
        TPosition= [target_P[0],target_P[1]]
        # print('TPosition', TPosition)
          
        errorCodeTarget,target0 = vrep.simxGetObjectHandle(clientID,'Target2',vrep.simx_opmode_oneshot_wait)
        returnCode,target0_P=vrep.simxGetObjectPosition(clientID,target0,-1,vrep.simx_opmode_oneshot_wait)        
        TPosition0= [target0_P[0],target0_P[1]]
        # print('TPosition', TPosition0)
          
        errorCodeTarget,target3 = vrep.simxGetObjectHandle(clientID,'Target3',vrep.simx_opmode_oneshot_wait)
        eturnCode,target3_P=vrep.simxGetObjectPosition(clientID,target3,-1,vrep.simx_opmode_oneshot_wait)        
        TPosition3= [target3_P[0],target3_P[1]]
        # print('TPosition3',TPosition3)
          
        errorCodeTarget,target4 = vrep.simxGetObjectHandle(clientID,'Target4',vrep.simx_opmode_oneshot_wait)
        returnCode,target4_P=vrep.simxGetObjectPosition(clientID,target4,-1,vrep.simx_opmode_oneshot_wait)        
        TPosition4= [target4_P[0],target4_P[1]]
        # print('TPosition4', TPosition4)
        end_maze = False
        while not (end_maze):
            errorCodeTarget, manipSphere = vrep.simxGetObjectHandle(clientID, 'Sphere', vrep.simx_opmode_oneshot_wait)
            returnCode, SPosition = vrep.simxGetObjectPosition(clientID, manipSphere, -1, vrep.simx_opmode_oneshot_wait)
            Sphere_P = [SPosition[0], SPosition[1]]
            if ((round(TPosition0[0], 1) == round(Sphere_P[0], 1)) and (round(Sphere_P[1], 1) == round(TPosition0[1], 1))):
                Testpoint = True
                # print('not started')
            else:
                print('___________start recording ________________')


                rawdata = []

                while not (end_maze):
                     temp = []
                     print("StartMaze")
                     print('-----PART1-----')

                     Testpoint=False
                     while not (Testpoint):
                         # print("Testpoint 1")
                         # get Sphere Position from v-rep
                         errorCodeTarget,manipSphere = vrep.simxGetObjectHandle(clientID,'Sphere',vrep.simx_opmode_oneshot_wait)
                         returnCode,SPosition=vrep.simxGetObjectPosition(clientID,manipSphere,-1,vrep.simx_opmode_oneshot_wait)
                         Sphere_P = [SPosition[0],SPosition[1]]


                         #get Velocity  from v-rep
                         errorCodeTarget,Sphere_Velocity = vrep.simxGetObjectHandle(clientID,'Sphere',vrep.simx_opmode_oneshot_wait)
                         returnCode, linearVelocity, angularVelocity =vrep.simxGetObjectVelocity( clientID,Sphere_Velocity,vrep.simx_opmode_streaming )
                         Shape_L_Velocity = [linearVelocity[0],linearVelocity[1]]
                         #Shape_An_Velocity = [angularVelocity[0],angularVelocity[1]]
                         T= time.time()-start
                         Time.append(T)
                         Values_Shape_L_Velocity.append(Shape_L_Velocity)
                         Values_manipSphere_Position.append(Sphere_P)
                         R=math.sqrt((SPosition[0])**2+(SPosition[1])**2)
                         R=round(R,4)
                         r.append(R)

                         Magnitude_V = math.sqrt((linearVelocity[0]) ** 2 + (linearVelocity[1]) ** 2)
                         Magnitude_P = math.sqrt((SPosition[0]) ** 2 + (SPosition[1]) ** 2)

                         temp.append(1)
                         temp.append(T)

                         temp.append(Sphere_P[0])
                         temp.append(Sphere_P[1])
                         temp.append(Magnitude_P)

                         temp.append(Shape_L_Velocity[0])
                         temp.append(Shape_L_Velocity[1])
                         temp.append(Magnitude_V)
                         temp.append(R)

                         # print('temp=', temp)
                         rawdata.append(temp)
                         temp = []

                         if( (round(TPosition3[0],1)==round(Sphere_P[0],1)) and (round(Sphere_P[1],1)==round(TPosition3[1],1))):
                            Testpoint=True
                            # print("testpoint")
                     temp = []
                     Testpoint2=False
                     print('-----PART2-----')
                     while not (Testpoint2):
                         # print("Testpoint 2")
                         #Sphere Position
                         errorCodeTarget,manipSphere = vrep.simxGetObjectHandle(clientID,'Sphere',vrep.simx_opmode_oneshot_wait)
                         returnCode,SPosition=vrep.simxGetObjectPosition(clientID,manipSphere,-1,vrep.simx_opmode_oneshot_wait)
                         Sphere_P = [SPosition[0],SPosition[1]]


                         #Velocity
                         errorCodeTarget,Sphere_Velocity = vrep.simxGetObjectHandle(clientID,'Sphere',vrep.simx_opmode_oneshot_wait)
                         returnCode, linearVelocity, angularVelocity =vrep.simxGetObjectVelocity( clientID,Sphere_Velocity,vrep.simx_opmode_streaming )
                         Shape_L_Velocity = [linearVelocity[0],linearVelocity[1]]
                         #Shape_An_Velocity = [angularVelocity[0],angularVelocity[1]]
                         T= time.time()-start
                         Time1.append(T)
                         Values_Shape_L_Velocity1.append(Shape_L_Velocity)
                         Values_manipSphere_Position1.append(Sphere_P)
                         R1=math.sqrt((SPosition[0])**2+(SPosition[1])**2)
                         R1=round(R1,4)
                         r1.append(R1)




                         Magnitude_V = math.sqrt((linearVelocity[0]) ** 2 + (linearVelocity[1]) ** 2)
                         Magnitude_P = math.sqrt((SPosition[0]) ** 2 + (SPosition[1]) ** 2)

                         temp.append(2)
                         temp.append(T)

                         temp.append(Sphere_P[0])
                         temp.append(Sphere_P[1])
                         temp.append(Magnitude_P)

                         temp.append(Shape_L_Velocity[0])
                         temp.append(Shape_L_Velocity[1])
                         temp.append(Magnitude_V)
                         temp.append(R)

                         # print('temp=', temp)
                         rawdata.append(temp)
                         temp = []
                         # condition to check if the Sphere position == test point postion
                         if( (round(TPosition4[0],1)==round(Sphere_P[0],1)) and (round(Sphere_P[1],1)==round(TPosition4[1],1))):
                            Testpoint2=True


                     print('-----PART3-----')
                     temp = []
                     while not (end_maze):
                        # print("Testpoint 3")
                        #Sphere Position
                        errorCodeTarget,manipSphere = vrep.simxGetObjectHandle(clientID,'Sphere',vrep.simx_opmode_oneshot_wait)
                        returnCode,SPosition=vrep.simxGetObjectPosition(clientID,manipSphere,-1,vrep.simx_opmode_oneshot_wait)
                        Sphere_P = [SPosition[0],SPosition[1]]

                        errorCodeTarget,Sphere_Velocity = vrep.simxGetObjectHandle(clientID,'Sphere',vrep.simx_opmode_oneshot_wait)
                        returnCode, linearVelocity, angularVelocity =vrep.simxGetObjectVelocity( clientID,Sphere_Velocity,vrep.simx_opmode_streaming )
                        Shape_L_Velocity = [linearVelocity[0],linearVelocity[1]]
                        #Shape_An_Velocity = [angularVelocity[0],angularVelocity[1]]
                        T= time.time()-start
                        Time2.append(T)
                        Values_Shape_L_Velocity2.append(Shape_L_Velocity)
                        Values_manipSphere_Position2.append(Sphere_P)
                        R2=math.sqrt((SPosition[0])**2+(SPosition[1])**2)
                        R2=round(R2,4)
                        r2.append(R2)

                        Magnitude_V = math.sqrt((linearVelocity[0]) ** 2 + (linearVelocity[1]) ** 2)
                        Magnitude_P = math.sqrt((SPosition[0]) ** 2 + (SPosition[1]) ** 2)

                        temp.append(3)
                        temp.append(T)

                        temp.append(Sphere_P[0])
                        temp.append(Sphere_P[1])
                        temp.append(Magnitude_P)

                        temp.append(Shape_L_Velocity[0])
                        temp.append(Shape_L_Velocity[1])
                        temp.append(Magnitude_V)
                        temp.append(R2)

                        # print('temp=', temp)
                        rawdata.append(temp)
                        temp = []



                        # condition to check if the Sphere position == end point postion
                        if( (round(TPosition[0],1)==round(Sphere_P[0],1)) and (round(Sphere_P[1],1)==round(TPosition[1],1))  ):
                            end_maze=True
                            print("End_maze")

                end = time.time()
                tasktime=end - start
                print('Task Time =',tasktime)
                return rawdata

def Save_Data(Data,num,tar,name):
        # print('data', Data)
        Data.to_csv('Training_against' + str(tar) + '_Task' + str(num) + '.csv', index=False)
        # Data.to_csv('Attacker_'+ +'_against_'+str(Tar)+'_Attempt_'+str(num)+'.csv',index=False)
        # with open(User_Name+'_Ex'+str(Experment)+'_Task'+str(Task_NO)+'.csv', 'a') as writeFile:
        #     writer = csv.writer(writeFile, delimiter=',')
        #     writer.writerow(Data)
        # writeFile.close()



def get_object_Position_time_segment(task):
        print('--------------------------------------- ')
        Values_manipSphere_Position = []  #empty array for postion measurements part 1
        Values_Shape_L_Velocity=[]       #empty array for L_Velocity measurements  part 1
        r=[]
        Time=[]

    


        start =time.time()
         
        # Target Position(End Point )
        errorCodeTarget,target = vrep.simxGetObjectHandle(clientID,'Target1',vrep.simx_opmode_oneshot_wait)
        returnCode,target_P=vrep.simxGetObjectPosition(clientID,target,-1,vrep.simx_opmode_oneshot_wait)        
        TPosition= [target_P[0],target_P[1]]
        
         #Target Position(start point) 
        errorCodeTarget,target0 = vrep.simxGetObjectHandle(clientID,'Target2',vrep.simx_opmode_oneshot_wait)
        returnCode,target0_P=vrep.simxGetObjectPosition(clientID,target0,-1,vrep.simx_opmode_oneshot_wait)        
       
        
        end_maze=False

        rawdata=[]

        print("StartMaze")

        while not (end_maze):

             temp = []

             #Sphere Position
             errorCodeTarget,manipSphere = vrep.simxGetObjectHandle(clientID,'Sphere',vrep.simx_opmode_oneshot_wait)
             returnCode,SPosition=vrep.simxGetObjectPosition(clientID,manipSphere,-1,vrep.simx_opmode_oneshot_wait)
             Sphere_P = [SPosition[0],SPosition[1]]


             #Velocity
             errorCodeTarget,Sphere_Velocity = vrep.simxGetObjectHandle(clientID,'Sphere',vrep.simx_opmode_oneshot_wait)
             returnCode, linearVelocity, angularVelocity =vrep.simxGetObjectVelocity( clientID,Sphere_Velocity,vrep.simx_opmode_streaming )
             Shape_L_Velocity = [linearVelocity[0],linearVelocity[1]]

             Magnitude_V = math.sqrt((linearVelocity[0]) ** 2 + (linearVelocity[1]) ** 2)
             Magnitude_P = math.sqrt((SPosition[0]) ** 2 + (SPosition[1]) ** 2)

             T= time.time()-start


             Time.append(T)
             Values_Shape_L_Velocity.append(Shape_L_Velocity)

             Values_manipSphere_Position.append(Sphere_P)
             R=math.sqrt((SPosition[0])**2+(SPosition[1])**2)
             R=round(R,4)
             r.append(R)


             temp.append(task)
             temp.append(T)

             temp.append(Sphere_P[0])
             temp.append(Sphere_P[1])
             temp.append(Magnitude_P)

             temp.append(Shape_L_Velocity[0])
             temp.append(Shape_L_Velocity[1])
             temp.append(Magnitude_V)
             temp.append(R)

             # print('temp=', temp)
             rawdata.append(temp)
             # condition to check if the Sphere position == first test point postion
             if( (round(TPosition[0],1)==round(Sphere_P[0],1)) and (round(Sphere_P[1],1)==round(TPosition[1],1))  ):
                 end_maze=True
                 print("End_maze")


            # end = time.time()
            # tasktime=end - start
            # print('Task Time =',tasktime)


        return rawdata

    


'''---------------------------------------------------------------------------------------------'''        
def X_Y_Acceleration(df):

     # print('T=',df.iloc[:,1].values)
     v0=df.iloc[:,6].values
     v1=df.iloc[:,6].values
     v0y = df.iloc[:, 7].values
     v1y = df.iloc[:, 7].values
     T = df.iloc[:,1].values
     # print(T)
     Accel_x = []
     Accel_y = []
     Accel_Magnitude = []
     Accel_x.append(0)
     Accel_y.append(0)
     Accel_Magnitude.append(0)
     for i in range(len(T) - 1):
         ax = (v1[i+1] - v1[i]) / (T[i + 1] - T[i])
         ay = (v1y[i+1] - v1y[i]) / (T[i + 1] - T[i])
         Magnitude = math.sqrt((ax) ** 2 + (ay) ** 2)
         # print(ax)
         Accel_x.append(ax)
         Accel_y.append(ay)
         Accel_Magnitude.append(Magnitude)

     return Accel_y,Accel_x,Accel_Magnitude
'''---------------------------------------------------------------------------------------------'''                 
def Distance(T,V):
     Dis=[]
     for i in range(len(T)):
         d=V[i] * T[i]
         Dis.append(d)
     return Dis 
'''---------------------------------------------------------------------------------------------'''           
def plotxy():
    dataset = pd.read_csv('test000_Ex1_Task3.csv')
    dataset2 = pd.read_csv('test002_Ex1_Task3.csv')
    # line 1 points
    x1 = dataset.iloc[:, 2:].values
    y1 = dataset.iloc[:, 2:].values
    # plotting the line 1 points
    plt.plot(x1, y1, label="line 1")

    # line 2 points
    x2 = dataset2.iloc[:, 2:].values
    y2 = dataset2.iloc[:, 2:].values
    # plotting the line 2 points
    plt.plot(x2, y2, label="line 2")

    # naming the x axis
    plt.xlabel('x - axis')
    # naming the y axis
    plt.ylabel('y - axis')
    # giving a title to my graph
    plt.title('Two lines on same graph!')

    # show a legend on the plot
    plt.legend()

    # function to show the plot
    plt.show()

'''---------------------------------------------------------------------------------------------'''    
# def Magnitude(V):
#     Velocity=[]
#     for i in range(len(V)):
#        speed=math.sqrt((V[i][0])**2+(V[i][1])**2)
#        Velocity.append(speed)
#
#     return Velocity

'''---------------------------------------------------------------------------------------------'''    
def similarity():
    dataSetI = [3, 45, 7, 2]
    dataSetII = [2, 54, 13, 15]
    result = 1 - spatial.distance.cosine(dataSetI, dataSetII)

def main (num,tar,name):


   # get postion ,velocity ,time form v-rep 
       #P,V,T,r,P1,V1,T1,r1,P2,V2,T2,r2,end_maze,Tasktime=get_object_Position_Point_Segmant()

   vrep.simxStartSimulation(clientID, vrep.simx_opmode_oneshot_wait)
   time.sleep(0.5)
   if Experment==1:
       temp=get_object_Position_time_segment(Task_NO)
       df = pd.DataFrame(temp)
   else:
       temp=get_object_Position_Point_Segmant()
       df = pd.DataFrame(temp)
   Accel_x,Accel_y,Accel_M=X_Y_Acceleration(df)
   # print ('data',df)
   df['Accl_x'] = Accel_x
   df['Accel_y'] = Accel_y
   df['Accel_M'] = Accel_M
   # print('data', df)
   Save_Data(df,num,tar,name)
   # print('data after save', df)
   # similarity()
   print( """ Stops the simulation. """)
   vrep.simxStopSimulation(clientID, vrep.simx_opmode_oneshot)
  
   print('=====================================================')
    # print(' DONE : All Data stored in Excel file{ '+ filename+' }')
   print('         Thank You{ '+User_Name+' }')
   print('=====================================================')
 
'''---------------------------------------------------------------------------------------------'''         
if __name__ == '__main__': 
    main() 

