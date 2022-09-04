
"""
Created on Tue Sep  2020

@author: shurook Almohamade
"""


import tkinter as tk



import tkinter.font as tkFont
from tkinter import ttk

import numpy as np

from tkinter import *

from cv2 import cv2

from PIL import Image, ImageTk
import Feedback2 as FB
import collect_data_Training


class App:


    All_Attemp=[]
    def __init__(self, root):
        #setting title
        root.title("Offline Training Attacks")
        #setting window size
        width=280
        height=480
        self.counter=0
        screenwidth = root.winfo_screenwidth()
        screenheight = root.winfo_screenheight()
        alignstr = '%dx%d+%d+%d' % (width, height, (screenwidth - width) / 2, (screenheight - height) / 2)
        root.geometry(alignstr)
        root.resizable(width=False, height=False)

        l1 = tk.Label(root, text='', width=10)  # added one Label
        l1.grid(row=1, column=1)
        self.t1 = tk.Text(root, height=1, width=9, highlightthickness=2)  # added one text box
        self.t1.grid(row=1, column=2)

        GLabel_814=tk.Label(root)
        ft = tkFont.Font(family='Times',size=14)
        GLabel_814["font"] = ft
        GLabel_814["fg"] = "#333333"
        GLabel_814["justify"] = "center"
        GLabel_814["text"] = "Targeted User "
        GLabel_814.place(x=40,y=50,width=100,height=25)
        #
        # Combobox creation

        self.n = tk.StringVar()
        Attackerchoosen = ttk.Combobox(root, width=27, textvariable=self.n)
        Attackerchoosen['values'] = (1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32)
        Attackerchoosen.place(x=160, y=50, width=60, height=25)
        defult = self.n.set(4)
        Targeted = self.n.get()
        FB.Targeted_plot(Targeted)



        self.n2 = tk.StringVar()
        vic_choosen = ttk.Combobox(root, width=27, textvariable=self.n2)
        vic_choosen['values'] = (1,2,3,4)

        defult=self.n2.set(1)

        GButton_281=tk.Button(root)
        GButton_281["bg"] = "#efefef"
        ft = tkFont.Font(family='Times',size=12)
        GButton_281["font"] = ft
        GButton_281["fg"] = "#000000"
        GButton_281["justify"] = "center"
        GButton_281["text"] = "Get Initial Information "
        GButton_281.place(x=40,y=100,width=150,height=30)
        GButton_281["command"] = self.information
        #
        GButton_282 = tk.Button(root)
        GButton_282["bg"] = "#efefef"
        ft = tkFont.Font(family='Times', size=12)
        GButton_282["font"] = ft
        GButton_282["fg"] = "#000000"
        GButton_282["justify"] = "center"
        GButton_282["text"] = "Run The Simulation "
        GButton_282.place(x=40, y=140, width=150, height=30)
        GButton_282["command"] = self.runsimulation

        GButton_283 = tk.Button(root)
        GButton_283["bg"] = "#efefef"
        ft = tkFont.Font(family='Times', size=12)
        GButton_283["font"] = ft
        GButton_283["fg"] = "#000000"
        GButton_283["justify"] = "center"
        GButton_283["text"] = "Get The Feedback "
        GButton_283.place(x=40, y=180, width=150, height=30)
        GButton_283["command"] = self.feedback


    def runsimulation(self):
        self.counter= self.counter+1
        Targeted = self.n.get()
        name = self.t1.get("1.0", END)
        print('Attempt #',self.counter)
        # call(["python", "collect_data_Attack.py"])
        Attacker = self.n2.get()
        collect_data_Training.main(self.counter,Targeted,name)

    def information(self):
        Targeted = self.n.get()
        FB.Targeted_plot(Targeted)
        load = Image.open('Targeted_' + Targeted + '.png')
        load = load.resize((250, 250), Image.ANTIALIAS)
        load = load.rotate(270)

        render = ImageTk.PhotoImage(load)

        img = tk.Label(root, image=render)
        img.image = render
        img.place(x=20, y=220)


    def feedback(self):
        Attacker = 1
        Targeted = self.n.get()


        print('Attacker=', Attacker, 'Targeted=', Targeted)
        probability_seg, profile_sim,task_probability,MAE,RMSE=\
            FB.plot_similarity(self, Attacker, Targeted,self.counter)

        PT_1, PT_2, PT_3, vm1_1, vm1_2, vm1_3,\
        PT_1A, PT_2A,PT_3A, vm2_1A, vm2_2A, vm2_3A,\
        DT1,DT2,DT3,DA1,DA2,DA3 = FB.Position_Velocity_similarity(self)
        T1 = PT_1-PT_1A
        print('T1',T1)


        print('Decision( User ', profile_sim,')')
        index=int(Targeted)-1
        probability_seg1=round(probability_seg[0],2)
        probability_seg2 = round(probability_seg[1],2)
        probability_seg3 = round(probability_seg[2],2)
        print('probability of User ', Targeted, 'part1 =',probability_seg[0])
        print('probability of User ', Targeted, 'part1 =', probability_seg[1])
        print('probability of User ', Targeted, 'part1 =', probability_seg[2])

        MAE1 = round(MAE[0], 2)
        MAE2 = round(MAE[1], 2)
        MAE3 = round(MAE[2], 2)

        if profile_sim == "Accepted":
            # profile_sim = "Accepted"
            color = "green"
            feedbk = ""
            accept = 1
            framesize="460x600"
            X=10
            Y=30
        else:
            profile_sim = "Rejected"
            accept = 0
            framesize = "500x800"
            color = "red"
            if PT_1 < PT_1A:
                if round(vm1_1,1)>round(vm2_1A,1):
                    feedbk = 'speed up'
                elif round(vm1_1, 1) < round(vm2_1A, 1):
                    feedbk = '(slow down)'
                elif round(vm1_1,1)==round(vm2_1A,1):
                    if DT1<DA1:
                         feedbk = 'Pay attention to the path'
                    else:
                        feedbk = 'Pay attention to the path'


            else:
                if round(vm1_1,1)>round(vm2_1A,1):
                    feedbk = 'Speed up'
                elif round(vm1_1, 1) < round(vm2_1A, 1):
                    feedbk = 'Slow down'
                elif round(vm1_1,1)==round(vm2_1A,1):
                    if DT1 < DA1:
                        feedbk = 'Pay attention to the path'#long
                    else:
                        feedbk = 'Pay attention to the path '



            # ==============feedbk2=========================
            if PT_2 < PT_2A:

                if round(vm1_2, 1) > round(vm2_2A, 1):
                    feedbk2 = "Speed up"
                elif round(vm1_2, 1) < round(vm2_2A, 1):
                    feedbk2 = "Slow down"
                elif round(vm1_2, 1) == round(vm2_2A, 1):
                    if DT2 < DA2:
                        feedbk2 = "Pay attention to the path"
                    else:
                        feedbk2 = "Pay attention to the path"

            else:
                if round(vm1_2, 1) > round(vm2_2A, 1):
                    feedbk2 = 'Speed up'
                elif round(vm1_2, 1) < round(vm2_2A, 1):
                    feedbk2 = 'Slow down'
                elif round(vm1_2, 1) == round(vm2_2A, 1):
                    if DT2 < DA2:
                        feedbk2 = 'Pay attention to the path'
                    else:
                        feedbk2 = 'Pay attention to the path'
                else:
                    feedbk2 = 'Pay attention to the path'

            # ==============feedbk3=========================
            if PT_3 < PT_3A:
                if round(vm1_3, 1) > round(vm2_3A, 1):
                    feedbk3 = '(Speed up)'
                elif round(vm1_3, 1) < round(vm2_3A, 1):
                    feedbk3 = 'Slow down'
                elif round(vm1_3, 1) == round(vm2_3A, 1):
                    if DT3 < DA3:
                        feedbk3 = 'pay attention to the path'
                    else:
                        feedbk3 = 'pay attention to the path'
                else:
                    feedbk3 = 'pay attention to the path'



            else:
                if round(vm1_3, 1) > round(vm2_3A, 1):
                    feedbk3 = "Speed up"
                elif round(vm1_3, 1) < round(vm2_3A, 1):
                    feedbk3 = "Slow down"
                elif round(vm1_3, 1) == round(vm2_3A, 1):
                    if DT3 < DA3:
                        feedbk3 = "Pay attention to the path"
                    else:
                        feedbk3 = "Pay attention to the path"
                else:
                    feedbk3 = "Pay attention to the path"

        toplevel = Toplevel()
        toplevel.title('Feedback')
        toplevel.geometry(framesize)
        toplevel.focus_set()

        top_frame = Frame(toplevel)
        top_frame.pack(side=TOP, pady=5)





        self.All_Attemp.append((profile_sim))
        print(self.All_Attemp)
        # ==============================================
        if accept==0 or accept==1:
            load = Image.open('Attacker_' + str(Attacker) + '.png')
            load = load.resize((330, 340), Image.ANTIALIAS)
            load = load.rotate(270)
            render = ImageTk.PhotoImage(load)

            img = tk.Label(toplevel, image=render)
            img.image = render
            img.pack(side=TOP)
            X = 40
            Y = 320

            similarity_p = tk.Label(toplevel)
            ft = tkFont.Font(family='Times', size=16)
            similarity_p["font"] = ft
            similarity_p["fg"] = "#333333"
            similarity_p["justify"] = "center"
            similarity_p["text"] = "Time:"
            similarity_p.place(x=X, y=Y)


            P1 = tk.Label(toplevel)
            ft = tkFont.Font(family='Times', size=14)
            P1["font"] = ft
            P1["fg"] = "#333333"
            P1["justify"] = "center"
            P1["text"] = "Part1"
            P1.place(x=X+80, y=Y)

            P2 = tk.Label(toplevel)
            ft = tkFont.Font(family='Times', size=14)
            P2["font"] = ft
            P2["fg"] = "#333333"
            P2["justify"] = "center"
            P2["text"] = "Part 2"
            P2.place(x=X + 140, y=Y)

            P2 = tk.Label(toplevel)
            ft = tkFont.Font(family='Times', size=14)
            P2["font"] = ft
            P2["fg"] = "#333333"
            P2["justify"] = "center"
            P2["text"] = "Part 3"
            P2.place(x=X + 220, y=Y)



            time = tk.Label(toplevel)
            ft = tkFont.Font(family='Times', size=13)
            time["font"] = ft
            time["fg"] = "blue"
            time["justify"] = "center"
            time["text"] = "Targeted:"
            time.place(x=X + 10, y=Y + 20)

            similarity_Position = tk.Label(toplevel)
            ft = tkFont.Font(family='Times', size=12)
            similarity_Position["font"] = ft
            similarity_Position["fg"] = "blue"
            similarity_Position["justify"] = "center"
            similarity_Position["text"] = np.round(PT_1,1)
            similarity_Position.place(x=X+80, y=Y+20)

            similarity_Position2 = tk.Label(toplevel)
            ft = tkFont.Font(family='Times', size=12)
            similarity_Position2["font"] = ft
            similarity_Position2["fg"] = "blue"
            similarity_Position2["justify"] = "center"
            similarity_Position2["text"] = np.round(PT_2, 1)
            similarity_Position2.place(x=X + 140, y=Y + 20)

            similarity_Position3 = tk.Label(toplevel)
            ft = tkFont.Font(family='Times', size=12)
            similarity_Position3["font"] = ft
            similarity_Position3["fg"] = "blue"
            similarity_Position3["justify"] = "center"
            similarity_Position3["text"] = np.round(PT_3, 1)
            similarity_Position3.place(x=X + 210, y=Y + 20)

            #  Attaker time
            similarity_V = tk.Label(toplevel)
            ft = tkFont.Font(family='Times', size=13)
            similarity_V["font"] = ft
            similarity_V["fg"] = color
            similarity_V["justify"] = "center"
            similarity_V["text"] = "You:"
            similarity_V.place(x=X + 10, y=Y + 40)

            similarity_Position = tk.Label(toplevel)
            ft = tkFont.Font(family='Times', size=12)
            similarity_Position["font"] = ft
            similarity_Position["fg"] = color
            similarity_Position["justify"] = "center"
            similarity_Position["text"] = np.round(PT_1A, 1)
            similarity_Position.place(x=X+80, y=Y +40)

            similarity_Position2 = tk.Label(toplevel)
            ft = tkFont.Font(family='Times', size=12)
            similarity_Position2["font"] = ft
            similarity_Position2["fg"] = color
            similarity_Position2["justify"] = "center"
            similarity_Position2["text"] = np.round(PT_2A, 1)
            similarity_Position2.place(x=X + 140, y=Y + 40)

            similarity_Position3 = tk.Label(toplevel)
            ft = tkFont.Font(family='Times', size=12)
            similarity_Position3["font"] = ft
            similarity_Position3["fg"] = color
            similarity_Position3["justify"] = "center"
            similarity_Position3["text"] = np.round(PT_3A, 1)
            similarity_Position3.place(x=X + 210, y=Y + 40)



            X = 40
            Y = 380

            similarity_p = tk.Label(toplevel)
            ft = tkFont.Font(family='Times', size=16)
            similarity_p["font"] = ft
            similarity_p["fg"] = "#333333"
            similarity_p["justify"] = "center"
            similarity_p["text"] = "Speed:"
            similarity_p.place(x=X, y=Y)



            velocity = tk.Label(toplevel)
            ft = tkFont.Font(family='Times', size=13)
            velocity["font"] = ft
            velocity["fg"] = "blue"
            velocity["justify"] = "center"
            velocity["text"] = "Targeted:"
            velocity.place(x=X + 10, y=Y + 20)

            similarity_V = tk.Label(toplevel)
            ft = tkFont.Font(family='Times', size=13)
            similarity_V["font"] = ft
            similarity_V["fg"] = color
            similarity_V["justify"] = "center"
            similarity_V["text"] = "You:"
            similarity_V.place(x=X + 10, y=Y + 40)


            V1 = tk.Label(toplevel)
            ft = tkFont.Font(family='Times', size=12)
            V1["font"] = ft
            V1["fg"] = "blue"
            V1["justify"] = "center"
            V1["text"] = np.round(vm1_1, 2)
            V1.place(x=X + 80, y=Y + 20)

            V2 = tk.Label(toplevel)
            ft = tkFont.Font(family='Times', size=12)
            V2["font"] = ft
            V2["fg"] = "blue"
            V2["justify"] = "center"
            V2["text"] = np.round(vm1_2, 2)
            V2.place(x=X + 140, y=Y + 20)

            V3 = tk.Label(toplevel)
            ft = tkFont.Font(family='Times', size=12)
            V3["font"] = ft
            V3["fg"] = "blue"
            V3["justify"] = "center"
            V3["text"] = np.round(vm1_3, 2)
            V3.place(x=X + 210, y=Y + 20)
            #
            V1A = tk.Label(toplevel)
            ft = tkFont.Font(family='Times', size=12)
            V1A["font"] = ft
            V1A["fg"] = color
            V1A["justify"] = "center"
            V1A["text"] = np.round(vm2_1A, 2)
            V1A.place(x=X + 80, y=Y + 40)

            V2A = tk.Label(toplevel)
            ft = tkFont.Font(family='Times', size=12)
            V2A["font"] = ft
            V2A["fg"] = color
            V2A["justify"] = "center"
            V2A["text"] = np.round(vm2_2A, 2)
            V2A.place(x=X + 140, y=Y + 40)

            V3A = tk.Label(toplevel)
            ft = tkFont.Font(family='Times', size=12)
            V3A["font"] = ft
            V3A["fg"] = color
            V3A["justify"] = "center"
            V3A["text"] = np.round(vm2_3A, 2)
            V3A.place(x=X + 210, y=Y + 40)



        X = 40
        Y = 450

        probability = tk.Label(toplevel)
        ft = tkFont.Font(family='Times', size=16)
        probability["font"] = ft
        probability["fg"] = "#333333"
        probability["justify"] = "center"
        probability["text"] = "Decision:"
        probability.place(x=X , y=Y )

        if probability_seg1>0.50:

            Decision='Accepted'
            segcolor = 'green'
        else:
            Decision = 'Rejected'
            segcolor = 'red'
        prob1 = tk.Label(toplevel)
        ft = tkFont.Font(family='Times', size=13)
        prob1["font"] = ft
        prob1["fg"] = segcolor
        prob1["justify"] = "center"
        prob1["text"] = Decision
        prob1.place(x=X + 80, y=Y )
        if probability_seg2 > 0.50:

            Decision = 'Accepted'
            segcolor = 'green'
        else:
            Decision = 'Rejected'
            segcolor = 'red'

        prob2 = tk.Label(toplevel)
        ft = tkFont.Font(family='Times', size=13)
        prob2["font"] = ft
        prob2["fg"] = segcolor
        prob2["justify"] = "center"
        prob2["text"] = Decision
        prob2.place(x=X + 140, y=Y )
        if probability_seg3 > 0.50:

            Decision = 'Accepted'
            segcolor = 'green'
        else:
            Decision = 'Rejected'
            segcolor = 'red'
        prob3 = tk.Label(toplevel)
        ft = tkFont.Font(family='Times', size=13)
        prob3["font"] = ft
        prob3["fg"] = segcolor
        prob3["justify"] = "center"
        prob3["text"] = Decision
        prob3.place(x=X + 210, y=Y )

        if task_probability > 0.65:
            Decision1 = 'Accepted'
            segcolor1 = 'green'
        else:
            Decision1 = 'Rejected'
            segcolor1 = 'red'

        similarity_profile = tk.Label(toplevel)
        ft = tkFont.Font(family='Times', size=15)
        similarity_profile["font"] = ft
        similarity_profile["fg"] = "#333333"
        similarity_profile["justify"] = "center"
        similarity_profile["text"] = 'Whole Task Decision:'
        similarity_profile.place(x=X, y=Y+60)

        similarity_Velocity = tk.Label(toplevel)
        ft = tkFont.Font(family='Times', size=14)
        similarity_Velocity["font"] = ft
        similarity_Velocity["fg"] = segcolor1
        similarity_Velocity["justify"] = "center"
        similarity_Velocity["text"] = Decision1, round((task_probability),1)*100,'%'
        similarity_Velocity.place(x=X+180 , y=Y + 60)

        probability = tk.Label(toplevel)
        ft = tkFont.Font(family='Times', size=16)
        probability["font"] = ft
        probability["fg"] = "#333333"
        probability["justify"] = "center"
        probability["text"] = "Probability:"
        probability.place(x=X , y=Y+30 )

        if probability_seg1>=0.50:
            segcolor='green'
            feedbk='Good'
        else:
            segcolor = 'red'
        prob1 = tk.Label(toplevel)
        ft = tkFont.Font(family='Times', size=13)
        prob1["font"] = ft
        prob1["fg"] = segcolor
        prob1["justify"] = "center"
        prob1["text"] = round(probability_seg1*100,2),'%'
        prob1.place(x=X + 80, y=Y+30 )
        if probability_seg2>0.50:
            segcolor='green'
            feedbk2='Good'
        else:
            segcolor = 'red'

        prob2 = tk.Label(toplevel)
        ft = tkFont.Font(family='Times', size=13)
        prob2["font"] = ft
        prob2["fg"] = segcolor
        prob2["justify"] = "center"
        prob2["text"] = round(probability_seg2*100,2) ,'%'
        prob2.place(x=X + 140, y=Y+30 )
        if probability_seg3>=0.50:
            segcolor='green'
            feedbk3 = 'Good'
        else:
            segcolor = 'red'
        prob3 = tk.Label(toplevel)
        ft = tkFont.Font(family='Times', size=13)
        prob3["font"] = ft
        prob3["fg"] = segcolor
        prob3["justify"] = "center"
        prob3["text"] = round(probability_seg3*100,2) ,'%'
        prob3.place(x=X + 210, y=Y+30 )



        bottom_frame = Frame(toplevel)
        bottom_frame.pack(side=BOTTOM, pady=5)



        # run Button
        self.btn_run = Button(bottom_frame, text="Run", width=15, command=self.runsimulation)
        self.btn_run.grid(row=0, column=0)

        # Update Feedback Button
        self.btn_run = Button(bottom_frame, text="Update Feedback", width=15, command=self.feedback)
        self.btn_run.grid(row=0, column=1)

        # Close Button
        self.btn_run = Button(bottom_frame, text="Quit", width=15, command=toplevel.destroy)
        self.btn_run.grid(row=0, column=2)

        if accept == 0:
            line=70
            similarity_profile = tk.Label(toplevel)
            ft = tkFont.Font(family='Times', size=15)
            similarity_profile["font"] = ft
            similarity_profile["fg"] = "#333333"
            similarity_profile["justify"] = "center"
            similarity_profile["text"] = "Feedback:"
            similarity_profile.place(x=X, y=Y + line+20)

            L = tk.Label(toplevel)
            ft = tkFont.Font(family='Times', size=15)
            L["font"] = ft
            L["fg"] = "darkred"
            L["justify"] = "center"
            L["text"] ="Part 1: "+feedbk
            L.place(x=X + 110, y=Y + line+20)

            L2 = tk.Label(toplevel)
            ft = tkFont.Font(family='Times', size=15)
            L2["font"] = ft
            L2["fg"] = "darkred"
            L2["justify"] = "center"
            L2["text"] = 'Part 2: '+ feedbk2
            L2.place(x=X + 110, y=Y + line+40)

            L3 = tk.Label(toplevel)
            ft = tkFont.Font(family='Times', size=15)
            L3["font"] = ft
            L3["fg"] = "darkred"
            L3["justify"] = "center"
            L3["text"] = 'Part 3: '+ feedbk3
            L3.place(x=X + 110, y=Y + line+60)



    def video(self):
        while (True):
            Targeted = self.n.get()
            self.filename = "video/User" + Targeted + ".mp4"
            cap = cv2.VideoCapture(self.filename)




            while (cap.isOpened()):

                ret, frame = cap.read()

                if ret == True:

                    cv2.imshow("video/User" + Targeted + ".mp4",frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                else:
                    break

            if cv2.waitKey(0) == ord('w'):
                break
        # Release everything if job is finished
        cap.release()
        cv2.destroyAllWindows()






if __name__ == "__main__":

    root = tk.Tk()
    app = App(root)
    root.mainloop()
