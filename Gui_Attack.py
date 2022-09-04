
"""
Created on Tue Sep  2020

@author: shurook Almohamade
"""

import tkinter as tk
import tkinter.font as tkFont
from tkinter import ttk
from tkinter import *
import Feedback_Attack as FB
import collect_data_Attack

class App:


    All_Attemp=[]
    def __init__(self, root):
        #setting title
        root.title("Real Attacks")
        #setting window size
        width=280
        height=280
        screenwidth = root.winfo_screenwidth()
        screenheight = root.winfo_screenheight()
        alignstr = '%dx%d+%d+%d' % (width, height, (screenwidth - width) / 2, (screenheight - height) / 2)
        root.geometry(alignstr)
        root.resizable(width=False, height=False)

        l1 = tk.Label(root, text='Attacker ID: ', width=10)  # added one Label
        l1.grid(row=1, column=1)

        self.t1 = tk.Text(root, height=1, width=9, highlightthickness=2,
                          highlightbackground="#333333")  # added one text box
        self.t1.grid(row=1, column=2)

        var0 = tk.Label(root, text='Attacke Type:', width=10)  # added one Label
        var0.grid(row=2, column=1)

        self.var = IntVar()
        R1 = Radiobutton(root, text="Training", variable=self.var, value=1).grid(row=2, column=2, sticky=W)
        # R1.pack(anchor=W)

        R2 = Radiobutton(root, text="Video", variable=self.var, value=2).grid(row=2, column=3, sticky=W)

        self.var.set(1)

        GLabel_814=tk.Label(root)
        ft = tkFont.Font(family='Times',size=14)

        GLabel_814["text"] = "Targeted User "
        GLabel_814.grid(row=4, column=1)
        GLabel_814.place(x=40,y=60,width=100,height=25)
        #
        # Combobox creation

        self.n = tk.StringVar()
        Attackerchoosen = ttk.Combobox(root, width=27, textvariable=self.n)
        Attackerchoosen['values'] = (1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19)
        # Attackerchoosen.grid(row=4, column=2)
        Attackerchoosen.place(x=180, y=60, width=60, height=25)
        defult = self.n.set(1)
        Targeted = self.n.get()
        FB.Targeted_plot(Targeted)


        GLabel_95=tk.Label(root)

        GLabel_95["text"] = "Attacker Attempt #"

        GLabel_95.place(x=40,y=90,width=120,height=25)
        # #
        self.n2 = tk.StringVar()
        vic_choosen = ttk.Combobox(root, width=27, textvariable=self.n2)
        vic_choosen['values'] = (1,2,3,4,5,6,7,8,9,10,11,12,13,14,15)

        vic_choosen.place(x=180, y=90, width=60, height=25)
        defult=self.n2.set(1)



        X = 40
        Y = 140
        GButton_282 = tk.Button(root)

        GButton_282["text"] = "Run The Simulation "

        GButton_282.place(x=X, y=Y, width=150, height=30)
        GButton_282["command"] = self.runsimulation

        GButton_283 = tk.Button(root)

        GButton_283["text"] = "Get The Result "

        GButton_283.place(x=X, y=Y+40, width=150, height=30)
        GButton_283["command"] = self.feedback


    def runsimulation(self):

        Attacker = self.n2.get()
        Targeted = self.n.get()
        name = self.t1.get("1.0",END)
        Type=str(self.var.get())
        collect_data_Attack.main(name,Attacker,Targeted,Type)




    def feedback(self):
        Attacker = self.n2.get()
        Targeted = self.n.get()
        name = self.t1.get("1.0", END)
        Type=str(self.var.get())

        print('Attacker=', Attacker, 'Targeted=', Targeted)
        probability_seg, profile_sim,task_probability= FB.plot_similarity(self, Attacker, Targeted,name,Type)


        print('Decision( User ', profile_sim,')')
        index=int(Targeted)-1
        probability_seg1=round(probability_seg[0],2)
        probability_seg2 = round(probability_seg[1],2)
        probability_seg3 = round(probability_seg[2],2)
        print('probability of User ', Targeted, 'part1 =',probability_seg[0])
        print('probability of User ', Targeted, 'part1 =', probability_seg[1])
        print('probability of User ', Targeted, 'part1 =', probability_seg[2])
        if profile_sim == "Accepted":
            profile_sim = "Accepted"
            color = "green"
            framesize="200x200"

        else:

            profile_sim = "Rejected"
            framesize = "200x200"
            color = "red"

        X = 10
        Y = 30
        toplevel = Toplevel()
        toplevel.title('The Result')
        toplevel.geometry(framesize)
        toplevel.focus_set()

        top_frame = Frame(toplevel)
        top_frame.pack(side=TOP, pady=5)

        self.All_Attemp.append((profile_sim))
        print(self.All_Attemp)
        # ==============================================

        similarity_profile = tk.Label(toplevel)
        ft = tkFont.Font(family='Times', size=16)
        similarity_profile["font"] = ft
        similarity_profile["fg"] = "#333333"
        similarity_profile["justify"] = "center"
        similarity_profile["text"] = " Decision"
        similarity_profile.place(x=X+40, y=Y)

        similarity_Velocity = tk.Label(toplevel)
        ft = tkFont.Font(family='Times', size=16)
        similarity_Velocity["font"] = ft
        similarity_Velocity["fg"] = color
        similarity_Velocity["justify"] = "center"
        similarity_Velocity["text"] = profile_sim
        similarity_Velocity.place(x=X+40 , y=Y+50 )

        bottom_frame = Frame(toplevel)
        bottom_frame.pack(side=BOTTOM, pady=5)

        # Close Button
        self.btn_run = Button(bottom_frame, text="Quit", width=15, command=toplevel.destroy)
        self.btn_run.grid(row=0, column=2)





if __name__ == "__main__":


    root = tk.Tk()
    app = App(root)
    root.mainloop()

