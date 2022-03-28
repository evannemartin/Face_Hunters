import tkinter
import pandas as pds
from PIL import ImageTk, Image
import random


characteristics={"woman":False,"man":False,"young":False,"old":False,"beard":False,"no_beard":False,"straight":False,"no_straight":False}
print(characteristics)

def initialize() :
    for cle, valeur in characteristics.items() :
        characteristics[cle]=False
    print(characteristics)

def onClick(event):
    for c in myWindow.winfo_children():
        c.destroy()
    initialize()
    mainFrame=tkinter.Frame(myWindow,bg='white',width=30, borderwidth=3, relief='groove')
    mainFrame.pack(side='top', padx=30, pady=30,fill="x" )
    tkinter.Label(mainFrame,bg='white',text='What are the agressor characteristics?',font=(10)).pack(padx=20,pady=20)

    #the women/man frame
    Frame1=tkinter.Frame(myWindow,borderwidth=3, relief='groove')
    Frame1.pack(padx=5, pady=5)

    #to select only one option
    value = tkinter.StringVar()
    fButton=tkinter.Radiobutton(Frame1,text="Woman",variable=value, value=1)
    fButton.pack(side='left', padx=5, pady=5)
    fButton.bind('<Button-1>', fevent)

    hButton=tkinter.Radiobutton(Frame1,text="Man",variable=value, value=2)
    hButton.pack(side='right', padx=5, pady=5)
    hButton.bind('<Button-1>', hevent)

    #the young/old frame
    Frame2=tkinter.Frame(myWindow,borderwidth=3, relief='groove')
    Frame2.pack(padx=5, pady=5)

    value = tkinter.StringVar()
    yButton=tkinter.Radiobutton(Frame2,text="Young",variable=value, value=1)
    yButton.pack(side='left', padx=5, pady=5)
    yButton.bind('<Button-1>', yevent)

    oButton=tkinter.Radiobutton(Frame2,text="Old",variable=value, value=2)
    oButton.pack(side='right', padx=5, pady=5)
    oButton.bind('<Button-1>', oevent)

    #the beard/no beard frame
    Frame3=tkinter.Frame(myWindow,borderwidth=3, relief='groove')
    Frame3.pack(padx=5, pady=5)

    value = tkinter.StringVar()
    bButton=tkinter.Radiobutton(Frame3,text="Beard",variable=value, value=1)
    bButton.pack(side='left', padx=5, pady=5)
    bButton.bind('<Button-1>', bevent)

    nbButton=tkinter.Radiobutton(Frame3,text="No Beard",variable=value, value=2)
    nbButton.pack(side='right', padx=5, pady=5)
    nbButton.bind('<Button-1>', nbevent)

    #the straight/no straight hair frame
    Frame4=tkinter.Frame(myWindow,borderwidth=3, relief='groove')
    Frame4.pack(padx=5, pady=5)

    value = tkinter.StringVar()
    sButton=tkinter.Radiobutton(Frame4,text="Straight Hair",variable=value, value=1)
    sButton.pack(side='left', padx=5, pady=5)
    sButton.bind('<Button-1>', sevent)

    nsButton=tkinter.Radiobutton(Frame4,text="No Straight Hair",variable=value, value=2)
    nsButton.pack(side='right', padx=5, pady=5)
    nsButton.bind('<Button-1>', nsevent)

    #to clear the characteristics is there is an error
    myButton=tkinter.Button(myWindow,text='Missclicking ? Do It Again !', width=50, bg="yellow")
    myButton.pack(padx=20, pady=20, fill="x")
    myButton.bind('<ButtonRelease-1>',onClick)

    #show a new window to propose some pictures depending on the characteristics
    myButton=tkinter.Button(myWindow,text='Suivant', width=50, bg="yellow", font=(1))
    myButton.pack(padx=20, pady=20, fill="x")
    myButton.bind('<ButtonRelease-1>',next)

def fevent(event):
    characteristics["woman"]=True

def hevent(event):
    characteristics["man"]=True

def yevent(event):
    characteristics["young"]=True

def oevent(event):
    characteristics["old"]=True

def bevent(event):
    characteristics["beard"]=True

def nbevent(event):
    characteristics["no_beard"]=True

def sevent(event):
    characteristics["straight"]=True

def nsevent(event):
    characteristics["no_straight"]=True

#base de données correspondant aux critères
def choice_database(char):
    if char["woman"] and char["young"] and char["straight"]:
        return 'female_young_straight.csv'
    if char["woman"] and char["young"] and char["no_straight"]:
        return 'female_young_wavy.csv'
    if char["woman"] and char["old"] and char["straight"]:
        return 'female_old_straight.csv'
    if char["woman"] and char["old"] and char["no_straight"]:
        return 'female_old_wavy.csv'
    if char["man"] and char["young"] and char["beard"]:
        return 'male_young_beard.csv'
    if char["man"] and char["young"] and char["no_beard"]:
        return 'male_young_nobeard.csv'
    if char["man"] and char["old"] and char["beard"]:
        return 'male_old_beard.csv'
    if char["man"] and char["old"] and char["no_beard"]:
        return 'male_old_nobeard.csv'

#choix d'une image dans la base choisie
def choice_image(database) :
    db = pds.read_csv(database, sep=",")
    #print(len(db))
    list_im=[]
    list_ind=[]
    for i in range(10):
        ind=random.randint(0,len(db)-1)
        while ind in list_ind : #so it shows distinct pictures
            ind=random.randint(0,len(database)-1)
        print(ind)
        list_im.append(db["image_id"][ind])
        list_ind.append(ind)

    return list_im


def next(event):
    database=choice_database(characteristics)
    for c in myWindow.winfo_children():
        c.destroy()
    print(characteristics)
    tkinter.Label(myWindow,text='Propose some pictures',font=(10)).grid(row=0,column=2)
    #.pack(padx=10,pady=10)
    im=choice_image(database)
    source_im=[]
    photo=[]
    for i in range (len(im)) :
        source_im.append('../database/img_align_celeba/img_align_celeba/'+im[i])
        photo.append(ImageTk.PhotoImage(Image.open(source_im[i])))
        # imLab2=tkinter.Label(myWindow,image=photo[i])
        # imLab2.pack(padx=5,pady=5, side="left")
        # imLab2.bind('<Button-1>', chooseimage(event))

    #canv = tkinter.Canvas(myWindow, width=80, height=80, bg='white')
    #canv.grid(row=2, column=
    #chooseBtn=canv.create_image(0.5, 0.5, image=photo)

    for i in range(5):
        myWindow.columnconfigure(i, weight=1)

    myWindow.rowconfigure(1, weight=1)

    # cadre=tkinter.Frame(myWindow)
    # cadre.pack(side="left")
    #
    # cadrebis=tkinter.Frame(myWindow)
    # cadrebis.pack(side="bottom")

    imLab1=tkinter.Label(myWindow,image=photo[0])
    imLab1.grid(row=1, column=0)
    imLab1.bind('<Button-1>', chooseimage)

    imLab2=tkinter.Label(myWindow,image=photo[1])
    imLab2.grid(row=1, column=1)
    imLab2.bind('<Button-1>', chooseimage)

    imLab3=tkinter.Label(myWindow,image=photo[2])
    imLab3.grid(row=1, column=2)
    imLab3.bind('<Button-1>', chooseimage)

    imLab4=tkinter.Label(myWindow,image=photo[3])
    imLab4.grid(row=1, column=3)
    imLab4.bind('<Button-1>', chooseimage)

    imLab5=tkinter.Label(myWindow,image=photo[4])
    imLab5.grid(row=1, column=4)
    imLab5.bind('<Button-1>', chooseimage)

    imLab6=tkinter.Label(myWindow,image=photo[5])
    imLab6.grid(row=2, column=0)
    imLab6.bind('<Button-1>', chooseimage)

    imLab7=tkinter.Label(myWindow,image=photo[6])
    imLab7.grid(row=2, column=1)
    imLab7.bind('<Button-1>', chooseimage)

    imLab8=tkinter.Label(myWindow,image=photo[7])
    imLab8.grid(row=2, column=2)
    imLab8.bind('<Button-1>', chooseimage)

    imLab9=tkinter.Label(myWindow,image=photo[8])
    imLab9.grid(row=2, column=3)
    imLab9.bind('<Button-1>', chooseimage)

    imLab10=tkinter.Label(myWindow,image=photo[9])
    imLab10.grid(row=2, column=4)
    imLab10.bind('<Button-1>', chooseimage)


    myWindow.mainloop()

def chooseimage(event) :
    for c in myWindow.winfo_children():
        c.destroy()
    tkinter.Label(myWindow,text='Propose child pictures').pack(padx=10,pady=10)
    #print(i)

#First Window

myWindow=tkinter.Tk()
myWindow.geometry("1000x500")
myWindow['bg']='white'

Frame=tkinter.Frame(myWindow,borderwidth=3, relief='groove')
Frame.pack(side='top', padx=10, pady=10,expand="yes",fill="both")
title= tkinter.Label(Frame,text='Project 4BIM',font=(20))
title.place(relx=0.5, rely=0.25, anchor="center")

descLabel=tkinter.Label(Frame, text='This is an application developped by 4BIM INSA students to help you create a robot portait of your agressor. ')
descLabel.place(relx=0.5, rely=0.5, anchor="center")
#descLabel.pack()

myButton=tkinter.Button(Frame,text='Start !', width=50, bg="yellow")
myButton.place(relx=0.5, rely=0.75, anchor="center")
myButton.bind('<ButtonRelease-1>',onClick)


myWindow.mainloop()
