import tkinter

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
    mainFrame.pack(side='top', padx=10, pady=10)
    tkinter.Label(mainFrame,bg='white',text='What are the agressor characteristics?').pack(padx=10,pady=10)

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
    myButton.pack(padx=5, pady=5)
    myButton.bind('<ButtonRelease-1>',onClick)

    #show a new window to propose some pictures depending on the characteristics
    myButton=tkinter.Button(myWindow,text='Suivant', width=50, bg="yellow")
    myButton.pack(padx=5, pady=5)
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

def next(event):
    for c in myWindow.winfo_children():
        c.destroy()
    print(characteristics)
    tkinter.Label(myWindow,text='Propose some pictures').pack(padx=10,pady=10)
#First Window

myWindow=tkinter.Tk()
myWindow['bg']='white'

Frame1=tkinter.Frame(myWindow,borderwidth=3, relief='groove')
Frame1.pack(side='top', padx=10, pady=10)
tkinter.Label(Frame1,text='Project 4BIM').pack(padx=10,pady=10)

descLabel=tkinter.Label(Frame1, text='This is an application developped by 4BIM INSA students to help you create a robot portait of your agressor. ')
descLabel.pack()

myButton=tkinter.Button(Frame1,text='Start !', width=50, bg="yellow")
myButton.pack()
myButton.bind('<ButtonRelease-1>',onClick)


myWindow.mainloop()
