#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  7 18:59:45 2018

@author: priyank
"""

#!/usr/bin/python3

from tkinter import *
fields = 'Last Name', 'First Name', 'Job', 'Country'

def fetch(entries):
   for entry in entries:
      field = entry[0]
      text  = entry[1].get()
      print('%s: "%s"' % (field, text)) 

def makeform(root, fields):
   entries = []
   for field in fields:
      row = Frame(root)
      lab = Label(row, width=15, text=field, anchor='w')
      ent = Entry(row)
      row.pack(side=TOP, fill=X, padx=5, pady=5)
      lab.pack(side=LEFT)
      ent.pack(side=RIGHT, expand=YES, fill=X)
      entries.append((field, ent))
   #print(entries[1])
   return entries

if __name__ == '__main__':
   root = Tk()
   ents = makeform(root, fields)
   root.bind('<Return>', (lambda event, e=ents: fetch(e)))   
   b1 = Button(root, text='Show',
          command=(lambda e=ents: fetch(e)))
   b1.pack(side=LEFT, padx=5, pady=5)
   b2 = Button(root, text='Quit', command=root.quit)
   b2.pack(side=LEFT, padx=5, pady=5)
   root.mainloop()

#
#***************************************************************************
#from tkinter import *
#
#root = Tk()
#
#myImg = PhotoImage(file= "c-1000-01.png") 
#
#btn= Button(root, image=myImg)
#btn.pack()
#
#root.mainloop()

wid=Tk()
def myown():
 file = tkFileDialog.askopenfile(parent=wid,mode='rb',title='Choose a file')
 if file != None:
   data = file.read()
   b=cv2.imread(data)
   #cv2.imshow('img',b)
   #cv2.waitKey(0)

   file.close()


wid3=Button(None,text="upload it",command=myown)
wid.mainloop
myown()