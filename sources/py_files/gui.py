from tkinter import *
from sources.py_files.model_search import ModelSearch
import pandas as pd
import json

class Gui:

    def __init__(self, model:ModelSearch, document_path:str) -> None:
        self.model = model

        self.win = Tk()
        self.win.title('Document search')
        self.win.geometry('500x600')

        self.search_label = Label(self.win, text="Search")
        self.search_label.grid(column=0, row=0)

        self.search_text = Entry(self.win, width=30,textvariable='text')
        self.search_text.grid(column=0, row=1)

        self.search_button = Button(self.win, text="Search", command=self.search_button_clicked)
        self.search_button.grid(column=3, row=1)

        with open(document_path, encoding="utf8") as f:
            self.docs = json.load(f)

        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', None)


        self.docs = pd.DataFrame(self.docs)
        self.win.mainloop()

    def search_button_clicked(self):
        """
        Reakce na stisk vyhledávacího tlačítka
        """

        results = self.model.ranking_ir(self.search_text.get(),10)
        labels_ids = [None] * 10

        for i,j in results.iterrows():
            labels_ids[i] = Label(self.win, text=str(j['id']))
            labels_ids[i].grid(column=0, row=3+i)

            labels_ids[i].bind("<Button-1>",self.document_clicked)

            document = self.docs.loc[self.docs['id']==j['id']]
            label = Label(self.win, text=str(document['title']).split()[:6])
            label.grid(column=1, row=3+i)
        
    def document_clicked(self, event):
        """
        Reakce na kliknutí na ID dokumentu
        """
        caller = event.widget
        document_id = caller.cget("text")
        
        document = self.docs.loc[self.docs['id']==document_id]

        document_win = Toplevel()
        scrollbar = Scrollbar(document_win)
        scrollbar.pack(side = RIGHT, fill = Y)

        document_win.title(document['id'])
        document_win.geometry('500x600')

        document_text = Label(document_win, text=document['text'], wraplength=450).pack()