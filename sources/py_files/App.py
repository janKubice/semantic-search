from tkinter import *
from searching import Search
import json
import pandas as pd

search = None
labels_ids = None

docs = None

#TODO udělat načtení dotazů ze souboru


def search_button_clicked():
    """
    Reakce na stisk vyhledávacího tlačítka
    """

    results = search.ranking_ir(search_text.get(),10)
    labels_ids = [None] * 10

    for i,j in results.iterrows():
        labels_ids[i] = Label(win, text=str(j['id']))
        labels_ids[i].grid(column=0, row=3+i)

        labels_ids[i].bind("<Button-1>",document_clicked)

        document = docs.loc[docs['id']==j['id']]
        label = Label(win, text=str(document['title']).split()[:6])
        label.grid(column=1, row=3+i)
        
def document_clicked(event):
    caller = event.widget
    document_id = caller.cget("text")
    
    document = docs.loc[docs['id']==document_id]

    document_win = Toplevel()
    scrollbar = Scrollbar(document_win)
    scrollbar.pack(side = RIGHT, fill = Y)

    document_win.title(document['id'])
    document_win.geometry('500x600')

    document_text = Label(document_win, text=document['text'], wraplength=450).pack()

win = Tk()
win.title('Document search')
win.geometry('500x600')

search_label = Label(win, text="Search")
search_label.grid(column=0, row=0)

search_text = Entry(win, width=30,textvariable='yo')
search_text.grid(column=0, row=1)

search_button = Button(win, text="Search", command=search_button_clicked)
search_button.grid(column=3, row=1)

with open("semantic-search/BP_data/czechData_test.json", encoding="utf8") as f:
    docs = json.load(f)

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)


docs = pd.DataFrame(docs)

search = Search(True,'semantic-search/BP_data/czechData_test.json')


win.mainloop()