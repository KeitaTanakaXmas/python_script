import openai
import tkinter



class Application(tkinter.Frame):
    def __init__(self,root=None):
        super().__init__(root,
            width=420,height=320,
            borderwidth=4,relief='groove'
            )
        self.root = root
        self.pack()
        self.pack_propagate(0)
        self.create_widgets()

    def create_widgets(self):
        quit_btn = tkinter.Button(self)
        quit_btn['text'] = 'QUIT'
        quit_btn['command'] = self.root.destroy
        quit_btn.pack(side='bottom')

    def create_txt(self):
        pass



class ChatGpt:
    def __init__(self):
        pass

    def do(self):
        res = openai.ChatCompletion.create(
            model='gpt-3.5-turbo',
            messages=[
                {'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': 'なぜ月は満ち欠けするのですか？'},
            ]
            )
        res_content = res['choices'][0]['message']['content']
        print(res_content)



root = tkinter.Tk()
root.title('Chat GPT')
root.geometry('400x300')
app = Application(root=root)
app.mainloop()
