import tkinter as tk
import time
import threading


def main():
    top = tk.Tk()
    tex = tk.Text(master=top)
    tex.pack(side=tk.TOP)
    bop = tk.Frame()
    bop.pack(side=tk.BOTTOM,fill=tk.X)
    b = tk.Button(bop, text='Visual', command=cbc("hey adfa", tex))
    b.pack(side=tk.LEFT,fill=tk.X)
    b = tk.Button(bop, text='Columns', command=cbc("ho adfa", tex))
    b.pack(side=tk.LEFT)    
    
    tk.Button(bop, text='Exit', command=top.destroy).pack(side=tk.RIGHT)
    threadit(tex,dosomething,(tex,))
    top.mainloop()  

def cbc(string, tex):
    return lambda : callback(string, tex)


def callback(string, tex):
    tex.insert(tk.END, string)
    tex.see(tk.END)             # Scroll if necessary
    
    
def threadit(tex,func,args):
    t = threading.Thread(target=func,args=args)
    t.start()
    
def dosomething(tex):
    for i in range(20):
        time.sleep(5)
        tex.insert(tk.END, "id: %s" %(i,))



main()