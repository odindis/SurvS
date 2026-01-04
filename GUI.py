from tkinter import ttk, filedialog, messagebox 
from tkinter.scrolledtext import ScrolledText   

import multiprocessing 
import tkinter as tk  
import threading  # 多线程模块，训练过程放在后台线程
import queue      # 线程安全队列，用于传递日志
import train 


class App(tk.Tk): #_____________________________________________________________
    def __init__(self):
        super().__init__()  
        self.title("UI for SurvS ")  
        self.center_window()       
       
        self.log_queue_train = queue.Queue() 
        self.log_queue_test  = queue.Queue()
        
        self.stop_event = threading.Event()  # 控制训练中止
        self.train_thread = None             
        self.test_thread = None


        self.create_tabs()       # 创建 Tab 页

        self.after(100, self.update_log)  # 每100ms刷新一次日志

    # 窗口居中
    def center_window(self):
        self.update_idletasks()  
        w, h = 600, 600          
        x = (self.winfo_screenwidth() - w) // 2  
        y = (self.winfo_screenheight() - h) // 2  
        self.geometry(f"{w}x{h}+{x}+{y}")  

    # 创建 Tab 页
    def create_tabs(self):
        style = ttk.Style()

        style.configure("TNotebook.Tab",
                        padding=[15, 1],        # [左右，上下] padding
                        foreground="#737373")      # 默认文字颜色
        style.map("TNotebook.Tab", foreground=[("selected", "#000000")])  # 被选中字体

        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True, padx=5, pady=5)

        # 创建三个 Tab 页
        self.tab1 = ttk.Frame(notebook)
        self.tab2 = ttk.Frame(notebook)
        self.tab3 = ttk.Frame(notebook)

        notebook.add(self.tab1, text="模型构建")
        notebook.add(self.tab2, text="模型使用")
        notebook.add(self.tab3, text="帮助")

        self.create_tab1()
        self.create_tab2()
        self.create_tab3()

    # Spinbox 自动回归校验
    def wrap_value(self, var, min_val, max_val):
        try:
            v = int(var.get())
            if v < min_val:
                var.set(max_val)
            elif v > max_val:
                var.set(min_val)
        except Exception:
            var.set(min_val)

    # 滚动条
    def make_spinbox(self, parent, var, v_min, v_max):
        sb = tk.Spinbox(
            parent,
            from_=v_min,  # 最小值
            to=v_max,    # 最大值
            wrap=True, # 超出范围滚动回绕（按钮操作有效）
            textvariable=var,  # 绑定 IntVar
            width=8,
            validate="focusout" # 焦点离开时触发校验
        )
        # 监听变量变化，触发 wrap_value 校验（手动输入生效）
        var.trace_add("write", lambda *_: self.wrap_value(var,v_min,v_max))
        return sb

    # Tab 1
    def create_tab1(self):
        f = self.tab1  
        f.columnconfigure(1, weight=1) # 设置第1列可水平拉伸，占据剩余空间
        f.rowconfigure(7, weight=1) # 设置第7行可垂直拉伸，占据剩余空间

        # 输入区域 --------------------------------------------------------------
        csv_frame = ttk.LabelFrame(f, text="导入文件")
        csv_frame.grid(row=0, column=0, columnspan=2, rowspan=3, sticky="nsew", padx=5, pady=10)

        # 按钮区域
        btns = ttk.Frame(csv_frame)
        btns.grid(row=0, column=0, sticky="nw", padx=5, pady=5)
        ttk.Button(btns, text="添加文件", command=self.add_train_files).pack(side="top", fill="x", pady=2)
        ttk.Button(btns, text="删除选中", command=lambda: self.remove_selected(self.train_files)).pack(side="top", fill="x", pady=2)

        # 文件列表 + 滚动条
        list_frame = ttk.Frame(csv_frame)
        list_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)

        self.train_files = tk.Listbox(list_frame, height=5)
        self.train_files.pack(side="left", fill="both", expand=True)

        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=self.train_files.yview)
        scrollbar.pack(side="right", fill="y")

        self.train_files.configure(yscrollcommand=scrollbar.set)

        # 设置 CSV 框内部自适应
        csv_frame.columnconfigure(1, weight=1)  # 文件列表列可横向拉伸
        csv_frame.rowconfigure(0, weight=1)     # 文件列表行可纵向拉伸


        # 参数区域 --------------------------------------------------------------
        param_frame = ttk.LabelFrame(f, text="参数设置")  
        param_frame.grid(row=3, column=0, columnspan=2, sticky="nsew", padx=5, pady=5)

        # 在 param_frame 内添加一个空白列，提高左边距离
        param_frame.columnconfigure(0, minsize=10)   
        param_frame.columnconfigure(1, weight=0)     
        param_frame.columnconfigure(2, weight=0)    
        param_frame.columnconfigure(3, weight=0)   
        param_frame.columnconfigure(4, weight=0)   
        param_frame.columnconfigure(5, weight=0)    
        param_frame.columnconfigure(6, weight=0)  
        param_frame.columnconfigure(7, weight=0)    
        param_frame.columnconfigure(8, weight=0)     


        # 参数
        r, c = 0, 1
        ttk.Label(param_frame, text="疗效组数量:").grid(row=r, column=c, sticky="e", pady=5, padx=5)
        self.p_groups = tk.IntVar(value=2)
        self.make_spinbox(param_frame, self.p_groups, 2, 3 ).grid(row=r, column=c+1, sticky="w", pady=5, padx=5)

        # --------------
        r, c = 0, 3
        ttk.Label(param_frame, text="最少人数:").grid(row=r, column=c, sticky="e", pady=5, padx=5)
        self.p_size_min = tk.IntVar(value=10)
        self.make_spinbox(param_frame, self.p_size_min, 10, 10000000 ).grid(row=r, column=c+1, sticky="w", pady=5, padx=5)

        r, c = 0, 5
        ttk.Label(param_frame, text="最多因素数量:").grid(row=r, column=c, sticky="e", pady=5, padx=5)
        self.p_range_nz_wg = tk.IntVar(value=10)
        self.make_spinbox(param_frame, self.p_range_nz_wg, 1, 10000000 ).grid(row=r, column=c+1, sticky="w", pady=5, padx=5)

        # --------------
        r, c = 1, 1
        ttk.Label(param_frame, text="种群数量:").grid(row=r, column=c, sticky="e", pady=5, padx=5)
        self.p_num_max_inds = tk.IntVar(value=500)
        self.make_spinbox(param_frame, self.p_num_max_inds, 50, 10000000 ).grid(row=r, column=c+1, sticky="w", pady=5, padx=5)

        r, c = 1, 3
        ttk.Label(param_frame, text="淘汰比例%:").grid(row=r, column=c, sticky="e", pady=5, padx=5)
        self.p_survivors = tk.IntVar(value=20)
        self.make_spinbox(param_frame, self.p_survivors, 0, 100 ).grid(row=r, column=c+1, sticky="w", pady=5, padx=5)

        r, c = 1, 5
        ttk.Label(param_frame, text="迭代次数:").grid(row=r, column=c, sticky="e", pady=5, padx=5)
        self.p_epoch = tk.IntVar(value=5000)
        self.make_spinbox(param_frame, self.p_epoch, 1, 10000000 ).grid(row=r, column=c+1, sticky="w", pady=5, padx=5)

        # --------------
        r, c = 2, 1
        ttk.Label(param_frame, text="父母概率%:").grid(row=r, column=c, sticky="e", pady=5, padx=5)
        self.p_prob_parent = tk.IntVar(value=80)
        self.make_spinbox(param_frame, self.p_prob_parent, 1, 100 ).grid(row=r, column=c+1, sticky="w", pady=5, padx=5)

        r, c = 2, 3
        ttk.Label(param_frame, text="变异概率%:").grid(row=r, column=c, sticky="e", pady=5, padx=5)
        self.p_prob_mutation = tk.IntVar(value=50)
        self.make_spinbox(param_frame, self.p_prob_mutation, 1, 100 ).grid(row=r, column=c+1, sticky="w", pady=5, padx=5)

        r, c = 2, 5
        ttk.Label(param_frame, text="交叉概率%:").grid(row=r, column=c, sticky="e", pady=5, padx=5)
        self.p_prob_crossover = tk.IntVar(value=50)
        self.make_spinbox(param_frame, self.p_prob_crossover, 1, 100 ).grid(row=r, column=c+1, sticky="w", pady=5, padx=5)

        # --------------
        r, c = 3, 1
        ttk.Label(param_frame, text="模型数:").grid(row=r, column=c, sticky="e", pady=5, padx=5)
        self.p_num_model= tk.IntVar(value=5)
        self.make_spinbox(param_frame, self.p_num_model, 1, 100000000 ).grid(row=r, column=c+1, sticky="w", pady=5, padx=5)

        r, c = 3, 3
        ttk.Label(param_frame, text="并行数:").grid(row=r, column=c, sticky="e", pady=5, padx=5)
        self.p_n_jobs= tk.IntVar(value=4)
        self.make_spinbox(param_frame, self.p_n_jobs, 1, 100 ).grid(row=r, column=c+1, sticky="w", pady=5, padx=5)



        ## 执行区域 ------------------------------------------------------------
        exec_frame = ttk.LabelFrame(f, text="程序执行")  # 创建一个带标题的大框
        exec_frame.grid(row=4, column=0, columnspan=2, sticky="nsew", padx=5, pady=5)

        actions = ttk.Frame(exec_frame)
        actions.grid(row=0, column=0, sticky="w", pady=5, padx=5)

        ttk.Button(actions, text="开始构建", command=self.start_train).pack(side="left", padx=5)
        ttk.Button(actions, text="停止迭代", command=self.stop_train).pack(side="left", padx=5)

        self.log_train = ScrolledText(exec_frame, height=12)
        self.log_train.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)

        exec_frame.columnconfigure(0, weight=1)  # 日志列可以水平拉伸
        exec_frame.rowconfigure(1, weight=1)     # 日志行可以垂直拉伸

    # Tab 2
    def create_tab2(self):
        f = self.tab2
        f.columnconfigure(1, weight=1) # 设置第1列可水平拉伸，占据剩余空间
        f.rowconfigure(7, weight=1) # 设置第7行可垂直拉伸，占据剩余空间

        # 输入区域 --------------------------------------------------------------
        csv_frame = ttk.LabelFrame(f, text="导入文件")
        csv_frame.grid(row=0, column=0, columnspan=2, rowspan=3, sticky="nsew", padx=5, pady=10)

        # 文件按钮
        btns = ttk.Frame(csv_frame)
        btns.grid(row=0, column=0, sticky="nw", padx=5, pady=5)
        ttk.Button(btns, text="添加文件", command=self.add_test_files).pack(side="top", fill="x", pady=2)
        ttk.Button(btns, text="删除选中", command=lambda: self.remove_selected(self.test_files)).pack(side="top", fill="x", pady=2)

        # 文件列表 + 滚动条
        list_frame = ttk.Frame(csv_frame)
        list_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)

        self.test_files = tk.Listbox(list_frame, height=6)
        self.test_files.pack(side="left", fill="both", expand=True)

        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=self.train_files.yview)
        scrollbar.pack(side="right", fill="y")

        self.test_files.configure(yscrollcommand=scrollbar.set)

        csv_frame.columnconfigure(1, weight=1)  # 文件列表列可横向拉伸
        csv_frame.rowconfigure(0, weight=1)     # 文件列表行可纵向拉伸


        # 模型选择按钮
        btns2 = ttk.Frame(csv_frame)
        btns2.grid(row=1, column=0, columnspan=2, sticky="nw", padx=5, pady=5)

        ttk.Button(btns2, text="选择模型", command=self.select_model).grid(row=0, column=0, padx=2, pady=2)

        self.model_path_text = tk.Text(btns2, height=1, bg="white", state="disabled")
        self.model_path_text.grid(row=0, column=1, sticky="ew", padx=5, pady=2)

        btns2.columnconfigure(1, weight=1)

  
        ## 执行区域 ------------------------------------------------------------
        exec_frame = ttk.LabelFrame(f, text="程序执行") 
        exec_frame.grid(row=4, column=0, columnspan=2, sticky="nsew", padx=5, pady=5)

        actions = ttk.Frame(exec_frame)
        actions.grid(row=0, column=0, sticky="w", pady=5, padx=5)

        ttk.Button(actions, text="运行", command=self.start_infer).pack(side="left", padx=5)

        self.log_test = ScrolledText(exec_frame, height=15)
        self.log_test.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)

        exec_frame.columnconfigure(0, weight=1)  # 日志列可以水平拉伸
        exec_frame.rowconfigure(1, weight=1)     # 日志行可以垂直拉伸

    # Tab 3
    def create_tab3(self):
        text = ScrolledText(self.tab3)
        text.pack(fill="both", expand=True)
    
        try:
            text.insert("end", open("manuel.txt", encoding="utf-8").read())
        except Exception:
            pass
    
        text.config(state="disabled")

    def add_train_files(self):
        files = filedialog.askopenfilenames( filetypes=[("CSV files", "*.csv")])

        # 取出当前 Listbox 中已有的文件
        existing = set(self.train_files.get(0, "end"))

        for f in files:
            if f not in existing:
                self.train_files.insert("end", f)

    def add_test_files(self):
        files = filedialog.askopenfilenames( filetypes=[("CSV files", "*.csv")])

        # 取出当前 Listbox 中已有的文件
        existing = set(self.test_files.get(0, "end"))

        for f in files:
            if f not in existing:
                self.test_files.insert("end", f)


    def remove_selected(self, listbox):
        for i in reversed(listbox.curselection()):  # 反向删除选中项，防止索引错位
            listbox.delete(i)

    def select_model(self):
        path = filedialog.askopenfilename(filetypes=[("模型文件", "*.*")])
        if path:
            self.model_path_text.config(state="normal")  # 允许编辑
            self.model_path_text.delete("1.0", "end")    # 清空
            self.model_path_text.insert("1.0", path)     # 插入
            self.model_path_text.config(state="disabled")# 不可编辑

    # 训练控制
    def start_train(self):
        if self.train_thread and self.train_thread.is_alive():
            messagebox.showwarning("提示", "训练正在进行中")
            return

        files = list( self.train_files.get(0, "end") )
        if not files:
            self.log_queue_train.put("> 请先选择 CSV 文件\n")
            return

        self.stop_event.clear()        # 清空中止标记
        # self.log_train.delete("1.0", "end")  # 清空日志框

        # 启动后台线程
        self.train_thread = threading.Thread(
            target = train.task_train,
            args   = ( files, self.p_groups.get(), self.p_size_min.get(), 
                       self.p_range_nz_wg.get(), self.p_num_max_inds.get(), 
                       self.p_survivors.get(), self.p_epoch.get(), 
                       self.p_prob_parent.get(), self.p_prob_mutation.get(), 
                       self.p_prob_crossover.get(), self.p_num_model.get(), 
                       self.p_n_jobs.get(),
                       self.stop_event, self.log_queue_train
                     ),
            daemon = True
                                             )
        self.train_thread.start()

    def stop_train(self):
        self.stop_event.set()  # 设置中止标记

    # 长度限制
    def append_log(self, text_widget, msg, max_lines=200):
        text_widget.insert("end", msg)
        text_widget.see("end")

        # 获取当前总行数
        line_count = int(text_widget.index("end-1c").split(".")[0])

        # 超过最大行数，删除最早的行
        if line_count > max_lines:
            text_widget.delete("1.0", f"{line_count - max_lines + 1}.0")

    # 日志刷新
    def update_log(self):
        try:
            while True:
                msg = self.log_queue_train.get_nowait()
                self.append_log(self.log_train, msg)
        except queue.Empty:
            pass
    
        try:
            while True:
                msg = self.log_queue_test.get_nowait()
                self.append_log(self.log_test, msg)
        except queue.Empty:
            pass
    
        self.after(100, self.update_log)

    def start_infer(self):
        if self.test_thread and self.test_thread.is_alive():
            messagebox.showwarning("提示", "进程占用，请等待")
            return
    
        files = list(self.test_files.get(0, "end"))
        if not files:
            messagebox.showwarning("提示", "请先选择 CSV 文件")
            return
    
        # self.log_test.delete("1.0", "end")
    
        self.test_thread = threading.Thread(
            target=train.task_infer,
            args=( files, 
                   self.model_path_text.get("1.0", "end").strip(),
                   self.log_queue_test
                 ),  
            daemon=True
                                           )
        self.test_thread.start()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    App().mainloop()
