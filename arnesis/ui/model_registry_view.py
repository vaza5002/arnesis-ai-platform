"""Arnesis Model Registry desktop workspace."""
from __future__ import annotations
import tkinter as tk
from tkinter import filedialog,ttk
from arnesis.ui.theme import ArnesisTheme
from arnesis.ui.ux_messages import DialogService,MessageLevel,UserMessage

class ModelRegistryView(tk.Frame):
    """CRUD view for verified local model definitions and CUDA validation."""
    def __init__(self,parent,service):
        c=ArnesisTheme.colors; super().__init__(parent,bg=c.background); self.service=service; self.selected_id=None
        self.vars={k:tk.StringVar() for k in ("name","version","type","framework","path","size","notes","gpu")}; self.vars["type"].set("DETECTION"); self.vars["framework"].set("Ultralytics"); self.vars["gpu"].set("0")
        self._build(); self.refresh()
    def _build(self):
        c=ArnesisTheme.colors; left=tk.Frame(self,bg=c.surface,padx=18,pady=18); left.pack(side="left",fill="y",padx=(0,10)); right=tk.Frame(self,bg=c.surface,padx=18,pady=18); right.pack(side="left",fill="both",expand=True)
        tk.Label(left,text="MODEL DEFINITION",bg=c.surface,fg=c.accent,font=(ArnesisTheme.font_family,14,"bold")).pack(anchor="w",pady=(0,12))
        for label,key in (("Name","name"),("Version","version"),("Framework","framework"),("Input size","size"),("Notes","notes")):
            tk.Label(left,text=label,bg=c.surface,fg=c.text,anchor="w").pack(fill="x"); ttk.Entry(left,textvariable=self.vars[key],style="Arnesis.TEntry").pack(fill="x",pady=(3,8))
        tk.Label(left,text="Model type",bg=c.surface,fg=c.text,anchor="w").pack(fill="x"); ttk.Combobox(left,textvariable=self.vars["type"],values=("DETECTION","CLASSIFICATION","POSE"),state="readonly",style="Arnesis.TCombobox").pack(fill="x",pady=(3,8))
        tk.Label(left,text="Model path",bg=c.surface,fg=c.text,anchor="w").pack(fill="x"); ttk.Entry(left,textvariable=self.vars["path"],style="Arnesis.TEntry").pack(fill="x",pady=(3,4)); ArnesisTheme.button(left,text="Browse...",command=self._browse).pack(fill="x",pady=(0,10))
        row=tk.Frame(left,bg=c.surface); row.pack(fill="x"); ArnesisTheme.button(row,text="New",command=self._clear).pack(side="left",expand=True,fill="x",padx=(0,3)); ArnesisTheme.button(row,text="Save",command=self._save,variant="primary").pack(side="left",expand=True,fill="x",padx=3); ArnesisTheme.button(row,text="Delete",command=self._delete,variant="danger").pack(side="left",expand=True,fill="x",padx=(3,0))
        tk.Label(left,text="CUDA device index",bg=c.surface,fg=c.text,anchor="w").pack(fill="x",pady=(16,0)); ttk.Entry(left,textvariable=self.vars["gpu"],style="Arnesis.TEntry").pack(fill="x",pady=(3,6)); ArnesisTheme.button(left,text="Validate CUDA Load",command=self._validate_cuda,variant="success").pack(fill="x")
        tk.Label(right,text="REGISTERED MODELS",bg=c.surface,fg=c.accent,font=(ArnesisTheme.font_family,14,"bold")).pack(anchor="w",pady=(0,12))
        self.tree=ttk.Treeview(right,columns=("name","version","type","framework","path","enabled"),show="headings",style="Arnesis.Treeview")
        for col,title,width in (("name","Name",150),("version","Version",80),("type","Type",110),("framework","Framework",110),("path","Verified path",400),("enabled","Enabled",70)): self.tree.heading(col,text=title); self.tree.column(col,width=width,anchor="w")
        self.tree.pack(fill="both",expand=True); self.tree.bind("<<TreeviewSelect>>",self._select)
    def refresh(self):
        self.tree.delete(*self.tree.get_children())
        for m in self.service.list_models(): self.tree.insert("","end",iid=str(m.id),values=(m.name,m.version,m.model_type,m.framework,m.model_path,"Yes" if m.enabled else "No"))
    def _browse(self):
        path=filedialog.askopenfilename(filetypes=(("Model files","*.pt *.pth *.onnx *.engine *.torchscript"),("All files","*.*")))
        if path:self.vars["path"].set(path)
    def _save(self):
        try:
            self.service.save(model_id=self.selected_id,name=self.vars["name"].get(),version=self.vars["version"].get(),model_type=self.vars["type"].get(),framework=self.vars["framework"].get(),model_path=self.vars["path"].get(),input_size=int(self.vars["size"].get()) if self.vars["size"].get().strip() else None,notes=self.vars["notes"].get() or None); self.refresh(); self._clear()
        except Exception as e:self._error("Unable to save model",e)
    def _delete(self):
        if self.selected_id is None:return
        try:self.service.delete(self.selected_id); self.refresh(); self._clear()
        except Exception as e:self._error("Unable to delete model",e)
    def _validate_cuda(self):
        if self.selected_id is None:return self._error("Select a saved model first")
        try:
            result=self.service.validate_cuda_load(self.selected_id,int(self.vars["gpu"].get())); DialogService.show(self,UserMessage(MessageLevel.INFO,"CUDA validation passed",f"Model loaded on {result['cuda_device']}.",result["parameter_device"]))
        except Exception as e:self._error("CUDA model validation failed",e)
    def _select(self,_=None):
        selected=self.tree.selection()
        if not selected:return
        self.selected_id=int(selected[0]); m=self.service.get(self.selected_id)
        for key,value in (("name",m.name),("version",m.version),("type",m.model_type),("framework",m.framework),("path",m.model_path),("size",m.input_size or ""),("notes",m.notes or "")):self.vars[key].set(value)
    def _clear(self):
        self.selected_id=None
        for key in ("name","version","path","size","notes"):self.vars[key].set("")
        self.vars["type"].set("DETECTION"); self.vars["framework"].set("Ultralytics")
    def _error(self,title,error=None):DialogService.show(self,UserMessage(MessageLevel.ERROR,title,"The requested model operation could not be completed.",str(error) if error else None))
