import os
# import sys
now_path=os.path.dirname(os.path.realpath(__file__))
os.chdir(now_path)
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import torch
import re
from  typing  import List
import numpy as np
import csv
from scipy.stats import norm
import pandas as pd
import seaborn as sns

class MyPlot():

    def plot_vectors_defference(grad_one_vector,vector,rate):
        g_o_v_values=[]
        v_values=[]
        for i in range(len(grad_one_vector)):
            if torch.rand(1)>(1-rate):
                g_o_v_values.append(grad_one_vector[i].cpu().detach().numpy()[0])
                v_values.append(vector[i].cpu().detach().numpy()[0])
        x=[i for i in range(len(g_o_v_values))]
        plt.subplot(1,3,1)
        plt.title("original data")
        plt.plot(x,g_o_v_values)
        plt.subplot(1,3,2)
        plt.title("handled data")
        plt.plot(x,v_values)
        plt.subplot(1,3,3)
        plt.title("comparison")
        plt.plot(x,g_o_v_values)
        plt.plot(x,v_values,"r-")
        plt.show()
    
    def plot_comparas(long,names,all_data,titles,start=0,save_path=None):
        x=list(range(start,start+long))
        p=list(range(len(names)))

        for i in range(len(names)):
            p[i],=plt.plot(x,all_data[i])
        # plt.legend(['STC', 'CER+STC ($\gamma=0.1$)', 'CER+STC ($\gamma=1.0$)', 'CER+STC ($\gamma=10$)'],loc=4,ncol=2)
        plt.legend(names)
        # plt.legend(names,loc=0,ncol=2)
        # plt.ylim([400,1500])
        plt.xlabel(titles[0])
        plt.ylabel(titles[1])
        if save_path:
            plt.savefig(save_path)
        plt.show()
    
    def plot_hists(data_list,names,titles,bins=15,save_path=None):
        p=list(range(len(names)))
        data_dict={}
        x_max=0
        for index,name in enumerate(names):
            data_dict[name]=data_list[index]*100
            x_max=max(data_list[index]*100)
        data=pd.DataFrame(data_dict)

        sns.histplot(data,bins=bins,kde=True,common_norm=True)
        # plt.legend(['$\gamma=50$','$\gamma=100$','$\gamma=500$'],loc=[0.6,0.6])
        plt.legend(['$\gamma=50$','$\gamma=100$','$\gamma=500$'],loc=[0.6,0.6])
        plt.xlabel(titles[0])
        plt.ylabel(titles[1])
        if save_path:
            plt.savefig(save_path)
        plt.show()

            
    
    def compara(dir_path,type="acc",start=0,avg_num=50,
                is_plot=True,titles=["Number of round","y"],reranke="data",save_path=None):
        long,names,all_data=DataHandle.get_all_data(dir_path,type)
        plot_data=[]
        for data in all_data:
            single_data=[]
            for i in range(start,long-avg_num+1):
                single_data.append(sum(data[i:i+avg_num])/avg_num)
            plot_data.append(single_data)
        last_data=[data[-1] for data in plot_data]
        if reranke=="data":
            DataHandle.reranke(last_data,names,plot_data)
        elif reranke=="name":
            names,new_list=DataHandle.reranke_by_name(names,last_data,plot_data)
            last_data,plot_data=new_list[0],new_list[1]
        if is_plot:
            MyPlot.plot_comparas(len(single_data),names,plot_data,
                                 titles=titles,start=start+avg_num,save_path=save_path)
        return len(single_data),names,plot_data
        

    def plot_legend(name:List[str],y:List[List[float]])->None:
        x=range(len(y[0]))
        p=range(len(y))
        for i in y:
            p,=plt.plot(x,i)
        plt.legend(p,name)

    def plot_percent_data(dir_path,target_name,
                          type="client_size",titles=["Number of round","size proportion"],plot_type="hist",
                          start=0,avg_num=1,bins=15,save_path=None):
        long,names,all_data=DataHandle.get_all_data(dir_path,type)
        for index in range(len(names)):
            if names[index] == target_name:
                names.pop(index)
                target_data=all_data.pop(index)
                break
        data_list=DataHandle._get_percent(target_data,all_data)
        if plot_type=="line":
            MyPlot.plot_comparas(long,names,data_list,
                                    titles=titles,start=start+avg_num)
        elif plot_type == "hist":
            names,data_list=DataHandle.reranke_by_name(names,data_list)
            MyPlot.plot_hists(data_list[0],names,titles,bins,save_path)
        

            

class DataHandle():
    def get_all_data(dir_path,type="acc"):
        dirs=os.listdir(dir_path)
        names=[]
        all_data=[]
        for file in dirs:
            path=dir_path+file
            if os.path.isfile(path):
                name=re.findall("(.*).txt",file)
                names.append(name[0])
                with open(path,"r") as f:
                    contents=f.read()
                    if type=="acc":
                        date=re.findall("global accuracy: (\d+\.\d+)",contents)
                    elif type=="loss":
                        date=re.findall("global loss: (\d+\.\d+)",contents)
                    elif type=="bacc":
                        date=re.findall("global bacc: (\d+\.\d+)",contents)
                    elif type=="iou":
                        date=re.findall("global iou: (\d+\.\d+)",contents)
                    elif type=="label_iou":
                        date=re.findall("global label_iou: (\d+\.\d+)",contents)
                    elif type=="server_size":
                        date=re.findall("server size: (\d+\.\d+)",contents)
                    elif type=="client_size":
                        date=re.findall("client size: (\d+\.\d+)",contents)
                    elif type=="client_rate":
                        date=re.findall("client compress rate: (\d+\.\d+)",contents)
                    elif type=="server_rate":
                        date=re.findall("server compress rate: (\d+\.\d+)",contents)
                    data=[float(i) for i in date]
                    all_data.append(data)
        return len(data),names,all_data
    
    def get_dirs_values(data_dir:str,num:int,type:str)->dict:
        dir_path_list=[]
        for path in os.listdir(data_dir):
            if os.path.isdir(data_dir+path+"/"):
                dir_path_list.append(data_dir+path+"/")
        data_dict={}
        for dir_index,single_dir in enumerate(dir_path_list):
            data_long,names,all_data=DataHandle.get_all_data(single_dir,type)
            for name_index,name in enumerate(names):
                if dir_index == 0:
                    data_dict[name]=all_data[name_index][data_long-num:]
                else:
                    data_dict[name]=all_data[name_index][data_long-num:]+data_dict[name]
        return data_dict
    
    def get_fiels_values(data_dir:str,num:int,type:str)->dict:
        data_long,names,all_data=DataHandle.get_all_data(data_dir,type)
        data_dict={}
        for name_index,name in enumerate(names):
            data_dict[name]=all_data[name_index][data_long-num:]
        return data_dict



    def reranke(data_list,*orther_list):
        long=len(data_list)
        for i in range(long):
            for j in range(i+1,long):
                a=data_list[i]
                if a<data_list[j]:
                    data_list[i]=data_list[j]
                    data_list[j]=a
                    a=data_list[j]
                    for arg in orther_list:
                        b=arg[i]
                        arg[i]=arg[j]
                        arg[j]=b
                        b=arg[j]
        return
    
    def reranke_by_name(name_list,*orther_list):
        long=len(name_list)
        droup=[]
        droup_other_list=[]
        count=0
        while count<long:
            x=re.findall("(\d+.?\d*)",name_list[count])
            if not x:
                droup.append(name_list.pop(count))
                for arg in orther_list:
                    if type(arg[count])==list:
                        droup_other_list.append(arg.pop(count))
                    else:
                        droup_other_list.append(arg.pop(count))
                count-=1
                long-=1
            count+=1
        for i in range(long):
            for j in range(i+1,long):
                a=name_list[i]
                b=re.findall("(\d+.?\d*)",a)
                c=re.findall("(\d+.?\d*)",name_list[j])
                if b and c:
                    if float(b[0])<float(c[0]):
                        name_list[i]=name_list[j]
                        name_list[j]=a
                        a=name_list[j]
                        for arg in orther_list:
                            b=arg[i]
                            arg[i]=arg[j]
                            arg[j]=b
                            b=arg[j]
        if droup:
            name_list=name_list+droup
            for index in range(len(orther_list)):
                orther_list[index].append(droup_other_list[index])
                # new_other_list.append(orther_list[index].append(droup_other_list[index]))
        return name_list,orther_list
    
    def _get_percent(target_data,otherdata_list):
        target_data=torch.tensor(target_data)
        data_list=[]
        for arg_index in range(len(otherdata_list)):
            arg=torch.tensor(otherdata_list[arg_index])
            data_list.append(((target_data-arg)/target_data).cpu().detach().numpy())
        return data_list
    
    


class DirsHandle():
    def select_need_files(dir_path,rename_list,useful_name_list,class_name_list):
        for class_name in class_name_list:
            if not os.path.exists(dir_path+"/"+class_name):
                os.makedirs(dir_path+"/"+class_name)
        for files_name in os.listdir(dir_path):
            for class_name in class_name_list:
                class_path= dir_path+"/"+class_name
                if class_name in files_name:
                    for rename_list_index in range(len(rename_list)):
                        if useful_name_list[rename_list_index] in files_name:
                            os.system("cp "+dir_path+"/"+files_name+" "+class_path)
                            os.rename(class_path+"/"+files_name,class_path+"/"+rename_list[rename_list_index]+".txt")
    
    def _get_row_list(data_dir,names,title,target_name_list,num=3):
        x=np.zeros((num+1+len(target_name_list),len(names)+1))
        row_list=[]
        for i in range(x.shape[0]):
            row_list.append(list(x[i,:]))
        row_list[0][0]=title
        for i in range(num):
            row_list[i+1][0]=" "
        for i in range(len(target_name_list)):
            row_list[num+1+i][0]=target_name_list[i]
        return row_list
    
    def _get_mean(data):
        if type(data)==np.ndarray or type(data)==torch.Tensor:
            return data.mean()
        elif type(data)==list:
            new_data=torch.tensor(data).mean()
            return new_data.cpu().detach().numpy()

    def _get_std(data):
        if type(data)==np.ndarray or type(data)==torch.Tensor:
            return data.std()
        elif type(data)==list:
            new_data=torch.tensor(data).std()
            return new_data.cpu().detach().numpy()
        
    def _get_dir_names(dir_path):
        names=[]
        for file_name in os.listdir(dir_path):
            if not os.path.isdir(file_name):
                names=names+re.findall("(.*).txt",file_name)
        return names

    
    def get_mean_std_files(data_dir,target_name_list,fn_list,
                           type="acc",num=3,name_seq=None):
        if name_seq:
            names=name_seq
        else:
            names=DirsHandle._get_dir_names(data_dir)
        row_list=DirsHandle._get_row_list(data_dir,names,type,target_name_list,num)
        
        data_dict=DataHandle.get_fiels_values(data_dir,num,type)
        for name_index in range(len(names)):
            row_list[0][name_index+1]=names[name_index]
            for i in range(num):
                row_list[i+1][name_index+1]=data_dict[names[name_index]][num-i-1]
            for i in range(len(target_name_list)):
                row_list[num+1+i][name_index+1]=fn_list[i](data_dict[names[name_index]])
        with open(data_dir+"parameters_{}.csv".format(type),"w") as f:
            writer=csv.writer(f)
            for row in row_list:
                writer.writerow(row)

    def split_txt2txts_by_linenum(file_path,save_name_list,line_num):
        dir_path=re.findall("(.*/).+txt",file_path)
        count=0
        name_index=0
        file_content=[]
        with open(file_path,"r") as f:
            for line in f.readlines():
                file_content.append(line)
                count+=1
                if count%line_num == 0:
                    with open(dir_path[0]+save_name_list[name_index]+".txt","w") as wf:
                        for i in file_content:
                            wf.writelines(i)
                    if int(count/line_num)==len(save_name_list):
                        break
                    file_content=[]
                    name_index+=1



if __name__ == "__main__":
    # file_path="/home/yawei/Documents/berifen/useful_data/communicate/gama/hinde/federated_log_classification_DenseNet_20230504163241_log.txt"
    # gama_list=[0,0.001,0.1,1,10]
    # save_name_list=["gama="+str(i) for i in gama_list]
    # # save_name_list=["stc1"]
    # DirsHandle.split_txt2txts_by_linenum(file_path,save_name_list,200)

    ################################################################################################
    # namda_dir="/home/yawei/Documents/berifen/useful_data/namda/all_namda_data"
    # dir_path="/home/yawei/Documents/berifen/useful_data/namda/unet_compara_namda/all/"
    # rename_list=[str(i) for i in range(1,9)]
    # # name_seq=["FedRoD","Ditto","FedAvg","FedPer","Fedprox","FPFC","IFCA","FedMGL","pFedME","pFedNet","SuPerFed","FedAMP"]
    # useful_name_list=["_0","_9","_13","_17","_21","_27","_29","_32"]
    # class_name_list=["balance","unbalance_2","unbalance_4","unbalance_7","lack0","lack1","lack2","lack3"]
    # class_name_list=["lack0","lack1","lack2","lack3"]
    # DirsHandle.select_need_files(dir_path,rename_list,useful_name_list,class_name_list)

    ################################################################################################
    data_dir="//home/yawei/Documents/liuqinghe/useful_data/communicate/gama_densenet/data"
    data_dir=data_dir+"/"
    DirsHandle._get_dir_names(data_dir)
    num=3
    # # name_seq=[str(i) for i in range(1,9)]
    # name_seq=["FedRoD","Ditto","FedAvg","FedPer","Fedprox","FPFC","IFCA","FedMGL","pFedME","pFedNet","SuPerFed","FedAMP"]
    name_seq=["stc","gama=0.1","gama=1","gama=10"]
    target_name_list=["mean","std"]
    fn_list=[DirsHandle._get_mean,DirsHandle._get_std]
    # rename_list=[str(i) for i in range(1,9)]
    # # try:
    DirsHandle.get_mean_std_files(data_dir,target_name_list,fn_list,
                                    type="server_size",num=3,name_seq=None)
    # except:
    #     DirsHandle.get_mean_std_files(data_dir,target_name_list,fn_list,
    #                                     type="iou",num=3,name_seq=None)
    #     DirsHandle.get_mean_std_files(data_dir,target_name_list,fn_list,
    #                                     type="label_iou",num=3,name_seq=None)
    ################################################################################################

    plot_dir="/home/yawei/Documents/liuqinghe/useful_data/communicate/gama_unet"
    # plot_dir="/home/yawei/Documents/berifen/useful_data/acc_iou/densenet/deta=2/data"
    plot_dir=plot_dir+"/"
    save_path="/home/yawei/Documents/berifen/useful_data/communicate/picture/unet/unet_conmunicate_iou_line.pdf"
    # save_path="/home/yawei/Documents/berifen/useful_data/communicate/picture/unet/unet_conmunicate_server.pdf"
    # data_long,names,all_data=MyPlot.get_all_data(plot_dir,"acc")

    # MyPlot.compara(plot_dir,start=0,type="client_rate",avg_num=99,
    #                titles=["Number of rounds","loss"],reranke="data",save_path=None)
    # MyPlot.plot_percent_data(plot_dir,target_name="stc",type="server_size",titles=["Improvement of communication efficiency (%)","Frequency (%)"],
    #                          start=0,avg_num=1,bins=50,save_path=save_path)
    ################################################################################################

 








