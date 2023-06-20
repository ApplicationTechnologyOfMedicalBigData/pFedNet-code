import os
import torch
import nibabel as nib
import numpy as np
from tqdm import tqdm
import re
import torchvision.transforms as transforms
from torchvision.utils import save_image
from PIL import Image
from UNet import UNet as Net
DEVICE="cuda:0"



class CutClients():
    def __init__(self,client_dir,path,client_num=None) -> None:
        os.system("rm -rf "+client_dir)
        if not os.path.exists(client_dir):
            os.makedirs(client_dir)
        self.client_dir=client_dir+"/"
        self.path=path+"/"
        self.client_num=client_num
    def walk_dirs(self):
        image_path=self.path+"images/"
        label_path=self.path+"labels/"
        image_list=os.listdir(image_path)
        label_list=os.listdir(label_path)
        return image_path,image_list,label_path,label_list
    
    def _get_limit_num(self,image_list,label_list,long):
        client_label_dict={}
        client_image_dict={}
        for i in range(self.client_num):
            client_image_dict[i]=[]
            client_label_dict[i]=[]
        start=0
        while start<long:
            for client in client_image_dict:
                client_image_dict[client].append(image_list[start])
                client_label_dict[client].append(label_list[start])
                start=start+1
                if start>=len(image_list):
                    break
        return client_label_dict,client_image_dict
    
    def make_balance_clients_dirs(self,long):
        channal_path_list=[self.path+path for path in os.listdir(self.path)]
        client_label_dict,client_image_dict={},{}
        single_long=int(long*self.client_num/len(channal_path_list))
        init_flag=1
        for channal_path in channal_path_list:
            image_path=channal_path+"/images/"
            label_path=channal_path+"/labels/"
            image_path_list=[image_path+path for path in os.listdir(image_path)]
            label_path_list=[label_path+path for path in os.listdir(label_path)]
            single_client_label_dict,single_client_image_dict=self._get_limit_num(image_path_list,label_path_list,single_long)
            if init_flag:
                client_image_dict=single_client_image_dict
                client_label_dict=single_client_label_dict

                init_flag=0
            else:
                for name in single_client_image_dict:
                    client_image_dict[name]=client_image_dict[name]+single_client_image_dict[name]
                    client_label_dict[name]=client_label_dict[name]+single_client_label_dict[name]
        for i in tqdm(range(self.client_num)):
            client_image_path=self.client_dir+"client"+str(i)+"/"+"images/"
            client_label_path=self.client_dir+"client"+str(i)+"/"+"labels/"
            if not os.path.exists(client_image_path):
                os.makedirs(client_image_path)
            if not os.path.exists(client_label_path):
                os.makedirs(client_label_path)
            for image in client_image_dict[i]:
                os.system("cp "+image+" "+client_image_path)
            for label in client_label_dict[i]:
                os.system("cp "+label+" "+client_label_path)
    
    def _get_num_from_config(self,client_proper_list):
        index_class_list=os.listdir(self.path)

        client_label_dict={}
        client_image_dict={}
        for i in client_proper_list:
            client_image_dict[i[0]]=[]
            client_label_dict[i[0]]=[]
        
        for index in index_class_list:
            image_dir=self.path+"/"+index+"/images/"
            label_dir=self.path+"/"+index+"/labels/"
            oneclass_imagepath_list=[image_dir+path for path in os.listdir(image_dir)]
            oneclass_labelpath_list=[label_dir+path for path in os.listdir(label_dir)]
            start=0
            for line in client_proper_list:
                num_class=int(line[int(index)+1])
                client_image_dict[line[0]]=client_image_dict[line[0]]+oneclass_imagepath_list[start:start+num_class]
                client_label_dict[line[0]]=client_label_dict[line[0]]+oneclass_labelpath_list[start:start+num_class]
                start=start+num_class
        return client_label_dict,client_image_dict
    
    def make_clients_dirs_by_config(self,config_path):
        with open(config_path,"r") as f:
            init_flag=1
            client_proper_list=[]
            for line in f.readlines():
                if init_flag:
                    init_flag=0
                else:
                    client_proper_list.append(re.findall("([\d]+),?",line))
        print(client_proper_list)
        client_label_dict,client_image_dict=self._get_num_from_config(client_proper_list)

        for name in tqdm(client_label_dict):
            client_image_path=self.client_dir+"client"+name+"/"+"images/"
            client_label_path=self.client_dir+"client"+name+"/"+"labels/"
            if not os.path.exists(client_image_path):
                os.makedirs(client_image_path)
            if not os.path.exists(client_label_path):
                os.makedirs(client_label_path)
            for image in client_image_dict[name]:
                os.system("cp "+image+" "+client_image_path)
            for label in client_label_dict[name]:
                os.system("cp "+label+" "+client_label_path)


    
    def make_client_dirs(self,long):
        image_path,image_list,label_path,label_list=self.walk_dirs()
        client_label_dict,client_image_dict=self._get_limit_num(image_list,label_list,long)
            
        for i in range(self.client_num):
            client_image_path=self.client_dir+"client"+str(i)+"/"+"images/"
            client_label_path=self.client_dir+"client"+str(i)+"/"+"labels/"
            if not os.path.exists(client_image_path):
                os.makedirs(client_image_path)
            if not os.path.exists(client_label_path):
                os.makedirs(client_label_path)
            for picture_path in client_image_dict[i]:
                os.system("cp "+image_path+picture_path+" "+client_image_path)
                os.system("cp "+label_path+picture_path+" "+client_label_path)
    
class PrehandleData():
    def _cut(data,span,pos):
        pos_range=[]
        dim_index=0
        if type(span)!=list:
            span=[span]*len(pos)
        for i in range(len(data.shape)):
            if (pos[i]+span[i])>data.shape[i]:
                x2=data.shape[i]
                x1=x2-2*span[i]
            elif (pos[i]-span[i])<0:
                x1=0
                x2=span[i]*2
            else:
                x1=pos[i]-span[i]
                x2=pos[i]+span[i]
            if span[i]!=0:
                pos_range=list(range(x1,x2))
            else:
                pos_range=list(range(x1-1,x2+2))
                dim_index=i

            pos_range=torch.tensor(pos_range,device=DEVICE)
            data=torch.index_select(data,dim=i,index=pos_range)
        for x in range(dim_index):
            # data=torch.transpose(data,dim_index-x,dim_index-x+1)
            data=data.transpose(dim_index-x,dim_index-x-1)
        data=data.contiguous()
        # print(PrehandleData.label_property(data))


        return data.cpu().detach().numpy()
    
    def label_property(data):
        return torch.sum(torch.clamp(data,max=1))/data.numel()
    
        
    def cut_label_nii(label_path,span=25):
        data=nib.load(label_path)
        af=data.affine
        data=data.get_fdata().astype(np.float32)
        # nib.viewers.OrthoSlicer3D(data).show()
        # ones_data=torch.ones(data.shape,device=DEVICE)
        data = torch.tensor(data,dtype=float,device=DEVICE)
        data=torch.clamp(data,max=1)
        x=torch.sum(torch.sum(data,dim=1),dim=1).argmax()
        y=torch.sum(torch.sum(data,dim=0),dim=1).argmax()
        z=torch.sum(torch.sum(data,dim=0),dim=0).argmax()
        cut_data=PrehandleData._cut(data,span,(x,y,z))
        # nib.viewers.OrthoSlicer3D(cut_data.cpu().detach().numpy()).show()

        return cut_data,af,span,(x,y,z)
    
    def cut_image_nii(image_path,span,pos):
        data=nib.load(image_path)
        af=data.affine
        data = data.get_fdata().astype(np.float32)
        data = torch.tensor(data,dtype=float,device=DEVICE)
        image_shape=data.shape
        # nib.viewers.OrthoSlicer3D(data).show()
        cut_data_list=[]
        for i in range(image_shape[3]):
            single_data=data[:,:,:,i]
            cut_data_list.append(PrehandleData._cut(single_data,span,pos))
        return cut_data_list,af

    def save_cut_nii(label_path,image_path,save_path,file_name,span=25):
        label_data,label_af,span,(x,y,z)=PrehandleData.cut_label_nii(label_path,span=span)
        image_data_list,image_af=PrehandleData.cut_image_nii(image_path,span,(x,y,z))
        # nib.viewers.OrthoSlicer3D(data).show()
        
        label_data=nib.Nifti1Image(label_data,label_af)
        for i in range(len(image_data_list)):
            save_label_path=save_path+str(i)+"/labels/"
            if not os.path.exists(save_label_path):
                os.makedirs(save_label_path)
            nib.save(label_data,save_label_path+"channal{}_".format(i)+file_name)

            image_data=nib.Nifti1Image(image_data_list[i],image_af)
            save_image_path=save_path+str(i)+"/images/"
            if not os.path.exists(save_image_path):
                os.makedirs(save_image_path)
            nib.save(image_data,save_image_path+"channal{}_".format(i)+file_name)
            
            

    def save_all_cut_nii(dir_path,save_dir,span=25):
        os.system("rm -rf "+save_dir)
        all_image_path=dir_path+"/images/"
        all_label_path=dir_path+"/labels/"
        image_list=os.listdir(all_image_path)
        label_list=os.listdir(all_label_path)
        for i in tqdm(range(len(image_list))):
            PrehandleData.save_cut_nii(all_label_path+label_list[i],all_image_path+image_list[i],
                                       save_dir,"{}.nii.gz".format(i+1),span=span)
    

class Drowbound():
    def get_edge(label_data):
        label_edge=torch.zeros(label_data.shape,device=DEVICE)
        for dim_index,shape in enumerate(label_data.shape):
            for i in range(1,shape):
                data1=torch.select(label_data,dim=dim_index,index=i-3)
                data2=torch.select(label_data,dim=dim_index,index=i)
                label_sub=(data1-data2).abs()
                if len(label_data.shape)==3:
                    if dim_index==0:
                        label_edge[i,:,:]=label_edge[i,:,:]+label_sub
                    if dim_index==1:
                        label_edge[:,i,:]=label_edge[:,i,:]+label_sub
                    if dim_index==2:
                        label_edge[:,:,i]=label_edge[:,:,i]+label_sub
                elif len(label_data.shape)==2:
                    if dim_index==0:
                        label_edge[i,:]=label_edge[i,:]+label_sub
                    if dim_index==1:
                        label_edge[:,i]=label_edge[:,i]+label_sub
        label_edge=label_edge/len(label_data.shape)
        return label_edge

    def show_add_niipicture(label_path,image_path):
        label_data=nib.load(label_path)
        label_data=label_data.get_fdata().astype(np.float32)
        label_data=torch.tensor(label_data,device=DEVICE)
        label_edge=Drowbound.get_edge(label_data)
        image_data=nib.load(image_path)
        image_data=image_data.get_fdata().astype(np.float32)
        image_data=image_data+label_edge.cpu().detach().numpy()*1000
        nib.viewers.OrthoSlicer3D(image_data).show()

    def expand_picture(data,coff=5):
        h,w=data.shape
        expand_data=torch.zeros((h*coff,w*coff),device=DEVICE)
        part_data=torch.ones((coff,coff),device=DEVICE)

        row_start=0
        for i in range(h):
            col_star=0
            for j in range(w):
                expand_data[row_start:row_start+coff,col_star:col_star+coff]=part_data*data[i,j]
                col_star=col_star+coff
            row_start=row_start+coff
        return expand_data
    
    def _normalize(data):
        data_max=torch.max(data)
        data_min=torch.min(data)
        data=(data-data_min)/(data_max-data_min)
        return data
    
    def _trans_ones_color(data,color=[1,0,0]):
        h,w=data.shape
        ones_data=torch.ones(data.shape,device=DEVICE)
        big_data=data>0
        one_data=ones_data*big_data
        color_data=torch.zeros((len(color),h,w),device=DEVICE)
        for i in range(len(color)):
            if sum(color)!=0:
                coff=color[i]/sum(color)
                color_data[i,:,:]=coff*one_data
            else:
                color_data[i,:,:]=i*one_data
        return color_data
    
    def replace_picture(image_data,label_data):
        c,h,w=image_data.shape
        mask_data=torch.ones((h,w),device=DEVICE)
        label_corlor_data=torch.sum(label_data,dim=0)
        mask_data=mask_data-label_corlor_data        
        for i in range(c):
            image_data[i,...]=image_data[i,...]*mask_data+label_data[i,...]
        return image_data

    
    def show_add_jpgpicture(label_path,image_path,coff=5,net=None,pred_model_path=None,image_save_path=None):
        label_data=nib.load(label_path)
        label_data=label_data.get_fdata().astype(np.float32)
        label_data=torch.tensor(label_data,device=DEVICE)
        expand_label_data=Drowbound.expand_picture(label_data[0,:,:],coff=coff)
        expand_label_edge=Drowbound.get_edge(expand_label_data)
        expand_label_edge=Drowbound._trans_ones_color(expand_label_edge,color=[1,0,0])

        image_data=nib.load(image_path)
        image_data=image_data.get_fdata().astype(np.float32)
        pred_image_data=torch.tensor([image_data],device=DEVICE)
        image_data=torch.tensor(image_data,device=DEVICE)

        if pred_model_path:
            predict_label_data=Drowbound.get_predict_label(net,pred_image_data,pred_model_path)
            expand_predict_data =Drowbound.expand_picture(predict_label_data,coff=coff)
            expand_pred_edge=Drowbound.get_edge(expand_predict_data)
            expand_pred_edge=Drowbound._trans_ones_color(expand_pred_edge,color=[0,0,1])


        c,h,w=image_data.shape
        expand_image_data=torch.zeros((c,h*coff,w*coff),device=DEVICE)
        for i in range(c):
            expand_image_data[i,:,:]=Drowbound.expand_picture(image_data[i,:,:],coff=coff)
            expand_image_data[i,:,:]=Drowbound._normalize(expand_image_data[i,:,:])

        image_data=Drowbound.replace_picture(expand_image_data,expand_label_edge)
        image_data=Drowbound.replace_picture(image_data,expand_pred_edge)
        # image_data=transforms.ToPILImage(image_data)

        if image_save_path:
            save_image(image_data,image_save_path)
        else:
            save_image(image_data,"./1.jpg")
            img=Image.open("./1.jpg")
            img.show()
    
    def get_predict_label(net,image_data,pred_model_path):
        net_stast_dict=torch.load(pred_model_path)
        net.load_state_dict(net_stast_dict, strict=True)
        # net.eval()
        outputs = net(image_data.nan_to_num(0))
        n,c,h,w=outputs.shape
        outputs=torch.argmax(outputs[0,...],dim=0)
        # outputs=outputs.transpose(1,2).transpose(2,3).contiguous().view(-1, c)
        return outputs
    
    def save_pred_picture(label_path,image_path,net,models_dirs,save_dir):
        for models_dir in os.listdir(models_dirs):
            dirs=models_dirs+"/"+models_dir
            model_path=dirs+"/"+os.listdir(dirs)[0]
            save_path=save_dir+"/"+models_dir+".jpg"
            print(models_dir)
            Drowbound.show_add_jpgpicture(label_path,image_path,net=net,
                                          pred_model_path=model_path,image_save_path=save_path)




        



if __name__=="__main__":
    # path="/home/yawei/Documents/LUNA16/Unet_data/05_Task01_BrainTumour/2d_save/0/images/channal0_2.nii.gz"
    # data=nib.load(path)
    # data = data.get_fdata().astype(np.float32)*400
    # nib.viewers.OrthoSlicer3D(data).show()
    # print(data.shape)



    #####################################################################################

    # client_dir="/home/yawei/Documents/LUNA16/Unet_data/clients"
    # save_dir="/home/yawei/Documents/LUNA16/Unet_data/05_Task01_BrainTumour/2d_save/"
    # config_path="../config/dataFederationConfig.csv"

    # clients=CutClients(client_dir,save_dir,client_num=6)
    # # # clients.make_balance_clients_dirs(long=320)
    # clients.make_clients_dirs_by_config(config_path)
    # clients.make_client_dirs(long=60)
    #####################################################################################


    # label_path="/home/yawei/Documents/berifen/Unet_lack23/FedStudy/unetdata/lack3/client0/labels/channal0_23.nii.gz"
    # image_path="/home/yawei/Documents/berifen/Unet_lack23/FedStudy/unetdata/lack3/client0/images/channal0_23.nii.gz"
    image_path="/home/yawei/Documents/berifen/Unet_lack23/FedStudy/unetdata/lack3/client0/images/channal0_10.nii.gz" # 10, 34,38
    label_path="/home/yawei/Documents/berifen/Unet_lack23/FedStudy/unetdata/lack3/client0/labels/channal0_10.nii.gz"
    pred_model_path="/home/yawei/Documents/berifen/useful_model/pFedNet/model_client1_segmentation_UNet2D.pth"
    models_dirs="/home/yawei/Documents/berifen/useful_model/models/"
    save_dir="/home/yawei/Documents/berifen/useful_model/picture/"
    net=Net().to(DEVICE)
    # data=Drowbound.show_add_jpgpicture(label_path,image_path,net=net,pred_model_path=pred_model_path)
    Drowbound.save_pred_picture(label_path,image_path,net,models_dirs,save_dir)

    #####################################################################################
    # dir_path="/home/yawei/Documents/LUNA16/Unet_data/05_Task01_BrainTumour/"
    # save_dir="/home/yawei/Documents/LUNA16/Unet_data/05_Task01_BrainTumour/save/"
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)
    # PrehandleData.save_all_cut_nii(dir_path,save_dir)
    #####################################################################################

    # image_path="/home/yawei/Documents/LUNA16/Unet_data/05_Task01_BrainTumour/images/BRATS_001.nii.gz"
    # dir_path="/home/yawei/Documents/LUNA16/Unet_data/05_Task01_BrainTumour/"
    # save_dir="/home/yawei/Documents/LUNA16/Unet_data/05_Task01_BrainTumour/2d_save/"
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)
    # data=PrehandleData.save_all_cut_nii(dir_path,save_dir,span=[0,32,32])

