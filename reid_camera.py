
#Equivalent to the evaluate_gpu.py foldar in the baseline code
import scipy.io
import torch
import numpy as np
from scipy import stats
import os
import pandas as pd
import matplotlib.pyplot as plt       
import numpy as np
import argparse
parser = argparse.ArgumentParser(description='evluate_nCamera_reid')
parser.add_argument('--feature_file_dir',default='C:\\Users\\Mavara\\Desktop\\person_reid\\Implementation\\mat_result\\D3_D1_pytorch_result.mat',type=str, help='path mat file(feature_file)')
parser.add_argument('--camQ',default='D1',type=str, help='name camera Query')
parser.add_argument('--camG',default='D3',type=str, help='name camera Gallery')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device:',device)
arg = parser.parse_args()
camQ = arg.camQ
camG = arg.camG
feature_file_dir = arg.feature_file_dir
result = scipy.io.loadmat(feature_file_dir)
query_feature = torch.FloatTensor(result['query_f'])
query_label = result['query_label']
gallery_feature = torch.FloatTensor(result['gallery_f'])
gallery_cam = result['gallery_cam']
query_cam = result['query_cam']
gallery_label = result['gallery_label']
gallery_name = result['gallery_name']
query_name = result['query_name']
gallery_path=result['gallery_path']
query_path=result['query_path']
gallery_frame=result['gallery_frame'][0]
query_frame=result['query_frame'][0]
x_data = result['gallery_frame'][0]
print('gallery_label:',np.unique(np.array(gallery_label)))
print('query_label:',np.unique(np.array(query_label)))
def csv_to_json(csvFilePath, jsonFilePath):
    import csv 
    import json 
    jsonArray = []
    import sys
    import csv
    maxInt = sys.maxsize

    while True:
        # decrease the maxInt value by factor 10 
        # as long as the OverflowError occurs.

        try:
            csv.field_size_limit(maxInt)
            break
        except OverflowError:
            maxInt = int(maxInt/10)  
    #read csv file
    with open(csvFilePath, encoding='utf-8') as csvf: 
        #load csv file data using csv library's dictionary reader
        csvReader = csv.DictReader(csvf) 

        #convert each csv row into python dict
        for row in csvReader: 
            #add this python dict to json array
            jsonArray.append(row)
  
    #convert python jsonArray to JSON String and write to file
    with open(jsonFilePath, 'w', encoding='utf-8') as jsonf: 
        jsonString = json.dumps(jsonArray, indent=4)
        jsonf.write(jsonString)


def select_best_gallery(score,gallery_label,gallery_name):
    arg_max1 = []
    for i in np.unique(gallery_label):
        index_gallery = list(np.where(np.array(gallery_label==i)))[0]
        arg_m = index_gallery[np.argmax(score[index_gallery].cpu().numpy())]
        
        value_max= score[arg_m]
        ID = [j.split('_') for j in gallery_name[index_gallery]]
        frame = list(zip(*ID))[1]
        # frame = [int(frame1[i][1:]) for i in range(len(frame1))]
        # frame.sort()
        # print("besttttt////frame:",'F'+str(frame[0]))
        # start_query_frame.append('F'+str(frame[0]))
        name_max = gallery_name[arg_m].replace('.', '')
        mean_score = np.mean(score[index_gallery].cpu().numpy())
        result_dict={"value_max":float(value_max),'mean_score':float(mean_score)}
        arg_max1.append({'name':name_max.replace(' ', ''),'start_frame':frame[0].replace(' ', ''),'score':result_dict})
    arg_max1 = np.array(arg_max1)

    # result = arg_max1[arg_max1[:,column index].argsort()]
    # sort_by_maxScore = np.flipud (arg_max1[arg_max1[:,3].argsort()])
    return arg_max1

def TwoCamera_reid(name_camQ,name_camG,query_label,query_feature,gallery_feature,query_name,gallery_label,gallery_name):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('TwoCamera_reid')
    name_query = []
    start_query_frame=[]
    row_score =[]
    info_obj=[]
    for i in np.unique(query_label):
            # به ازای هر ایدی از کوئری، اندیس ان ایدی خاص را جدا میکنیم
            index_q = list(np.where(np.array(query_label==i)))[0]
            # از حیث زمانی فریم وسط آن را انتخاب کرده
            num= index_q[int(len(index_q)/2)]
            ID = [j.split('_') for j in query_name[index_q]]
            # frame1: ['F0', 'F100', 'F101', 'F102',...]
            frame1 = list(list(zip(*ID))[1])
            # frame: [0, 100, 101, 102,...]
            frame = [int(frame1[i][1:]) for i in range(len(frame1))]
            frame.sort()
            # فریم شروع آن ایدی در ویدئو را برمیگردانیم
            start_query_frame.append('F'+str(frame[0]))
            # ویژگی های فریم وسط آن ایدی را استخراج کرده
            query = query_feature[num].view(-1,1)
            # امتیاز بین آن کوئری و کل گالری را درمیاوریم
            score = torch.mm(gallery_feature.to(device),query.to(device))
            row_score.append([item[0] for item in list(score.cpu().numpy())])
            #   شبیه ترین تصاویر گالری (به ازای هر آیدی از گالری--> شبیه ترین عکس آن آیدی) به کوئری از نظر امتیاز را برمیگردانیم
            # {'name': 'fruit12_F3232_T67', 'start_frame': 'F3230', 'score': {'value_max': 0.5293111205101013, 'mean_score': 0.47513994574546814}}
            list_object = select_best_gallery(score=score,gallery_label=gallery_label,gallery_name=gallery_name)
            # result_max.append([item[3] for item in sort_by_maxScore])
            name_query.append(query_name[num])
            # info_obj.append(list_object)
            info_obj.append(list(list_object))
            print('list_object:',list_object)
    # dict = {'name_query': name_query, 'info_obj': info_obj,'row_score':row_score,'len_unique(gallery_label)':len(np.unique(gallery_label))}
    dict1 = [{'name': (name_query.replace(' ', '')).replace('.', ''),'start_frame':start_query_frame, 'TopGalleryItems': info_obj,"video_query_path":'D1_1_1part.mp4',"video_gallery_path":'D3_1_1part.mp4'} for name_query,start_query_frame,info_obj in zip(name_query,start_query_frame,info_obj)]


    df = pd.DataFrame(dict1)
    if not os.path.exists("./json_reid"):
         os.makedirs("./json_reid")
    name = "./json_reid/"+str(name_camQ)+'_'+str(name_camG)+'_TwoCamera_reid'
    df.to_csv(name + '.csv', index=False)
    csv_to_json(csvFilePath= name + '.csv', jsonFilePath= name + '.json')
    os.remove(name + '.csv')


def OneCamera_reid(name_camQ,label,feature,name):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('OneCamera_reid')
    name_query = []
    row_score =[]
    start_query_frame=[]
    info_obj=[]
    for i in np.unique(label):
            index_Q = list(np.where(np.array(label==i)))[0]
            # اندیس همه ی تاصویر به غیر از ایدی ای که به عنوان کوئری معرفی شده
            index_G = list(np.where(np.array(label!=i)))[0]
            num= index_Q[int(len(index_Q)/2)]
            name_G = name[index_G]
            label_G = label[index_G]
            feature_G = feature[index_G]
            ID = [j.split('_') for j in name[index_Q]]
            frame = list(zip(*ID))[1]
            start_query_frame.append(frame[0])
            query = feature[num].view(-1,1)
            score = torch.mm(feature_G.to(device),query.to(device))
            row_score.append([item[0] for item in list(score.cpu().numpy())])
            list_object = select_best_gallery(score=score,gallery_label=label_G,gallery_name=name_G)
            # result_max.append([item[3] for item in sort_by_maxScore])
            x = name[num].replace('.', '')
            name_query.append(x)
            info_obj.append(list(list_object))
            # info_obj.append((list_object))
            dict = [{'name': name_query.replace(' ', ''), 'TopGalleryItems': info_obj,'start_frame':start_query_frame,"video_path":str(name_camQ)+'_1_1part.mp4','len_unique_label_G':len(np.unique(label_G))} for name_query,info_obj,start_query_frame in zip(name_query,info_obj,start_query_frame)]
    # key.replace('"', ''):val
    # dict = {key:val.replace('"[{','[{' ) for key, val in dict.items()}
    # dict = {'name_query': name_query.replace(' ', ''), 'info_obj': info_obj,'len_unique_label_G':len(np.unique(label_G))}
    # dict.update({"video_path":str(camQ)+'_1_1part.mp4'})
    df = pd.DataFrame(dict)
    if not os.path.exists("./json_reid"):
         os.makedirs("./json_reid")
    name = "./json_reid/"+str(name_camQ)+'_OneCamera_reid'
    df.to_csv(name + '.csv', index=False)
    csv_to_json(csvFilePath= name + '.csv', jsonFilePath= name + '.json')
    os.remove(name + '.csv')



def score_time(name,label,label_id):
    # قراره بین هر دو ایدی از کوئری و گالری، فریم وسط گالری انتخاب بشه 
    index = list(np.where(np.array(label==np.array(label_id))))[0]
    name_frame = name[index]
    ID = [i.split('_') for i in name_frame]
    frame = list(zip(*ID))[1]
    print('frame[len(frame)/2]:',frame[int(len(frame)/2)])
    return frame[int(len(frame)/2)] 


# score_time(name=query_name,label=query_label,label_id=query_label[0])

TwoCamera_reid(name_camQ = camQ,name_camG = camG ,query_label=query_label,query_feature=query_feature,gallery_feature=gallery_feature,query_name=query_name,gallery_label=gallery_label,gallery_name=gallery_name)
OneCamera_reid(name_camQ = camQ ,label = query_label,feature = query_feature,name = query_name)
OneCamera_reid(name_camQ = camG ,label =gallery_label,feature = gallery_feature,name =gallery_name)
