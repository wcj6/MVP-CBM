
# python infer.py -c ./checkpoint/busi/test/best.pth -d busi --data-path ./dataset/busi/ --gpu 1

import os
import sys
import time
import math
import pdb
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms, models
from sklearn.metrics import balanced_accuracy_score
from optparse import OptionParser
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid, save_image
import matplotlib.pyplot as plt
import timm
from dataset.dataset import SkinDataset, cmmdDataset, busiDataset, idridDataset, cmDataset, edemaDataset, nctDataset, siimDataset
from model import mvpcbm
import utils

DEBUG = False


dataset_dict = {
    'isic2018': SkinDataset,
    'cmmd': cmmdDataset,
    'busi': busiDataset,
    'idrid': idridDataset,
    'cm': cmDataset,
    'edema': edemaDataset,
    'nct': nctDataset,
    'siim': siimDataset
}


def train_net(model, config):

    print(config.unique_name)
    
    train_transforms = copy.deepcopy(config.preprocess)
    train_transforms.transforms.pop(0)
    if model.model_name != 'clip':
        train_transforms.transforms.pop(0)
    train_transforms.transforms.insert(0, transforms.RandomVerticalFlip())
    train_transforms.transforms.insert(0, transforms.RandomHorizontalFlip())
    train_transforms.transforms.insert(0, transforms.RandomResizedCrop(size=(224,224), scale=(0.75, 1.0), ratio=(0.75, 1.33), interpolation=utils.get_interpolation_mode('bicubic')))
    train_transforms.transforms.insert(0, transforms.ToPILImage())

    val_transforms = copy.deepcopy(config.preprocess)
    val_transforms.transforms.insert(0, transforms.ToPILImage())

    trainset = dataset_dict[config.dataset](config.data_path, mode='train', transforms=train_transforms, flag=config.flag, debug=DEBUG, config=config, return_concept_label=True)
    trainLoader = DataLoader(trainset, batch_size=config.batch_size, shuffle=True, num_workers=8, drop_last=True)

    valset = dataset_dict[config.dataset](config.data_path, mode='val', transforms=val_transforms, flag=config.flag, debug=DEBUG, config=config, return_concept_label=True)
    valLoader = DataLoader(valset, batch_size=config.batch_size, shuffle=False, num_workers=2, drop_last=False)
    
    testset = dataset_dict[config.dataset](config.data_path, mode='test', transforms=val_transforms, flag=config.flag, debug=DEBUG, config=config, return_concept_label=True)
    testLoader = DataLoader(testset, batch_size=config.batch_size, shuffle=False, num_workers=2, drop_last=False)
    
    BMAC, acc, _, _ = validation(model, valLoader, config)
    print('BMAC: %.5f, Acc: %.5f'%(BMAC, acc))
    writer = SummaryWriter(config.log_path+config.unique_name)
    

        
def validation(model, dataloader,config):
    
    net = model
    net.eval()
   
    losses_cls = 0
    losses_concepts = 0

    pred_list = np.zeros((0), dtype=np.uint8)
    gt_list = np.zeros((0), dtype=np.uint8)

    test_img = []
    test_concept = []
    concept_labels = []
    logits = []

    test_time = []
    with torch.no_grad():
        for i, (data, label, concept_label) in enumerate(dataloader):
            test_img.append(data)
            data, label = data.cuda(), label.long().cuda()
            
            concept_label = concept_label.long().cuda()
            concept_labels.append(concept_label)
            start = time.perf_counter() 

            cls_logits, image_logits_dict,kl_loss = net(data)
            end = time.perf_counter()
            test_time.append(end - start)

            logits.append(cls_logits)

            image_logits_list = []
            for key in image_logits_dict.keys():
                image_logits_list.append(image_logits_dict[key])
        
            image_logits = torch.cat(image_logits_list, dim=-1)
        
            test_concept.append(image_logits)

            losses_concepts = 0

            _, label_pred = torch.max(cls_logits, dim=1)
            
            pred_list = np.concatenate((pred_list, label_pred.cpu().numpy().astype(np.uint8)), axis=0)
            gt_list = np.concatenate((gt_list, label.cpu().numpy().astype(np.uint8)), axis=0)
    total_sim = torch.cat(test_concept, dim=0)
    total_img = torch.cat(test_img, dim=0)
    concept_labels = torch.cat(concept_labels, dim=0)
 
    np_data = np.array(test_time)
    mean = np.mean(np_data)
    variance = np.var(np_data, ddof=0)  
    print("测试时间统计：")
    print(f"数据集: {config.dataset}")
    print(np_data)
    print(f"次数: {len(np_data)}")
    print(f"均值: {mean:.6f}")
    print(f"方差: {variance:.6f}")

    

    logits = torch.cat(logits, dim=0)
    print(total_sim.shape)
    print(gt_list.shape)
    print(pred_list.shape)
    print(total_img.shape)

    prefix = "/data/chunjiangwang/work/cbm_med/cem/Explicd/vis/siim/"
    prefix = os.path.dirname(prefix)
    np.save(prefix+'/data.npy',total_img.cpu().numpy())
    np.save(prefix+'/pred_sim.npy',total_sim.cpu().numpy())
    np.save(prefix+'/logits.npy',logits.cpu().numpy())
    np.save(prefix+'/concept_label.npy',concept_labels.cpu().numpy())
    np.save(prefix+'/gt.npy', gt_list)
    np.save(prefix+'/pred.npy',pred_list)

    # np.save('/data/chunjiangwang/work/cbm_med/cem/Explicd/dataset/test_img.npy',test_img)

    BMAC = balanced_accuracy_score(gt_list, pred_list)
    correct = np.sum(gt_list == pred_list)
    acc = 100 * correct / len(pred_list)

    return BMAC, acc, losses_cls/(i+1), losses_concepts/(i+1)




if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('-e', '--epochs', dest='epochs', default=100, type='int',
            help='number of epochs')
    parser.add_option('-b', '--batch_size', dest='batch_size', default=128,
            type='int', help='batch size')
    parser.add_option('--warmup_epoch', dest='warmup_epoch', default=5, type='int')
    parser.add_option('--optimizer', dest='optimizer', default='adamw', type='str')
    parser.add_option('-l', '--lr', dest='lr', default=0.0001, 
            type='float', help='learning rate')
    parser.add_option('-c', '--resume', type='str', dest='load', default=False,
            help='load pretrained model')
    parser.add_option('-p', '--checkpoint-path', type='str', dest='cp_path',
            #default='/data/yunhe/Liver/auto-aug/checkpoint/', help='checkpoint path')
            default='./checkpoint/', help='checkpoint path')
    parser.add_option('-o', '--log-path', type='str', dest='log_path', 
            default='./log/', help='log path')
    parser.add_option('-m', '--model', type='str', dest='model',
            default='MVPCBM', help='use which model')
    parser.add_option('--linear-probe', dest='linear_probe', action='store_true', help='if use linear probe finetuning')
    parser.add_option('-d', '--dataset', type='str', dest='dataset', 
            default='isic2018', help='name of dataset')
    parser.add_option('--data-path', type='str', dest='data_path',     
            default='/data/local/yg397/dataset/isic2018/', help='the path of the dataset')
    parser.add_option('-u', '--unique_name', type='str', dest='unique_name',
            default='MVPCBM', help='name prefix')
     

    parser.add_option('--flag', type='int', dest='flag', default=2)

    parser.add_option('--gpu', type='str', dest='gpu',
            default='0')
    parser.add_option('--amp', action='store_true', help='if use mixed precision training')

    (config, args) = parser.parse_args()
    
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu

    config.log_path = config.log_path + config.dataset + '/'
    config.cp_path = config.cp_path + config.dataset + '/'
    
    print('use model:', config.model)
    
    num_class_dict = {
        'isic2018': 7,
        'cmmd': 2,
        'idrid':5,
        'busi':3,
        'cm':2,
        'edema':2,
        'nct':9,
        'siim':2
    }

    # cls_weight_dict = {
    #     'isic2018': [1, 0.5, 1.2, 1.3, 1, 2, 2], 
        
    # }
   
    cls_weight_dict = {
        'isic2018': [1.2855, 0.2134, 2.7835, 4.3753, 1.3018,12.4410, 10.0755], 
        'busi': [3.1579,  0.9611, 2.0635], 
        'cmmd': [3.7037, 1.3776],
        # 'idrid': [3.071,20.64,3.071,5.548,8.323],
         'idrid': [0.614,4.128, 0.614, 1.11, 1.66 ],
         'cm': [1.985, 0.668],
         'edema': [1.206, 0.854],
         'nct':[0.6, 0.94, 2.35, 1.26, 0.77, 1.35, 1.08, 1.89, 0.65],
         'siim': [0.6423 , 2.2568]
    }
    concept_dict = {
    'isic2018': {
    'color': ['highly variable, often with multiple colors (black, brown, red, white, blue)',   'uniformly tan, brown, or black',  'translucent, pearly white, sometimes with blue, brown, or black areas',   'red, pink, or brown, often with a scale', 'light brown to black',   'pink brown or red', 'red, purple, or blue'],
    'shape': ['irregular', 'round', 'round to irregular', 'variable'],
    'border': ['often blurry and irregular', 'sharp and well-defined', 'rolled edges, often indistinct'],
    'dermoscopic patterns': ['atypical pigment network, irregular streaks, blue-whitish veil, irregular',  'regular pigment network, symmetric dots and globules',  'arborizing vessels, leaf-like areas, blue-gray avoid nests',  'strawberry pattern, glomerular vessels, scale',   'cerebriform pattern, milia-like cysts, comedo-like openings',    'central white patch, peripheral pigment network', 'depends on type (e.g., cherry angiomas have red lacunae; spider angiomas have a central red dot with radiating legs'],
    'texture': ['a raised or ulcerated surface', 'smooth', 'smooth, possibly with telangiectasias', 'rough, scaly', 'warty or greasy surface', 'firm, may dimple when pinched'],
    'symmetry': ['asymmetrical', 'symmetrical', 'can be symmetrical or asymmetrical depending on type'],
    'elevation': ['flat to raised', 'raised with possible central ulceration', 'slightly raised', 'slightly raised maybe thick']
},

    "cmmd": {
    "Mass Shape": [
        "Round/Oval: Smooth, well-defined edges",
        "Irregular: Asymmetrical with no definable shape",
        "Spiculated: Star-shaped with radiating lines"
    ],
    "Mass Margin": [
        "Circumscribed: Clear, well-defined borders",
        "Ill-defined: Blurred, indistinct borders",
        "Spiculated: Spiky, radiating margins"
    ],
    "Mass Density": [
        "Low Density (Radiolucent)",
        "Isodense: Similar to surrounding tissue",
        "High Density (Radiopaque)"
    ],
    "Calcifications": [
        "Absent: No calcifications present",
        "Benign Calcifications: Macrocalcifications with smooth shapes",
        "Suspicious Calcifications: Clustered microcalcifications with irregular patterns"
    ],
    "Architectural Distortion": [
        "None: Normal breast architecture",
        "Minimal: Slight distortion without associated mass",
        "Significant: Noticeable distortion often linked to an underlying mass"
    ],
    "Asymmetry": [
        "None: Symmetrical breast tissue",
        "Mild: Slight differences in breast tissue density or shape",
        "Marked: Pronounced differences with associated suspicious features"
    ]
},

    "idrid": {
        "Color": [
            "Small red dots",
             "Increased redness",
             "Deeper red hue",
            "Dark red to purple",
             "Very dark red to obscured"
        ],
        "Shape": [
             "Circular or slightly irregular",
             "Irregular shapes emerging",
            "More irregular and varied shapes",
            "Highly irregular and varied shapes",
             "Highly distorted, irregular forms"
        ],
        "Border": [
             "Well-defined",
             "Slightly blurred",
             "Partially obscured",
             "Poorly defined",
             "Obscured by neovascularization or fibrous tissue"
        ],
        "Texture": [
            "Smooth",
             "Slightly granular",
            "Varied textures",
             "Rough and heterogeneous",
             "Irregular, coarse textures"
        ],
        "Symmetry": [
             "Typically symmetrical",
             "Mild asymmetry possible",
             "Asymmetrical distribution",
            "Highly asymmetrical and extensive",
             "Highly asymmetrical"
        ],
        "Elevation": [
             "Flat",
            "Mostly flat",
             "Mostly flat",
             "Some elevation may occur",
            "Possible elevation or distortion"
        ]
    },
    "busi": {
    "Echogenicity": [
        "Anechoic (completely dark, fluid-filled)",
        "Hypoechoic (slightly darker than surrounding tissue)",
        "Isoechoic (similar echogenicity to surrounding tissue)",
        "Hyperechoic (brighter than surrounding tissue)",
        "Markedly hyperechoic with shadowing"
    ],
    "Shape": [
        "Oval (smooth, regular edges)",
        "Round (circular, symmetric)",
        "Lobulated (irregular but with smooth transitions)",
        "Angular (sharp, distinct edges)",
        "Irregular (no definable shape, spiculated)"
    ],
    "Margin": [
        "Circumscribed (clear, well-defined borders)",
        "Fuzzy (slightly blurred borders)",
        "Microlobulated (small, multiple lobules at the edges)",
        "Obscured (poorly defined borders)",
        "Spiculated (spiky, radiating lines from the margin)"
    ],
    "Orientation": [
        "Parallel (aligned with the skin surface)",
        "Not parallel (perpendicular or non-aligned)",
        "Antiparallel (tilted away from the skin surface)",
        "Complex orientation with mixed alignment",
        "Variable orientation with no consistent pattern"
    ],
    "Posterior_Features": [
        "Enhancement (increased brightness behind the lesion)",
        "No significant change",
        "Shadowing (reduced echogenicity behind the lesion)",
        "Reverberation artifacts",
        "Echogenic foci with shadowing"
    ],
    "Surrounding_Tissue": [
        "Normal echotexture with no surrounding abnormalities",
        "Minimal surrounding tissue changes",
        "Increased echogenicity in surrounding tissue",
        "Decreased echogenicity or fibrosis around the lesion",
        "Significant architectural distortion of surrounding tissue"
    ]
},
    "edema": {
    "Lung Texture": [
        "Normal lung texture with no abnormal increase",
        "Mild increase in lung texture with slight blurring",
        "Moderate increase in lung texture, widely distributed",
        "Severe increase in lung texture, exhibiting honeycombing or grid-like patterns",
        "Extremely increased lung texture, obscuring normal structures"
    ],
    "Lung Field Opacity": [
        "Normal lung field opacity with no abnormal shadows",
        "Mildly elevated opacity with localized blurring",
        "Moderately elevated opacity with regional shadows",
        "Significantly elevated opacity with extensive shadows",
        "Extremely high opacity with diffuse infiltrative shadows"
    ],
    "Fluid Accumulation": [
        "No fluid accumulation",
        "Mild fluid accumulation with localized effusion",
        "Moderate fluid accumulation with noticeable effusion",
        "Significant fluid accumulation with large effusion",
        "Extensive fluid accumulation, disrupting normal structures"
    ],
    "Cardiomegaly": [
        "Normal heart size with no enlargement",
        "Mild heart enlargement with slightly widened cardiac silhouette",
        "Moderate heart enlargement with noticeably widened cardiac silhouette",
        "Significant heart enlargement with markedly expanded cardiac silhouette",
        "Severe heart enlargement with extremely expanded cardiac silhouette"
    ],
    "Pleural Reaction": [
        "No pleural reaction; pleura appears clear",
        "Mild pleural thickening with slightly blurred borders",
        "Moderate pleural thickening with noticeably blurred borders",
        "Significant pleural thickening with unclear borders",
        "Severe pleural thickening or adhesions with distorted pleural structure"
    ],
    "Tracheal Position": [
        "Trachea is centered with no deviation",
        "Mild tracheal deviation from the center without affecting symmetry",
        "Moderate tracheal deviation from the center with slight symmetry changes",
        "Significant tracheal deviation from the center with noticeable symmetry alterations",
        "Severe tracheal deviation with mediastinal structure displacement"
    ]
},
   "cm": {

    "Heart Silhouette": [
        "Oval shape with smooth borders",
        "Slightly globular shape with minor irregularities",
        "Moderately globular or elongated shape",
        "Significantly globular or irregular shape with pronounced borders",
        "Highly irregular or distorted shape with unclear borders"
    ],
    "Mediastinal Shift": [
        "No shift; mediastinum is central",
        "Minor shift towards one side without significant displacement",
        "Moderate shift towards one side with noticeable displacement",
        "Significant shift towards one side affecting mediastinal structures",
        "Severe shift causing substantial displacement of mediastinal structures"
    ],
    "Pulmonary Congestion": [
        "No signs of pulmonary congestion",
        "Mild vascular congestion without interstitial markings",
        "Moderate vascular congestion with some interstitial markings",
        "Severe vascular and interstitial congestion with prominent markings",
        "Extensive pulmonary congestion with alveolar filling patterns"
    ],
    "Associated Findings": [
        "No associated findings",
        "Mild pleural effusion or slight pulmonary edema",
        "Moderate pleural effusion, cardiogenic pulmonary edema",
        "Significant pleural effusion, extensive pulmonary edema",
        "Massive pleural effusion, acute respiratory distress signs"
    ]
},
   "nct": {
    "Color": [
        "Light pink to yellow",
        "Variable, depending on surrounding tissues",
        "Dark brown to black",
        "Dark blue to purple",
        "Light blue to clear",
        "Deep pink to reddish-brown",
        "Light pink to reddish",
        "Darker pink to brown",
        "Deep pink to purple"
    ],
    "Texture": [
        "Soft, homogeneous",
        "Homogeneous or heterogeneous without specific features",
        "Irregular, clumped",
        "Compact, dense",
        "Gelatinous, amorphous",
        "Striated, elongated cells",
        "Layered, glandular",
        "Fibrotic, dense",
        "Pleomorphic, hyperchromatic nuclei"
    ],
    "Shape": [
        "Rounded or lobulated clusters",
         "Fragmented and amorphous",
         "Small, round nuclei with scant cytoplasm",
         "Variable, often forming pools or secretory droplets",
        "Spindle-shaped nuclei aligned in parallel bundles",
         "Regular crypt structures with uniform size",
         "Irregular, interwoven fibers",
         "Irregular glandular structures, loss of uniform crypt shape"
    ],
    "Size": [
         "Variable, often dispersed throughout the tissue",
        "Small to medium-sized clusters",
        "Consistently small across samples",
       "Variable, can be widespread or localized",
         "Uniform, forming fascicles",
         "Consistent glandular units throughout",
         "Variable, often expanding around tumor cells",
         "Variable, with increased nuclear size and mitotic figures"
    ],
    "Additional Features": [
        "Contains lipid droplets and clear cytoplasm",
         "Non-specific areas without identifiable structures",
         "Presence of necrotic material and cellular fragments",
         "High nucleus-to-cytoplasm ratio, lack of prominent nucleoli",
       "Presents as extracellular material, often surrounding glands",
         "Presence of muscle fibers with characteristic striations",
         "Presence of goblet cells and well-organized epithelial layers",
         "Increased collagen deposition, myofibroblasts, and altered extracellular matrix",
         "Disorganized architecture, glandular crowding, mucin production, and invasive growth patterns"
    ]
},
"siim": {
    "Radiolucency": [
        "Distinct radiolucent area indicating air presence",
        "Large radiolucent space in the apex of the lung",
        "Radiolucent zone typically located at the periphery of the lung fields"
    ],
    "Pleural Line": [
        "Sharp and clear pleural line separating lung tissue from air",
        "Pleural line may appear as a thin, straight or irregular line",
        "Absence of lung markings beyond the pleural line"
    ],
    "Lung Markings": [
        "No visible lung markings beyond the pleural line",
        "Sudden termination of lung markings at the pleural line",
        "Absence of normal lung parenchymal structures beyond the pleural line"
    ],
    "Mediastinal Shift": [
        "Shift of mediastinal structures towards the opposite side in tension pneumothorax",
        "Displacement of the heart shadow and trachea towards the unaffected side",
        "Mediastinal shift observed in severe cases as an indication of tension pneumothorax"
    ],
    "Lung Edge": [
        "Visible lung edge with a clear boundary against the chest wall",
        "Lung edge may appear serrated or smooth",
        "No extension of normal lung tissue beyond the edge"
    ],
    "Size": [
        "Small pneumothorax: < 3 cm between the pleural line and chest wall at the level of the hilum",
        "Moderate pneumothorax: 3-6 cm distance",
        "Large pneumothorax: > 6 cm distance",
        "Pneumothorax size can be assessed by measuring the distance between the pleural line and chest wall"
    ],
    "Additional Features": [
        "Compression or collapse of lung tissue in the affected area",
        "Blurring of the cardiac silhouette if significant",
        "Presence of subcutaneous emphysema",
        "Visible diaphragmatic depression on the affected side",
        "In lateral views, elevated hemidiaphragm on the affected side"
    ]
}

    }
    
    
    
    config.cls_weight = cls_weight_dict[config.dataset]
    config.num_class = num_class_dict[config.dataset]
    concept_list = concept_dict[config.dataset]


    
    # from concept_dataset import explicid_isic_dict
    # concept_list = explicid_isic_dict
    # net = ExpLICD(concept_list=concept_list, model_name='biomedclip', config=config)

    net = prototype(concept_list=concept_list, model_name='biomedclip', config=config)

   
    # We find using orig_in21k vit weights works better than biomedclip vit weights
    # Delete the following if want to use biomedclip weights
    vit = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=config.num_class)
    vit.head = nn.Identity()
    net.model.visual.trunk.load_state_dict(vit.state_dict())

  

    if config.load:
        net.load_state_dict(torch.load(config.load))
        print('Model loaded from {}'.format(config.load))
 
    net.cuda()
    

    train_net(net, config)

    print('done')
        

# python test.py -c /data/chunjiangwang/work/cbm_med/cem/Explicd/checkpoint/isic2018/test/CP100.pth -d isic2018 --data-path /data/chunjiangwang/work/cbm_med/cbm/Explicd/dataset/isic2018/ --gpu 1