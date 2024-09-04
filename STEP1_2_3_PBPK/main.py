from TrainFunction import*

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.00000002, help='learning rate, default=0.0002')
parser.add_argument('--modelLr', type=float, default=0.00001, help='learning rate, default=0.000001')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', default = True, action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--gpu_ids', type=int, default=0, help='ids of GPUs to use')
parser.add_argument('--epoch_iter', type=int,default=300, help='number of epochs on entire dataset')
parser.add_argument('--decay', type=float,default=0.0001, help='regolarizzazione')
parser.add_argument('--numberOfClasses', type=int,default=2, help='numero di classi del problema')
parser.add_argument('-f',type=str,default= '', help='dummy input required for jupyter notebook')
parser.add_argument('--Continue', default = False,  help='continue learning')
parser.add_argument('--model', default = 'tk_w', help='model to fit')
parser.add_argument('--fold',type=int, default = 0, help='CV FOLD')
parser.add_argument('--model_cnn', default = 'alexnet', help='CNN model')
opt = parser.parse_args()


CV_FOLD_test = [
    ['AMBL-001','AMBL-003','AMBL-004','AMBL-005','AMBL-007','AMBL-008','AMBL-010','AMBL-011','AMBL-013','AMBL-016','AMBL-620'],
    ['AMBL-018','AMBL-022','AMBL-023','AMBL-025','AMBL-028','AMBL-029','AMBL-031','AMBL-032','AMBL-038','AMBL-615'],
    ['AMBL-026','AMBL-033','AMBL-034','AMBL-036','AMBL-047','AMBL-050','AMBL-496','AMBL-507','AMBL-514','AMBL-557'],
    ['AMBL-042','AMBL-043','AMBL-045','AMBL-046','AMBL-049','AMBL-541','AMBL-562','AMBL-564','AMBL-565','AMBL-567','AMBL-568','AMBL-569'],
    ['AMBL-044','AMBL-559','AMBL-570','AMBL-571','AMBL-572','AMBL-574','AMBL-575','AMBL-577','AMBL-578','AMBL-579'],
    ['AMBL-561','AMBL-563','AMBL-573','AMBL-580','AMBL-581','AMBL-582','AMBL-583','AMBL-584','AMBL-595','AMBL-605','AMBL-626'],
    ['AMBL-009','AMBL-585','AMBL-586','AMBL-587','AMBL-590','AMBL-591'],
    ['AMBL-593','AMBL-594','AMBL-596','AMBL-597','AMBL-598','AMBL-599','AMBL-600','AMBL-603'],
    ['AMBL-604','AMBL-607','AMBL-608','AMBL-610','AMBL-612','AMBL-613','AMBL-618','AMBL-619','AMBL-627','AMBL-629'],
    ['AMBL-024','AMBL-035','AMBL-588','AMBL-606','AMBL-621','AMBL-622','AMBL-623','AMBL-625','AMBL-628','AMBL-631','AMBL-632'],
    ]
####### TRAINING #######

torch.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]=str(opt.gpu_ids)
cuda_str = 'cuda:' + str(opt.gpu_ids)
print(cuda_str)


opt.imgSize=64
opt.use_dropout = 0
opt.ngf = 32
opt.ndf = 32
# dimensionality: shading latent code
opt.sdim = 16
# dimensionality: albedo latent code
opt.tdim = 16
# dimensionality: texture (shading*albedo) latent code
opt.idim = opt.sdim + opt.tdim
# dimensionality: warping grid (deformation field) latent code
opt.wdim = 128
# dimensionality of general latent code (before disentangling)
opt.zdim = 128
opt.use_gpu = True
opt.ngpu = 1
opt.nc = 3
opt.useDense = False
cudnn.benchmark = True

fold = opt.fold
net_name = opt.model_cnn 
print(net_name)
MaxNumberOfEpoch = opt.epoch_iter
numberOfClasses = opt.numberOfClasses
batch_size = opt.batchSize
decay = opt.decay
dim_to_resize = 224

loader_opts = {'batch_size': batch_size, 'num_workers': 0, 'pin_memory': True} 
classes = ['benign', 'malignant']
basePath = 'D:/LavoroJBHI_PBPK_DAE/PUBLIC_DATA/'

pathToRealData = basePath+ 'FINAL_DATASET/'
pathTocsv = basePath
malignant_csv_name = 'df_malignant.csv'
benign_csv_file = 'df_benign.csv'

pathResult_train = basePath+'Results/STEP_PBPK_FINAL_Version/DAE_CNN_STEP1_2_3_' + net_name + '/TRAIN_LOOP/Fold'+str(fold) + '/' #ci mettiamo i pesi nel corso degli addestramenti->ci sono i pesi migliori e quelli di ogni epoca
pathResult_final = basePath+'Results/STEP_PBPK_FINAL_Version/DAE_CNN_STEP1_2_3_' + net_name + '/FINAL_LOOP/Fold'+str(fold) + '/' #ci mettiamo i pesi nel riaddestramento -> pesi di ogni epoca del riaddestramento

pathDAE = basePath+'Results/DAE_noDense_PBPK_LOSS_data_Aug/pesi/'
pathDAE_type = opt.model
pathDAE += f'/{pathDAE_type}/'
modelPath = basePath + 'Results/BASIC_PBPK_FINAL/Class_Task_'+ net_name +'_WarpedPBPK/Fold' + str(fold)+ '/'

os.makedirs(pathResult_train, exist_ok=True)
os.makedirs(pathResult_final, exist_ok=True)

val_patient = CV_FOLD_test[fold-1]
test_patient = CV_FOLD_test[fold]


choice = transforms.RandomChoice([Random_Rotation_both(p=0.5, n=1),
                                  Random_Rotation_both(p=0.5, n=2), 
                                  Random_Rotation_both(p=0.5, n=3)])

train_transform = transforms.Compose([ToTensor_both(),
                                      Random_Horizontal_Flip_both(p=0.5),
                                      Random_Vertical_Flip_both(p=0.5),
                                      choice
                                      ])

val_transform = ToTensor_both()
test_transform = ToTensor_both()


include_patient_train = lambda path: path.split('/')[-1].split('_')[0] not in (val_patient + test_patient)
include_patient_val = lambda path: path.split('/')[-1].split('_')[0] in val_patient 
include_patient_test = lambda path: path.split('/')[-1].split('_')[0] in test_patient 


#-------------------------> TRAIN IMAGES
print('---->Loading Train')
Trainset = []
for c in classes:
  print(f'   Loading: {c}')
  is_valid_class = lambda path: c in path.split('/') #c == path.split('/')[5]
  check_file = lambda path: include_patient_train(path) and is_valid_class(path)
  dataset = My_DatasetFolder(root = pathToRealData, transform=train_transform, is_valid_file=check_file, list_classes=classes, loader = loader_fc)
  Trainset.append(dataset)
print('   [TRAIN] Benign: %d Malignant %d ' %(len(Trainset[0]),len(Trainset[1])))
completeTrainSet = BalanceConcatDataset(Trainset)
print('   [TRAIN] Balance - - - Benign: %d Malignant %d ' %(len(Trainset[0]),len(Trainset[1])))

print('---->Loading Validation')
dataset_real_validation = My_DatasetFolder(root = pathToRealData, transform=val_transform, is_valid_file=include_patient_val, list_classes=classes, loader = loader_fc)
print('   [Real VAL] %d' %(len(dataset_real_validation)))

print('---->Loading Test')
dataset_real_test = My_DatasetFolder(root = pathToRealData, transform=test_transform, is_valid_file=include_patient_test, list_classes=classes, loader = loader_fc)
print('   [Real TEST] %d' %(len(dataset_real_test)))


print('---->Loading vector for train data augmentation')
dataset_benign = CSV_DatasetFolder(root = pathTocsv, csv_path = benign_csv_file, 
                            imagePath = pathToRealData, exclude_list= val_patient + test_patient, 
                            transform = train_transform,
                            include_list = None, list_classes = classes, valid_class = 'benign')
                         
dataset_malignant = CSV_DatasetFolder(root = pathTocsv, csv_path = malignant_csv_name, 
                            imagePath = pathToRealData, exclude_list= val_patient + test_patient, 
                            transform = train_transform,
                            include_list = None, list_classes = classes, valid_class = 'malignant')

print('   [TRAIN CSV] Benign: %d Malignant %d ' %(len(dataset_benign),len(dataset_malignant)))


Val_csv = []
dataset = CSV_DatasetFolder(root = pathTocsv, csv_path = benign_csv_file,
                            imagePath = pathToRealData, exclude_list= None, 
                            transform = val_transform,
                            include_list = val_patient,  list_classes = classes, valid_class = 'benign')
Val_csv.append(dataset)
dataset = CSV_DatasetFolder(root = pathTocsv, csv_path = malignant_csv_name, 
                            imagePath = pathToRealData, exclude_list= None, 
                            transform = test_transform,
                            include_list = val_patient,  list_classes = classes, valid_class = 'malignant')
Val_csv.append(dataset)

Val_csv_final = ConcatDataset(Val_csv)
print('   [VAL CSV] Benign: %d Malignant %d ' %(len(Val_csv[0]),len(Val_csv[1])))

criterionRecon      = nn.L1Loss()
criterionTVWarp     = TotalVaryLoss(opt)
criterionBiasReduce = BiasReduceLoss(opt)
criterionSmoothL1   = TotalVaryLoss(opt)
criterionSmoothL2   = SelfSmoothLoss2(opt) 
criterionPBPK       = PBPKLoss(opt)
criterionCNN = nn.CrossEntropyLoss()


model_conv = buildModel(net_name,numberOfClasses)
if opt.useDense:
  encoders = Dense_Encoders_Intrinsic(opt)
  decoders = Dense_DecodersIntegralWarper2_Intrinsic(opt)
else:
  encoders = Encoders_Intrinsic(opt)
  decoders = DecodersIntegralWarper2_Intrinsic(opt)
    



updator_encoders     = optim.Adam(encoders.parameters(), lr = opt.lr, betas=(opt.beta1, 0.999))  #ottimizzazione encoder
updator_decoders     = optim.Adam(decoders.parameters(), lr = opt.lr, betas=(opt.beta1, 0.999))  #ottimizzazione decoder

optimizer_conv       = optim.Adam([{'params': encoders.parameters(), 'lr': opt.modelLr},
                                   {'params': decoders.parameters(), 'lr': opt.modelLr},
                                   {'params': model_conv.parameters(), 'lr': opt.modelLr, 'weight_decay':opt.decay}])

optimizer_conv_decoders = optim.Adam([{'params': decoders.parameters(), 'lr': opt.modelLr},
                                   {'params': model_conv.parameters(), 'lr': opt.modelLr, 'weight_decay':opt.decay}])
#--------------------------------------- INIZIALIZZAZIONE DEI PESI
if opt.Continue:
    print('Reload previous model at: ' + pathResult_train)
    print(f'Dense: {opt.useDense}')
    encoders.load_state_dict(torch.load(pathResult_train + 'encoders_weights.pth'))
    decoders.load_state_dict(torch.load(pathResult_train + 'decoders_weights.pth'))
    model_conv.load_state_dict(torch.load(pathResult_train + 'cnn_weights.pth'))
    stato = sio.loadmat(os.path.join(pathResult_train , f'check_point.mat'))
    best_acc = stato['best_acc'][0][0]
    best_loss = stato['best_loss'][0][0]
    best_epoca = stato['best_epoca'][0][0]
    best_acc_m = stato['best_acc_m'][0][0]
    startEpoch = stato['last_epoch'][0][0] + 1
else:
    print('load DAE initial weights --> ')
    print(f'Dense: {opt.useDense}')
    print(pathDAE)
    print(modelPath)
    
    model_conv.load_state_dict(torch.load(modelPath + 'best_model_weights.pth'))
    
    #if opt.useDense:
    #  encoders.load_state_dict(torch.load(pathDAE + f'wasp_model_epoch_encoders_{pathDAE_type}_dense.pth'))
    #  decoders.load_state_dict(torch.load(pathDAE + f'wasp_model_epoch_decoders_{pathDAE_type}_dense.pth'))
    #else:
    #  encoders.load_state_dict(torch.load(pathDAE + f'wasp_model_epoch_encoders_{pathDAE_type}.pth'))
    #  decoders.load_state_dict(torch.load(pathDAE + f'wasp_model_epoch_decoders_{pathDAE_type}.pth'))
   
    encoders.load_state_dict(torch.load(pathDAE + f'wasp_model_epoch_encoders_{pathDAE_type}_dense.pth'))
    decoders.load_state_dict(torch.load(pathDAE + f'wasp_model_epoch_decoders_{pathDAE_type}_dense.pth'))
      
    best_acc = 0.0   
    best_acc_m = 0.0
    best_loss = 0.0 
    best_epoca = 0
    startEpoch = 1

  
for p in encoders.parameters():
  p.requires_grad=True

for p in decoders.parameters():
  p.requires_grad=True
  
  
model_conv.cuda()
encoders.cuda()
decoders.cuda()

now = datetime.datetime.utcnow()

print(f'START TRAIN LOOP VALIDATION {now} [UTC]')

train_loop_validation(model_conv, encoders, decoders, dim_to_resize,
                      completeTrainSet, dataset_benign, dataset_malignant, dataset_real_validation, Val_csv_final, dataset_real_test,
                          optimizer_conv, updator_encoders, updator_decoders, optimizer_conv_decoders,
                          MaxNumberOfEpoch, loader_opts, startEpoch,
                          criterionCNN, criterionRecon, criterionTVWarp, criterionBiasReduce, criterionSmoothL1, criterionSmoothL2, criterionPBPK,
                          best_acc, best_acc_m, best_loss, best_epoca, pathResult_train, opt, True)

model_conv.cpu()
encoders.cpu()
decoders.cpu()
del model_conv
del encoders
del decoders


#-------------------> predizione sul test
model_conv = buildModel(net_name,numberOfClasses)


if opt.useDense:
  encoders = Dense_Encoders_Intrinsic(opt)
  decoders = Dense_DecodersIntegralWarper2_Intrinsic(opt)
else:
  encoders = Encoders_Intrinsic(opt)
  decoders = DecodersIntegralWarper2_Intrinsic(opt)

encoders.load_state_dict(torch.load(pathResult_train + 'best_encoders_weights.pth'))
decoders.load_state_dict(torch.load(pathResult_train + 'best_decoders_weights.pth'))
model_conv.load_state_dict(torch.load(pathResult_train + 'best_cnn_weights.pth'))

model_conv.eval()
encoders.eval()
decoders.eval()

model_conv.cuda()
encoders.cuda()
decoders.cuda()

tabella = predictedOnTest(model_conv, encoders, decoders, dataset_real_test, test_transform, opt, dim_to_resize, 
                          criterionRecon, criterionTVWarp, criterionBiasReduce, 
                          criterionSmoothL1, criterionSmoothL2, criterionPBPK)
tabella.to_csv(pathResult_train + 'BestTable.csv', sep=',', index = False)
model_conv.cpu()
encoders.cpu()
decoders.cpu()
del model_conv
del encoders
del decoders

drawPlot(pathResult_train)