from Utils import*

def loader_fc(path):
  x = sio.loadmat(path)
  img = x['image'][:,:,[0,2,4]]
  img = (img - np.min(img))/(np.max(img) - np.min(img))
  assert np.max(img) <= 1
  assert np.min(img) >= 0
  
  roi=x['mask']
  tk_w=x['tk_weiman'][0, [0,2,4]]
  
  etk_w= 0
  tk_p = 0
  etk_p = 0
  
  return (img, roi, tk_w, tk_p, etk_w, etk_p, path)

#creare dataloaser 
class My_DatasetFolder(Dataset):
  def __init__(self, root,  transform, is_valid_file, list_classes, loader):
    self.root = root 
    self.transform = transform
    self.is_valid_file = is_valid_file
    self.list_classes = list_classes
    self.samples = self.__get_samples()
    self.loader = loader

  def __len__(self):
    return len(self.samples)

  def __get_samples(self):
    ListFiles=[]
    for c in self.list_classes:
      listofFiles = os.listdir(self.root + '/' + c)
      for file in listofFiles:
        if self.is_valid_file(self.root + '/' + c + '/' + file):
          ListFiles.append((self.root + '/' + c + '/' + file, self.list_classes.index(c)))   
    return ListFiles

  def __getitem__(self, index):
    filename,classe = self.samples[index]
    (img, roi, tk_w, tk_p, etk_w, etk_p, path) = self.loader(filename)
    if self.transform is not None:
      img, roi = self.transform((img, roi)) #applico la trasformazione sia alla slice che alla maschera
    return img, roi, tk_w, tk_p, etk_w, etk_p, path, classe 


#dataloader per generare un vettore unendo i vettori letti grazie al csv.
class CSV_DatasetFolder(Dataset):
  def __init__(self, root, csv_path, imagePath, exclude_list, transform, include_list, list_classes, valid_class):
    self.root = root 
    self.csv_path = csv_path
    self.imagePath = imagePath
    self.exclude_list = exclude_list
    self.include_list = include_list
    self.list_classes = list_classes
    self.valid_class = valid_class
    self.samples = self.__get_samples()
    self.transform = transform

  def __len__(self):
    return len(self.samples)

  def __get_samples(self):
    ListFiles=[]
    csv_path = os.path.join(self.root, self.csv_path)
    #leggo il csv
    data = pd.read_csv(csv_path)

    if self.include_list == None:
      data['to_consider'] = data.apply(lambda x: (x.patient_albedo_slice.split('_')[0] not in self.exclude_list) and 
                                        (x.patient_warping_slice.split('_')[0] not in self.exclude_list), axis=1)
    else:
      data['to_consider'] = data.apply(lambda x: (x.patient_albedo_slice.split('_')[0] in self.include_list) and
                                       (x.patient_warping_slice.split('_')[0] in self.include_list), axis=1)

    data_to_consider = data[data.to_consider == True]
    albedo_list = data_to_consider.patient_albedo_slice.values
    shading_list = data_to_consider.patient_shading_slice.values
    warp_list = data_to_consider.patient_warping_slice.values

    for i in range(0,data_to_consider.shape[0]):
      albedo_filename = albedo_list[i]
      shading_filename = shading_list[i]
      warping_filename = warp_list[i]
      final_label = self.list_classes.index(self.valid_class)
      ListFiles.append((albedo_filename, shading_filename , warping_filename, final_label))
    return ListFiles

  def __getitem__(self, index: int):
    alb_path, sha_path, warp_path, target = self.samples[index]
    
    alb_image, mask_roi, tk_w, tk_p, etk_w, etk_p, path = loader_fc(self.imagePath + '/' +  self.list_classes[target] + '/' + alb_path)
    
    sha_image = sio.loadmat(self.imagePath + '/' + self.list_classes[target] + '/' + sha_path)['image'][:,:,[0,2,4]]
    sha_image = (sha_image - np.min(sha_image))/(np.max(sha_image) - np.min(sha_image))
    assert np.max(sha_image)<=1
    assert np.min(sha_image)>=0
    
    warp_image = sio.loadmat(self.imagePath + '/' + self.list_classes[target] + '/' + warp_path)['image'][:,:,[0,2,4]]
    warp_image = (warp_image - np.min(warp_image))/(np.max(warp_image) - np.min(warp_image))
    assert np.max(warp_image)<=1
    assert np.min(warp_image)>=0
    
    alb_image, sha_image,  warp_image, mask_roi = self.transform((alb_image, sha_image,  warp_image, mask_roi))
    return alb_image, sha_image, warp_image, mask_roi, tk_w, tk_p, etk_w, etk_p, target

def rangeNormalization(x, supLim, infLim): 
  #normalizzazione nel range
  x_norm = ( (x - np.min(x)) / (np.max(x)- np.min(x)) )*(supLim - infLim) + infLim
  assert np.min(x_norm) >= infLim
  assert np.max(x_norm) <= supLim
  return x_norm

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

 
def imshow_grid(images, batch_size, shape=[2, 8] ):
    fig = plt.figure(1)
    grid = ImageGrid(fig, 111, nrows_ncols=shape, axes_pad=0.05)
    app = shape[0] * shape[1]
    size = app if (app < batch_size) else batch_size
    for i in range(size):
        grid[i].axis('off')
        grid[i].imshow(images[i,:,:,:].cpu().detach().numpy().swapaxes(0,2))  # The AxesGrid object work as a list of axes.
    plt.show()


def buildModel(net_name, numberOfClasses):
  
  if net_name == 'alexnet':
    model_conv = torchvision.models.alexnet(pretrained=True)
    model_conv.classifier[6] = nn.Linear(4096, numberOfClasses)
  
  elif net_name == 'resnet':
    model_conv =  torchvision.models.resnet34(pretrained=True)  
    model_conv.fc = nn.Linear(model_conv.fc.in_features, numberOfClasses)
  
  elif net_name == 'vgg':
    model_conv = torchvision.models.vgg19(pretrained=True)
    model_conv.classifier[6] = nn.Linear(4096, numberOfClasses)
  
  elif net_name == 'mobilenet':
    model_conv = torchvision.models.mobilenet_v2(pretrained=True)
    model_conv.classifier[1] = nn.Linear(1280,numberOfClasses)
    
  for param in model_conv.parameters():
    param.requires_grad = True
  return model_conv


#-----------------------------------------------------------------------------> FUNZIONI DI ADDESTRAMENTO
def step1_DAE_CNN_onReal(model_conv,encoders,decoders, dp0_img, labels, roi, baseg,
                         updator_decoders, updator_encoders, optimizer_conv, optimizer_conv_decoders,
                         dim_to_resize, criterionCNN, totPred_real, totLabels_real, modelLoss_real, modelAcc_real, epoch, save_path):
  
  updator_decoders.zero_grad(set_to_none=True)
  updator_encoders.zero_grad(set_to_none=True)
  optimizer_conv.zero_grad(set_to_none=True)
  optimizer_conv_decoders.zero_grad(set_to_none=True)
  model_conv.zero_grad(set_to_none=True)
  decoders.zero_grad(set_to_none=True)
  encoders.zero_grad(set_to_none=True)
  
  dp0_z, dp0_zS, dp0_zT, dp0_zW = encoders(dp0_img)
  dp0_S, dp0_T, dp0_I, dp0_W, dp0_output, dp0_Wact, masked_out, warpedAlbedo, fakeROI, fakeMaskedOut = decoders(dp0_zS, dp0_zT, dp0_zW, baseg, roi)
  dp0_Ts = F.interpolate(warpedAlbedo, [dim_to_resize,dim_to_resize],mode='bilinear', align_corners=True)
  y = model_conv(dp0_Ts) 
  
  outp, preds = torch.max(y, 1)
  lossCNN = criterionCNN(y, labels)
  
  lossCNN.backward()
  optimizer_conv.step()

  totPred_real = torch.cat((totPred_real, preds.detach().cpu()))
  totLabels_real = torch.cat((totLabels_real, labels.detach().cpu()))
  modelLoss_real += lossCNN.detach().cpu().item() * dp0_img.size(0)
  modelAcc_real += torch.sum(preds.detach().cpu() == labels.detach().cpu()).item()
  return model_conv,encoders,decoders,totPred_real, totLabels_real, modelLoss_real, modelAcc_real

def step2_DAE_onReal(model_conv, encoders,decoders, dp0_img,baseg,params,roi,zeroWarp,
                     updator_decoders, updator_encoders, optimizer_conv, optimizer_conv_decoders,
                     criterionRecon, criterionTVWarp, criterionBiasReduce, criterionSmoothL2,criterionPBPK,
                     train_loss_totale_real, totalLoss_rec_real, totalLoss_pbpk_real, epoch, save_path ):
  
  updator_decoders.zero_grad(set_to_none=True)
  updator_encoders.zero_grad(set_to_none=True)
  optimizer_conv.zero_grad(set_to_none=True)
  optimizer_conv_decoders.zero_grad(set_to_none=True)
  model_conv.zero_grad(set_to_none=True)
  decoders.zero_grad(set_to_none=True)
  encoders.zero_grad(set_to_none=True)
  
  dp0_z, dp0_zS, dp0_zT, dp0_zW = encoders(dp0_img)
  dp0_S, dp0_T, dp0_I, dp0_W, dp0_output, dp0_Wact, masked_out, warpedAlbedo, fakeROI, fakeMaskedOut = decoders(dp0_zS, dp0_zT, dp0_zW, baseg, roi)

  loss_recon = criterionRecon(dp0_output, dp0_img)
  loss_tvw = criterionTVWarp(dp0_W, weight=1e-6)
  loss_br = criterionBiasReduce(dp0_W, zeroWarp, weight=1e-2)
  loss_intr_S = criterionSmoothL2(dp0_S, weight = 1e-6)
  loss_pbpk=criterionPBPK(masked_out, roi, params, weight=1)
      
  loss_all = loss_recon + loss_tvw + loss_br + loss_intr_S + loss_pbpk 
  loss_all.backward()
  updator_decoders.step()
  updator_encoders.step()

  loss_encdec = loss_recon.detach().cpu().item() + loss_br.detach().cpu().item() + loss_tvw.detach().cpu().item() + loss_intr_S.detach().cpu().item() + loss_pbpk.detach().cpu().item()
  train_loss_totale_real += loss_encdec*dp0_img.size(0) 
  totalLoss_rec_real += loss_recon.detach().cpu().item()*dp0_img.size(0)
  totalLoss_pbpk_real+= loss_pbpk.detach().cpu().item()*dp0_img.size(0)
  return model_conv, encoders,decoders, train_loss_totale_real, totalLoss_rec_real, totalLoss_pbpk_real

def step3_Deconder_CNN_onFake(model_conv, encoders,decoders, 
                              updator_decoders, updator_encoders,optimizer_conv, optimizer_conv_decoders,
                              criterionCNN,
                              alb_image,sha_image, warp_image, labels, baseg,mask_roi,dim_to_resize,
                              totPred_fake, totLabels_fake,
                              modelLoss_fake, modelAcc_fake):
  updator_decoders.zero_grad(set_to_none=True)
  updator_encoders.zero_grad(set_to_none=True)
  optimizer_conv.zero_grad(set_to_none=True)
  optimizer_conv_decoders.zero_grad(set_to_none=True)
  model_conv.zero_grad(set_to_none=True)
  decoders.zero_grad(set_to_none=True)
  encoders.zero_grad(set_to_none=True)
  
  _, _, albedo_to_consider, _ = encoders(alb_image)  
  _, shading_to_consider, _, _ = encoders(sha_image)
  _, _, _, warping_to_consider = encoders(warp_image)
  
  dp0_S, dp0_T, dp0_I, dp0_W, dp0_output, dp0_Wact, masked_out, warpedAlbedo, fakeROI, fakeMaskedOut = decoders(shading_to_consider, albedo_to_consider, 
                                                                                                                    warping_to_consider, baseg, mask_roi)
  dp0_Ts = F.interpolate(warpedAlbedo, [dim_to_resize,dim_to_resize],mode='bilinear', align_corners=True)
  y = model_conv(dp0_Ts) 
  outp, preds = torch.max(y, 1)
  lossCNN = criterionCNN(y, labels)
  
  lossCNN.backward()
  optimizer_conv_decoders.step() #solo cnn e decoder

  totPred_fake = torch.cat((totPred_fake, preds.detach().cpu()))
  totLabels_fake = torch.cat((totLabels_fake, labels.detach().cpu()))
  modelLoss_fake += lossCNN.detach().cpu().item() * alb_image.size(0)
  modelAcc_fake += torch.sum( preds.detach().cpu() == labels.detach().cpu()).item()
  return model_conv, encoders,decoders, albedo_to_consider, shading_to_consider, warping_to_consider,totPred_fake, totLabels_fake, modelLoss_fake, modelAcc_fake

def step4_Decoder_onFake(model_conv, encoders,decoders, 
                         optimizer_conv,updator_encoders, updator_decoders, optimizer_conv_decoders,
                         shading_to_consider, albedo_to_consider, warping_to_consider, baseg, mask_roi, params, zeroWarp,
                         criterionTVWarp, criterionBiasReduce, criterionSmoothL2, criterionPBPK, 
                         train_loss_totale_fake, totalLoss_pbpk_fake):
  
  updator_decoders.zero_grad(set_to_none=True)
  updator_encoders.zero_grad(set_to_none=True)
  optimizer_conv.zero_grad(set_to_none=True)
  optimizer_conv_decoders.zero_grad(set_to_none=True)
  model_conv.zero_grad(set_to_none=True)
  decoders.zero_grad(set_to_none=True)
  encoders.zero_grad(set_to_none=True)
  
  dp0_S, dp0_T, dp0_I, dp0_W, dp0_output, dp0_Wact, masked_out, warpedAlbedo, fakeROI, fakeMaskedOut = decoders(shading_to_consider, albedo_to_consider, 
                                                                                                                    warping_to_consider, baseg, mask_roi)
  loss_tvw = criterionTVWarp(dp0_W, weight=1e-6)
  loss_br = criterionBiasReduce(dp0_W, zeroWarp, weight=1e-2)
  loss_intr_S = criterionSmoothL2(dp0_S, weight = 1e-6)
  loss_pbpk=criterionPBPK(fakeMaskedOut, fakeROI, params, weight=1) 

  loss_all =  loss_tvw + loss_br + loss_intr_S + loss_pbpk 
  loss_all.backward()
  updator_decoders.step()

  loss_encdec = loss_br.detach().cpu().item() + loss_tvw.detach().cpu().item() + loss_intr_S.detach().cpu().item() + loss_pbpk.detach().cpu().item()
  train_loss_totale_fake += loss_encdec*albedo_to_consider.size(0) 
  totalLoss_pbpk_fake+= loss_pbpk.detach().cpu().item()*albedo_to_consider.size(0)
  return model_conv, encoders,decoders, train_loss_totale_fake, totalLoss_pbpk_fake

#-----------------------------------------------------------------------------------------------------------------------------> VAL
#in validation i modelli sono in inferenza, quindi possiamo unire step 1 e 2, poi step 3 e 4 
def step1_DAE_CNN_onReal_val(model_conv,encoders,decoders, dp0_img, labels, roi, baseg, params, zeroWarp,
                             dim_to_resize, 
                             criterionCNN, criterionRecon, criterionTVWarp, criterionBiasReduce, criterionSmoothL2,criterionPBPK,
                             totPred_real, totLabels_real, modelLoss_real, modelAcc_real,
                             train_loss_totale_real, totalLoss_rec_real, totalLoss_pbpk_real):

  dp0_z, dp0_zS, dp0_zT, dp0_zW = encoders(dp0_img)
  dp0_S, dp0_T, dp0_I, dp0_W, dp0_output, dp0_Wact, masked_out, warpedAlbedo, fakeROI, fakeMaskedOut = decoders(dp0_zS, dp0_zT, dp0_zW, baseg, roi)
  dp0_Ts = F.interpolate(warpedAlbedo, [dim_to_resize,dim_to_resize],mode='bilinear', align_corners=True)
  y = model_conv(dp0_Ts) 
  
  outp, preds = torch.max(y, 1)
  lossCNN = criterionCNN(y, labels)
  
  totPred_real = torch.cat((totPred_real, preds.detach().cpu()))
  totLabels_real = torch.cat((totLabels_real, labels.detach().cpu()))
  modelLoss_real += lossCNN.detach().cpu().item() * dp0_img.size(0)
  modelAcc_real += torch.sum(preds.detach().cpu() == labels.detach().cpu()).item()
  
  loss_recon = criterionRecon(dp0_output, dp0_img)
  loss_tvw = criterionTVWarp(dp0_W, weight=1e-6)
  loss_br = criterionBiasReduce(dp0_W, zeroWarp, weight=1e-2)
  loss_intr_S = criterionSmoothL2(dp0_S, weight = 1e-6)
  loss_pbpk=criterionPBPK(masked_out, roi, params, weight=1)
      
  loss_encdec = loss_recon.detach().cpu().item() + loss_br.detach().cpu().item() + loss_tvw.detach().cpu().item() + loss_intr_S.detach().cpu().item() + loss_pbpk.detach().cpu().item()

  train_loss_totale_real += loss_encdec*dp0_img.size(0) 
  totalLoss_rec_real += loss_recon.detach().cpu().item()*dp0_img.size(0)
  totalLoss_pbpk_real+= loss_pbpk.detach().cpu().item()*dp0_img.size(0)
 
  return totPred_real, totLabels_real, modelLoss_real, modelAcc_real, train_loss_totale_real, totalLoss_rec_real, totalLoss_pbpk_real


def step3_Deconder_CNN_onFake_val(model_conv, encoders,decoders, params, zeroWarp,
                                  criterionCNN, criterionTVWarp, criterionBiasReduce, criterionSmoothL2, criterionPBPK,
                                  alb_image,sha_image, warp_image, labels, baseg,mask_roi,dim_to_resize,
                                  totPred_fake, totLabels_fake, modelLoss_fake, modelAcc_fake, train_loss_totale_fake, totalLoss_pbpk_fake):

  #self.z, self.zShade, self.zTexture, self.zWarp
  _, _, albedo_to_consider, _ = encoders(alb_image)  
  _, shading_to_consider, _, _ = encoders(sha_image)
  _, _, _, warping_to_consider = encoders(warp_image)
  
  #self.shading, self.texture, self.img, self.resWarping, self.output, self.warping , self.masked_out, self.warpedAlbedo, self.newroi, self.masked_out_fake
  #Input sha, al, warp
  dp0_S, dp0_T, dp0_I, dp0_W, dp0_output, dp0_Wact, masked_out, warpedAlbedo, fakeROI, fakeMaskedOut = decoders(shading_to_consider, albedo_to_consider, 
                                                                                                                    warping_to_consider, baseg, mask_roi)
  dp0_Ts = F.interpolate(warpedAlbedo, [dim_to_resize,dim_to_resize],mode='bilinear', align_corners=True)
  y = model_conv(dp0_Ts) 
  outp, preds = torch.max(y, 1)
  lossCNN = criterionCNN(y, labels)
  
  totPred_fake = torch.cat((totPred_fake, preds.detach().cpu()))
  totLabels_fake = torch.cat((totLabels_fake, labels.detach().cpu()))
  modelLoss_fake += lossCNN.detach().cpu().item() * alb_image.size(0)
  modelAcc_fake += torch.sum(preds.detach().cpu() == labels.detach().cpu()).item()
  
  loss_tvw = criterionTVWarp(dp0_W, weight=1e-6)
  loss_br = criterionBiasReduce(dp0_W, zeroWarp, weight=1e-2)
  loss_intr_S = criterionSmoothL2(dp0_S, weight = 1e-6)
  loss_pbpk=criterionPBPK(fakeMaskedOut, fakeROI, params, weight=1)

  loss_encdec = loss_br.detach().cpu().item() + loss_tvw.detach().cpu().item() + loss_intr_S.detach().cpu().item() + loss_pbpk.detach().cpu().item()

  train_loss_totale_fake += loss_encdec*albedo_to_consider.size(0) 
  totalLoss_pbpk_fake+= loss_pbpk.detach().cpu().item()*albedo_to_consider.size(0)

  return totPred_fake, totLabels_fake, modelLoss_fake, modelAcc_fake, train_loss_totale_fake, totalLoss_pbpk_fake

def computePerformance(modelLoss_real, modelAcc_real,totalSize_real, 
                      totPred_real, totLabels_real ):
  modelLoss_epoch_train = modelLoss_real/totalSize_real
  modelAcc_epoch_train  = modelAcc_real/totalSize_real
    
  totPred_real = totPred_real.numpy()
  totLabels_real = totLabels_real.numpy()
  acc = np.sum((totPred_real == totLabels_real).astype(int))/totalSize_real
    
  x = totLabels_real[np.where(totLabels_real == 1)]
  y = totPred_real[np.where(totLabels_real == 1)]
  acc_1_T = np.sum((x == y).astype(int))/x.shape[0]
    
  x = totLabels_real[np.where(totLabels_real == 0)]
  y = totPred_real[np.where(totLabels_real == 0)]
  acc_0_T = np.sum((x == y).astype(int))/y.shape[0]
  return acc, acc_0_T, acc_1_T, modelLoss_epoch_train, modelAcc_epoch_train


def train_iteration(epochs, opt, model_conv, encoders, decoders, trainSet_real,
                    dataset_benign, dataset_malignant, 
                    TrainLoader_csv_benign, TrainLoader_csv_malignant, 
                    csv_iter_benign, csv_iter_malignant,
                    optimizer_conv, updator_encoders, updator_decoders, optimizer_conv_decoders,
                    loader_opts,
                    criterionCNN, criterionRecon, criterionTVWarp, criterionBiasReduce, criterionSmoothL1, criterionSmoothL2, criterionPBPK,
                    outputPath, dim_to_resize):
                    
  #passo il TrainLoader_csv in modo che si resetta solo se veramente è finito
  totalLoss_rec_real=0.0
  totalLoss_pbpk_real=0.0
  train_loss_totale_real = 0.0
  modelLoss_real = 0.0
  modelAcc_real = 0.0
  totPred_real = torch.empty(0)
  totLabels_real = torch.empty(0)
  totalSize_real = 0

  totalLoss_rec_fake=0.0
  totalLoss_pbpk_fake=0.0
  train_loss_totale_fake = 0.0
  modelLoss_fake = 0.0
  modelAcc_fake = 0.0
  totPred_fake = torch.empty(0)
  totLabels_fake = torch.empty(0)
  totalSize_fake = 0
  gc.collect() # collect garbage

  TrainLoader_realImages = DataLoader(trainSet_real, shuffle=True, **loader_opts)
  real_iter = iter(TrainLoader_realImages)
  
  niter =  len(TrainLoader_realImages)
  start = time.time()
  
  for i in range(0, niter):
    try:
      dp0_img, roi, tk_w, tk_p, etk_w, etk_p, path, labels = real_iter.next()
    except:
      print('!!!!!!!!!  qualcosa non va')
      TrainLoader_realImages = DataLoader(trainSet_real, shuffle=True, **loader_opts)
      real_iter = iter(TrainLoader_realImages)
      dp0_img, roi, tk_w, tk_p, etk_w, etk_p, path, labels = real_iter.next()

    if opt.model=='tk_w':
      params=tk_w     
    elif opt.model=='tk_p':
      params=tk_p
    elif opt.model=='etk_w':
      params=etk_w
    elif opt.model=='etk_p':
      params=etk_p
        
    params = params.type(torch.FloatTensor).cuda()
    dp0_img = dp0_img.type(torch.FloatTensor).cuda()
    roi = roi.type(torch.FloatTensor).cuda()
    labels = labels.cuda()

    totalSize_real += dp0_img.size(0)
    
    baseg = getBaseGrid(N=opt.imgSize, getbatch = True, batchSize = dp0_img.size()[0])
    zeroWarp = torch.cuda.FloatTensor(1, 2, opt.imgSize, opt.imgSize).fill_(0)
    dp0_img, baseg, zeroWarp = setCuda(dp0_img, baseg, zeroWarp)
    dp0_img, = setAsVariable(dp0_img)
    baseg = Variable(baseg, requires_grad=False)
    zeroWarp = Variable(zeroWarp, requires_grad=False)

    #print('step 1')
    model_conv,encoders,decoders,totPred_real, totLabels_real, modelLoss_real, modelAcc_real = step1_DAE_CNN_onReal(model_conv,encoders,decoders,
                                                                                                                      dp0_img, labels, roi, baseg,
                                                                                                                      updator_decoders, 
                                                                                                                      updator_encoders, 
                                                                                                                      optimizer_conv,
                                                                                                                      optimizer_conv_decoders,
                                                                                                                      dim_to_resize, criterionCNN, 
                                                                                                                      totPred_real, totLabels_real, 
                                                                                                                      modelLoss_real, modelAcc_real,
                                                                                                                      epochs, outputPath)


    gc.collect()
    
   
    model_conv, encoders,decoders, train_loss_totale_real, totalLoss_rec_real, totalLoss_pbpk_real = step2_DAE_onReal(model_conv, encoders,decoders, 
                                                                                                                        dp0_img,baseg,params,roi,zeroWarp,
                                                                                                                        updator_decoders, 
                                                                                                                        updator_encoders, 
                                                                                                                        optimizer_conv,
                                                                                                                        optimizer_conv_decoders,
                                                                                                                        criterionRecon, criterionTVWarp,
                                                                                                                        criterionBiasReduce, criterionSmoothL2,
                                                                                                                        criterionPBPK,
                                                                                                                        train_loss_totale_real, 
                                                                                                                        totalLoss_rec_real, 
                                                                                                                        totalLoss_pbpk_real,epochs, 
                                                                                                                        outputPath )
    
    
    gc.collect()
    
    try:
      alb_image_ben, sha_image_ben, warp_image_ben, mask_roi_ben, tk_w_ben, tk_p_ben, etk_w_ben, etk_p_ben, labels_ben = csv_iter_benign.next()
    except:
      print('-------------------------> FAKE image_ben RESTORE')
      TrainLoader_csv_benign = DataLoader(dataset_benign, shuffle=True, batch_size = int(loader_opts['batch_size']/2),  pin_memory = loader_opts['pin_memory'])
      csv_iter_benign = iter(TrainLoader_csv_benign)
      alb_image_ben, sha_image_ben, warp_image_ben, mask_roi_ben, tk_w_ben, tk_p_ben, etk_w_ben, etk_p_ben, labels_ben = csv_iter_benign.next()
      
    if opt.model=='tk_w':
      params_ben=tk_w_ben     
    elif opt.model=='tk_p':
      params_ben=tk_p_ben
    elif opt.model=='etk_w':
      params_ben=etk_w_ben
    elif opt.model=='etk_p':
      params_ben=etk_p_ben
     
    
    
    try:
      alb_image_mal, sha_image_mal, warp_image_mal, mask_roi_mal, tk_w_mal, tk_p_mal, etk_w_mal, etk_p_mal, labels_mal = csv_iter_malignant.next()
    except:
      print('-------------------------> FAKE image_mal RESTORE')
      TrainLoader_csv_malignant = DataLoader(dataset_malignant, shuffle=True, batch_size = int(loader_opts['batch_size']/2), pin_memory = loader_opts['pin_memory'])
      csv_iter_malignant = iter(TrainLoader_csv_malignant)
      alb_image_mal, sha_image_mal, warp_image_mal, mask_roi_mal, tk_w_mal, tk_p_mal, etk_w_mal, etk_p_mal, labels_mal = csv_iter_malignant.next()
      
    if opt.model=='tk_w':
      params_mal=tk_w_mal     
    elif opt.model=='tk_p':
      params_mal=tk_p_mal
    elif opt.model=='etk_w':
      params_mal=etk_w_mal
    elif opt.model=='etk_p':
      params_mal=etk_p_mal
     
    
      
    alb_image = torch.cat((alb_image_ben, alb_image_mal), dim=0).type(torch.FloatTensor).cuda()
    sha_image = torch.cat((sha_image_ben, sha_image_mal), dim=0).type(torch.FloatTensor).cuda()
    warp_image =torch.cat((warp_image_ben, warp_image_mal), dim=0).type(torch.FloatTensor).cuda()
    mask_roi =  torch.cat((mask_roi_ben, mask_roi_mal), dim=0).type(torch.FloatTensor).cuda() 
    params =    torch.cat((params_ben, params_mal), dim=0).type(torch.FloatTensor).cuda() 
    labels =    torch.cat((labels_ben, labels_mal), dim=0).cuda()
    
    totalSize_fake += alb_image.size(0)

    gc.collect()
    baseg = getBaseGrid(N=opt.imgSize, getbatch = True, batchSize = alb_image.size()[0])
    zeroWarp = torch.cuda.FloatTensor(1, 2, opt.imgSize, opt.imgSize).fill_(0)
    alb_image, baseg, zeroWarp = setCuda(alb_image, baseg, zeroWarp)
    alb_image, = setAsVariable(alb_image)
    sha_image, = setAsVariable(sha_image)
    warp_image, = setAsVariable(warp_image)
    baseg = Variable(baseg, requires_grad=False)
    zeroWarp = Variable(zeroWarp, requires_grad=False)
    
    model_conv, encoders,decoders, albedo_to_consider, shading_to_consider, warping_to_consider,totPred_fake, totLabels_fake, modelLoss_fake, modelAcc_fake = step3_Deconder_CNN_onFake(model_conv, encoders,decoders, 
                                                                                                                  updator_decoders, updator_encoders,
                                                                                                                  optimizer_conv,
                                                                                                                  optimizer_conv_decoders,
                                                                                                                  criterionCNN,
                                                                                                                  alb_image,sha_image, warp_image, labels, baseg,
                                                                                                                  mask_roi,dim_to_resize,
                                                                                                                  totPred_fake, totLabels_fake,
                                                                                                                  modelLoss_fake, modelAcc_fake)
                                                                                                                  
    gc.collect() 

  print(' ----------------------> Epoch time')
  end = time.time()
  exec_time = (end-start)
  print(f'Time: {exec_time:.4f}')

  #----------------------------------> 1 real image classification
  acc, acc_0_T, acc_1_T, modelLoss_epoch_train, modelAcc_epoch_train = computePerformance(modelLoss_real, modelAcc_real,totalSize_real, 
                                                                                         totPred_real, totLabels_real )
  with open(outputPath + 'lossTrainClassificationREAL.txt', "a") as file_object:
    file_object.write(str(modelLoss_epoch_train) +'\n')
  with open(outputPath + 'AccTrainClassificationREAL.txt', "a") as file_object:
    file_object.write(str(modelAcc_epoch_train)+'\n')
    
  #----------------------------------> 2 real image DAE
  dae_total_loss_real = train_loss_totale_real/totalSize_real
  dae_rec_loss_real = totalLoss_rec_real/totalSize_real
  dae_pbpk_loss_real = totalLoss_pbpk_real/totalSize_real
  
  with open(outputPath + 'DAE_Total_real.txt', "a") as file_object:
    file_object.write(str(dae_total_loss_real) +'\n')
  with open(outputPath + 'DAE_Rec_real.txt', "a") as file_object:
    file_object.write(str(dae_rec_loss_real)+'\n')
  with open(outputPath + 'DAE_Pbpk_real.txt', "a") as file_object:
    file_object.write(str(dae_pbpk_loss_real)+'\n')
    
  #----------------------------------> 3 fake classification
  acc_fake, acc_0_T_fake, acc_1_T_fake, modelLoss_epoch_train_fake, modelAcc_epoch_train_fake = computePerformance(modelLoss_fake, modelAcc_fake,totalSize_fake, 
                                                                                                                  totPred_fake, totLabels_fake )
  with open(outputPath + 'lossTrainClassificationFAKE.txt', "a") as file_object:
    file_object.write(str(modelLoss_epoch_train_fake) +'\n')
  with open(outputPath + 'AccTrainClassificationFAKE.txt', "a") as file_object:
    file_object.write(str(modelAcc_epoch_train_fake)+'\n')
    
 
  print('[Epoch %d]--->' %(epochs))
  print('  [TRAIN]')
  #1
  print('      [Real Classification: %d [Loss: %.4f - ACC_T: %.4f - ACC_0: %.4f - ACC_1: %.4f]]' %(totalSize_real, modelLoss_epoch_train, modelAcc_epoch_train, acc_0_T, acc_1_T))
  #2
  print('      [Real DAE [Total: %.4f - Rec: %.4f - PBPK: %.4f]' %(dae_total_loss_real, dae_rec_loss_real, dae_pbpk_loss_real))
  #3
  print('      [Fake Classification: %d [Loss: %.4f - ACC_T: %.4f - ACC_0: %.4f - ACC_1: %.4f]]' %(totalSize_fake, modelLoss_epoch_train_fake, modelAcc_epoch_train_fake, 
                                                                                                   acc_0_T_fake, acc_1_T_fake))

  torch.save(model_conv.state_dict(), outputPath + 'cnn_weights.pth')
  torch.save(encoders.state_dict(), outputPath + 'encoders_weights.pth')
  torch.save(decoders.state_dict(), outputPath + 'decoders_weights.pth')
  return model_conv, encoders, decoders, TrainLoader_csv_benign, TrainLoader_csv_malignant, csv_iter_benign, csv_iter_malignant


def validate(epochs, opt, model_conv, encoders, decoders, trainSet_real,trainSet_csv,
                    loader_opts,
                    criterionCNN, criterionRecon, criterionTVWarp, criterionBiasReduce, criterionSmoothL1, criterionSmoothL2, criterionPBPK,
                    outputPath, dim_to_resize):
  
  totalLoss_rec_real=0.0
  totalLoss_pbpk_real=0.0
  train_loss_totale_real = 0.0
  modelLoss_real = 0.0
  modelAcc_real = 0.0
  totPred_real = torch.empty(0)
  totLabels_real = torch.empty(0)
  totalSize_real = 0

  totalLoss_rec_fake=0.0
  totalLoss_pbpk_fake=0.0
  train_loss_totale_fake = 0.0
  modelLoss_fake = 0.0
  modelAcc_fake = 0.0
  totPred_fake = torch.empty(0)
  totLabels_fake = torch.empty(0)
  totalSize_fake = 0
  gc.collect() # collect garbage

  TrainLoader_realImages = DataLoader(trainSet_real, shuffle=True, **loader_opts)
  TrainLoader_csv = DataLoader(trainSet_csv, shuffle=True, **loader_opts)

  for dp0_img, roi, tk_w, tk_p, etk_w, etk_p, path, labels in TrainLoader_realImages:
    
    if opt.model=='tk_w':
      params=tk_w     
    elif opt.model=='tk_p':
      params=tk_p
    elif opt.model=='etk_w':
      params=etk_w
    elif opt.model=='etk_p':
      params=etk_p
        
    params = params.type(torch.FloatTensor).cuda()
    dp0_img = dp0_img.type(torch.FloatTensor).cuda()
    roi = roi.type(torch.FloatTensor).cuda()
    labels = labels.cuda()

    totalSize_real += dp0_img.size(0)

    gc.collect()
    baseg = getBaseGrid(N=opt.imgSize, getbatch = True, batchSize = dp0_img.size()[0])
    zeroWarp = torch.cuda.FloatTensor(1, 2, opt.imgSize, opt.imgSize).fill_(0)
    dp0_img, baseg, zeroWarp = setCuda(dp0_img, baseg, zeroWarp)
    dp0_img, = setAsVariable(dp0_img)
    baseg = Variable(baseg, requires_grad=False)
    zeroWarp = Variable(zeroWarp, requires_grad=False)
    
    
    totPred_real, totLabels_real, modelLoss_real, modelAcc_real, train_loss_totale_real, totalLoss_rec_real, totalLoss_pbpk_real = step1_DAE_CNN_onReal_val(model_conv,encoders,decoders, 
                                                                                                                                                            dp0_img, labels, roi, baseg, params, zeroWarp, 
                                                                                                                                                            dim_to_resize, 
                                                                                                                                                            criterionCNN,
                                                                                                                                                            criterionRecon, 
                                                                                                                                                            criterionTVWarp, 
                                                                                                                                                            criterionBiasReduce, 
                                                                                                                                                            criterionSmoothL2,
                                                                                                                                                            criterionPBPK,
                                                                                                                                                            totPred_real, 
                                                                                                                                                            totLabels_real, 
                                                                                                                                                            modelLoss_real, 
                                                                                                                                                            modelAcc_real,
                                                                                                                                                            train_loss_totale_real, 
                                                                                                                                                            totalLoss_rec_real, 
                                                                                                                                                            totalLoss_pbpk_real)
                                                                                                                                                            
    gc.collect()

        
                
  #statistiche sul train
  #real image classification
  acc, acc_0_T, acc_1_T, modelLoss_epoch_train, modelAcc_epoch_train = computePerformance(modelLoss_real, modelAcc_real,totalSize_real, 
                                                                                         totPred_real, totLabels_real )
  with open(outputPath + 'lossVALClassificationREAL.txt', "a") as file_object:
    file_object.write(str(modelLoss_epoch_train) +'\n')
  with open(outputPath + 'AccVALClassificationREAL.txt', "a") as file_object:
    file_object.write(str(modelAcc_epoch_train)+'\n')

  #real image DAE
  #train_loss_totale_real, totalLoss_rec_real, totalLoss_pbpk_real
  dae_total_loss_real = train_loss_totale_real/totalSize_real
  dae_rec_loss_real = totalLoss_rec_real/totalSize_real
  dae_pbpk_loss_real = totalLoss_pbpk_real/totalSize_real
  
  with open(outputPath + 'DAE_Total_real_val.txt', "a") as file_object:
    file_object.write(str(dae_total_loss_real) +'\n')
  with open(outputPath + 'DAE_Rec_real_val.txt', "a") as file_object:
    file_object.write(str(dae_rec_loss_real)+'\n')
  with open(outputPath + 'DAE_Pbpk_real_val.txt', "a") as file_object:
    file_object.write(str(dae_pbpk_loss_real)+'\n')

  mean_accuracy_val = (acc_1_T + acc_0_T)/2
  
  for alb_image, sha_image, warp_image, mask_roi, tk_w, tk_p, etk_w, etk_p, labels in TrainLoader_csv:
    gc.collect()
    
    if opt.model=='tk_w':
      params=tk_w     
    elif opt.model=='tk_p':
      params=tk_p
    elif opt.model=='etk_w':
      params=etk_w
    elif opt.model=='etk_p':
      params=etk_p
      
    alb_image = alb_image.type(torch.FloatTensor).cuda()
    sha_image = sha_image.type(torch.FloatTensor).cuda()
    warp_image = warp_image.type(torch.FloatTensor).cuda()
    mask_roi = mask_roi.type(torch.FloatTensor).cuda()
    params = params.type(torch.FloatTensor).cuda()
    labels = labels.cuda()

    totalSize_fake += alb_image.size(0)

    gc.collect()
    baseg = getBaseGrid(N=opt.imgSize, getbatch = True, batchSize = alb_image.size()[0])
    zeroWarp = torch.cuda.FloatTensor(1, 2, opt.imgSize, opt.imgSize).fill_(0)
    alb_image, baseg, zeroWarp = setCuda(alb_image, baseg, zeroWarp)
    alb_image, = setAsVariable(alb_image)
    sha_image, = setAsVariable(sha_image)
    warp_image, = setAsVariable(warp_image)
    baseg = Variable(baseg, requires_grad=False)
    zeroWarp = Variable(zeroWarp, requires_grad=False)
    
    totPred_fake, totLabels_fake, modelLoss_fake, modelAcc_fake, train_loss_totale_fake, totalLoss_pbpk_fake=  step3_Deconder_CNN_onFake_val(model_conv, encoders,decoders, 
                                                                                                                                             params, zeroWarp,
                                                                                                                                             criterionCNN, 
                                                                                                                                             criterionTVWarp, 
                                                                                                                                             criterionBiasReduce, 
                                                                                                                                             criterionSmoothL2, 
                                                                                                                                             criterionPBPK,
                                                                                                                                             alb_image,sha_image, 
                                                                                                                                             warp_image, labels, 
                                                                                                                                             baseg,mask_roi,dim_to_resize,
                                                                                                                                             totPred_fake, totLabels_fake, 
                                                                                                                                             modelLoss_fake, modelAcc_fake, 
                                                                                                                                             train_loss_totale_fake, totalLoss_pbpk_fake)
    
  #fake image classification

  acc_fake, acc_0_T_fake, acc_1_T_fake, modelLoss_epoch_train_fake, modelAcc_epoch_train_fake = computePerformance(modelLoss_fake, modelAcc_fake,totalSize_fake, 
                                                                                                                  totPred_fake, totLabels_fake )
  with open(outputPath + 'lossTrainClassificationFAKE_val.txt', "a") as file_object:
    file_object.write(str(modelLoss_epoch_train_fake) +'\n')
  with open(outputPath + 'AccTrainClassificationFAKE_val.txt', "a") as file_object:
    file_object.write(str(modelAcc_epoch_train_fake)+'\n')

  #train_loss_totale_fake, totalLoss_rec_fake, totalLoss_pbpk_fake
  dae_total_loss_fake = train_loss_totale_fake/totalSize_fake
  dae_pbpk_loss_fake= totalLoss_pbpk_fake/totalSize_fake

  with open(outputPath + 'DAE_Total_fake_val.txt', "a") as file_object:
    file_object.write(str(dae_total_loss_fake) +'\n')
  with open(outputPath + 'DAE_Pbpk_fake_val.txt', "a") as file_object:
    file_object.write(str(dae_pbpk_loss_fake)+'\n')
  
  
  
  print('      [Real Classification: %d [Loss: %.4f - ACC_T: %.4f - ACC_0: %.4f - ACC_1: %.4f - AvgAcc: %.4f]]' %(totalSize_real, modelLoss_epoch_train, modelAcc_epoch_train, 
                                                                                                                   acc_0_T, acc_1_T, mean_accuracy_val))
  
  print('      [Real DAE [Total: %.4f - Rec: %.4f - PBPK: %.4f]' %(dae_total_loss_real, dae_rec_loss_real, dae_pbpk_loss_real))
  
 
  print('      [Fake Classification: %d [Loss: %.4f - ACC_T: %.4f - ACC_0: %.4f - ACC_1: %.4f]]' %(totalSize_fake, modelLoss_epoch_train_fake, modelAcc_epoch_train_fake, 
                                                                                                   acc_0_T_fake, acc_1_T_fake))
  print('      [Fake DAE [Total: %.4f  - PBPK: %.4f]' %(dae_total_loss_fake, dae_pbpk_loss_fake)) #STEP 3
  
  return modelLoss_epoch_train, modelAcc_epoch_train, acc_0_T, acc_1_T, mean_accuracy_val 


def validateOnTest(epochs, opt, model_conv, encoders, decoders, trainSet_real,
                    loader_opts,
                    criterionCNN, criterionRecon, criterionTVWarp, criterionBiasReduce, criterionSmoothL1, criterionSmoothL2, criterionPBPK,
                    outputPath, dim_to_resize):
  
  totalLoss_rec_real=0.0
  totalLoss_pbpk_real=0.0
  train_loss_totale_real = 0.0
  modelLoss_real = 0.0
  modelAcc_real = 0.0
  totPred_real = torch.empty(0)
  totLabels_real = torch.empty(0)
  totalSize_real = 0

  totalLoss_rec_fake=0.0
  totalLoss_pbpk_fake=0.0
  train_loss_totale_fake = 0.0
  modelLoss_fake = 0.0
  modelAcc_fake = 0.0
  totPred_fake = torch.empty(0)
  totLabels_fake = torch.empty(0)
  totalSize_fake = 0
  gc.collect() # collect garbage

  TrainLoader_realImages = DataLoader(trainSet_real, shuffle=True, **loader_opts)

  niter = len(TrainLoader_realImages)
  real_iter = iter(TrainLoader_realImages)

  for dp0_img, roi, tk_w, tk_p, etk_w, etk_p, path, labels in TrainLoader_realImages:
  
    if opt.model=='tk_w':
      params=tk_w     
    elif opt.model=='tk_p':
      params=tk_p
    elif opt.model=='etk_w':
      params=etk_w
    elif opt.model=='etk_p':
      params=etk_p
    
    
    params = params.type(torch.FloatTensor).cuda()
    dp0_img = dp0_img.type(torch.FloatTensor).cuda()
    roi = roi.type(torch.FloatTensor).cuda()
    labels = labels.cuda()
    totalSize_real += dp0_img.size(0)

    baseg = getBaseGrid(N=opt.imgSize, getbatch = True, batchSize = dp0_img.size()[0])
    zeroWarp = torch.cuda.FloatTensor(1, 2, opt.imgSize, opt.imgSize).fill_(0)
    dp0_img, baseg, zeroWarp = setCuda(dp0_img, baseg, zeroWarp)
    dp0_img, = setAsVariable(dp0_img)
    baseg = Variable(baseg, requires_grad=False)
    zeroWarp = Variable(zeroWarp, requires_grad=False)
    
    
    totPred_real, totLabels_real, modelLoss_real, modelAcc_real, train_loss_totale_real, totalLoss_rec_real, totalLoss_pbpk_real = step1_DAE_CNN_onReal_val(model_conv,encoders,decoders, 
                                                                                                                                                            dp0_img, labels, roi, baseg, params, zeroWarp, 
                                                                                                                                                            dim_to_resize, 
                                                                                                                                                            criterionCNN,
                                                                                                                                                            criterionRecon, 
                                                                                                                                                            criterionTVWarp, 
                                                                                                                                                            criterionBiasReduce, 
                                                                                                                                                            criterionSmoothL2,
                                                                                                                                                            criterionPBPK,
                                                                                                                                                            totPred_real, 
                                                                                                                                                            totLabels_real, 
                                                                                                                                                            modelLoss_real, 
                                                                                                                                                            modelAcc_real,
                                                                                                                                                            train_loss_totale_real, 
                                                                                                                                                            totalLoss_rec_real, 
                                                                                                                                                            totalLoss_pbpk_real)
    gc.collect()                                                                                                                            
  
  #statistiche 
  #real image classification
  acc, acc_0_T, acc_1_T, modelLoss_epoch_train, modelAcc_epoch_train = computePerformance(modelLoss_real, modelAcc_real,totalSize_real, 
                                                                                         totPred_real, totLabels_real )
  #real image DAE
  #train_loss_totale_real, totalLoss_rec_real, totalLoss_pbpk_real
  dae_total_loss_real = train_loss_totale_real/totalSize_real
  dae_rec_loss_real = totalLoss_rec_real/totalSize_real
  dae_pbpk_loss_real = totalLoss_pbpk_real/totalSize_real
  
  mean_accuracy_val = (acc_1_T + acc_0_T)/2
  print('      [Real Classification: %d [Loss: %.4f - ACC_T: %.4f - ACC_0: %.4f - ACC_1: %.4f - AvgAcc: %.4f]]' %(totalSize_real, modelLoss_epoch_train, modelAcc_epoch_train, 
                                                                                                                   acc_0_T, acc_1_T, mean_accuracy_val))
  print('      [Real DAE [Total: %.4f - Rec: %.4f - PBPK: %.4f]' %(dae_total_loss_real, dae_rec_loss_real, dae_pbpk_loss_real))



def train_loop_validation(model_conv, encoders, decoders, dim_to_resize,
                          trainSet_real, dataset_benign, dataset_malignant, valSet_real, valSet_csv, testSet_real,
                          optimizer_conv, updator_encoders, updator_decoders, optimizer_conv_decoders,
                          num_epoch, loader_opts, start,
                          criterionCNN, criterionRecon, criterionTVWarp, criterionBiasReduce, criterionSmoothL1, criterionSmoothL2, criterionPBPK,
                          best_acc, best_acc_m, best_loss, best_epoca, outputPath, opt, to_validate):
  
  TrainLoader_csv_benign = DataLoader(dataset_benign, shuffle=True, batch_size = int(loader_opts['batch_size']/2),  pin_memory = loader_opts['pin_memory'])
  TrainLoader_csv_malignant = DataLoader(dataset_malignant, shuffle=True, batch_size = int(loader_opts['batch_size']/2), pin_memory = loader_opts['pin_memory'])
  
  csv_iter_benign = iter(TrainLoader_csv_benign)
  csv_iter_malignant = iter(TrainLoader_csv_malignant)
  
  for epochs in range(start, num_epoch + 1):
    model_conv.train()                  
    encoders.train()
    decoders.train()
    model_conv, encoders, decoders, TrainLoader_csv_benign, TrainLoader_csv_malignant, csv_iter_benign, csv_iter_malignant= train_iteration(epochs, opt, model_conv, encoders, decoders, 
                                                                                                                            trainSet_real,dataset_benign, dataset_malignant,  
                                                                                                                            TrainLoader_csv_benign, TrainLoader_csv_malignant, 
                                                                                                                            csv_iter_benign, csv_iter_malignant,
                                                                                                                            optimizer_conv, updator_encoders, 
                                                                                                                            updator_decoders, optimizer_conv_decoders,
                                                                                                                            loader_opts,
                                                                                                                            criterionCNN, criterionRecon, criterionTVWarp, 
                                                                                                                            criterionBiasReduce, criterionSmoothL1, criterionSmoothL2, 
                                                                                                                            criterionPBPK,
                                                                                                                            outputPath, dim_to_resize)
    
    model_conv.eval()
    encoders.eval()
    decoders.eval()
    
    if to_validate:
      print('  [VALIDATE]')
      modelLoss_epoch_val, modelAcc_epoch_val, acc_0_T, acc_1_T, mean_accuracy_val = validate(epochs, opt, model_conv, encoders, decoders, valSet_real,valSet_csv,
                                                                                              loader_opts,
                                                                                              criterionCNN, criterionRecon, criterionTVWarp, 
                                                                                              criterionBiasReduce, criterionSmoothL1, criterionSmoothL2, 
                                                                                              criterionPBPK,outputPath, dim_to_resize)
                                                                                              
      if (epochs == 1 or (modelLoss_epoch_val<best_loss)):  #or (mean_accuracy_val >= best_acc_m) or (modelLoss_epoch_val<best_loss)):   
        print('     .... Saving best weights ....')
        best_acc = modelAcc_epoch_val
        best_acc_m = mean_accuracy_val
        best_loss = modelLoss_epoch_val
        best_epoca = epochs

        torch.save(model_conv.state_dict(), outputPath + 'best_cnn_weights.pth')
        torch.save(encoders.state_dict(), outputPath + 'best_encoders_weights.pth')
        torch.save(decoders.state_dict(), outputPath + 'best_decoders_weights.pth')

        print('  [     TEST]')
        validateOnTest(epochs, opt, model_conv, encoders, decoders, testSet_real,
                      loader_opts,
                      criterionCNN, criterionRecon, criterionTVWarp, criterionBiasReduce, criterionSmoothL1, criterionSmoothL2, criterionPBPK,
                      outputPath, dim_to_resize)
      
      sio.savemat(outputPath + 'check_point.mat', {'best_acc': best_acc, 
                                                   'best_acc_m':best_acc_m,
                                                   'best_loss': best_loss,
                                                   'best_epoca': best_epoca,
                                                   'last_epoch': epochs})
    
    else:
      print('  [     TEST]')
      validateOnTest(epochs, opt, model_conv, encoders, decoders, testSet_real,
                      loader_opts,
                      criterionCNN, criterionRecon, criterionTVWarp, criterionBiasReduce, criterionSmoothL1, criterionSmoothL2, criterionPBPK,
                      outputPath, dim_to_resize)
    


def predictedOnTest(model_conv, encoders, decoders, testSet, transform, opt, dim_to_resize, 
                    criterionRecon, criterionTVWarp, criterionBiasReduce, criterionSmoothL1, criterionSmoothL2, criterionPBPK):
  Tabella_predizioni = pd.DataFrame()
  func = nn.Softmax(dim=1)
  testFiles = testSet.samples
  
  for path, label in testFiles:
    dp0_img, roi, tk_w, tk_p, etk_w, etk_p, path = loader_fc(path)

    pixel = np.sum(roi)
    dp0_img, roi = transform((dp0_img,roi))
    
    if opt.model=='tk_w':
      params=tk_w     
    elif opt.model=='tk_p':
      params=tk_p
    elif opt.model=='etk_w':
      params=etk_w
    elif opt.model=='etk_p':
      params=etk_p
    

    dp0_img = dp0_img.type(torch.FloatTensor).unsqueeze(0).cuda()
    roi = roi.type(torch.FloatTensor).unsqueeze(0).cuda()
    params = torch.from_numpy(params).type(torch.FloatTensor).unsqueeze(0).cuda()

    baseg = getBaseGrid(N=opt.imgSize, getbatch = True, batchSize = dp0_img.size()[0])
    zeroWarp = torch.cuda.FloatTensor(1, 2, opt.imgSize, opt.imgSize).fill_(0)
    dp0_img, baseg, zeroWarp = setCuda(dp0_img, baseg, zeroWarp)
    dp0_img, = setAsVariable(dp0_img)
    baseg = Variable(baseg, requires_grad=False)
    zeroWarp = Variable(zeroWarp, requires_grad=False)


    dp0_z, dp0_zS, dp0_zT, dp0_zW = encoders(dp0_img)
    dp0_S, dp0_T, dp0_I, dp0_W, dp0_output, dp0_Wact, masked_out, warpedAlbedo, fakeROI, fakeMaskedOut = decoders(dp0_zS, dp0_zT, dp0_zW, baseg, roi)
    dp0_Ts = F.interpolate(warpedAlbedo, [dim_to_resize,dim_to_resize],mode='bilinear', align_corners=True)
    
    y = model_conv(dp0_Ts).detach().cpu() 
    outp, preds = torch.max(y, 1)
    y = func(y)


    loss_recon = criterionRecon(dp0_output, dp0_img)
    loss_tvw = criterionTVWarp(dp0_W, weight=1e-6)
    loss_br = criterionBiasReduce(dp0_W, zeroWarp, weight=1e-2)
    loss_intr_S = criterionSmoothL2(dp0_S, weight = 1e-6)
    loss_pbpk = criterionPBPK(masked_out, roi, params, weight=1)

    loss_all = loss_recon.detach().cpu()  + loss_tvw.detach().cpu()  + loss_br.detach().cpu()  + loss_intr_S.detach().cpu()  + loss_pbpk.detach().cpu()  

    prob_0 =y[0][0]
    prob_1= y[0][1]

    name_f = path.split('/')[-1]  #prendo il nome del file
    patient = name_f.split('_')[0] #predo il nome del paziente
    lesione = name_f.split('_')[0] + '_' + name_f.split('_')[1]
    
    data = {'filename': name_f,
            'patient': patient,
            'lesione': lesione,
            'dim': pixel,
            'prob0': prob_0.item(),
            'prob1': prob_1.item(),
            'predicted': preds.item(),
            'true_class': label,
            'loss_recon': loss_recon.item(),
            'loss_tvw': loss_tvw.item(),
            'loss_br': loss_br.item(),
            'loss_intr_S': loss_intr_S.item(),
            'loss_pbpk': loss_pbpk.item(),
            'loss_all': loss_all.item()
            }

    

    Tabella_predizioni = pd.concat((Tabella_predizioni, pd.DataFrame( data = data, index=[0])), ignore_index=True)
  return Tabella_predizioni 


def save_images(save_path:str, epoch: int, rec_tens: torch.Tensor, warped_albedo_tens: torch.Tensor, step: str, folder_name: str = 'rec_warpalb_images', img_to_save: int = 1):
  save_path_complete = os.path.join(save_path, folder_name)
  os.makedirs(save_path_complete, exist_ok=True)
  epoch_str: str = str(epoch).zfill(3)

  rec_out_tens = rec_tens.cpu().detach().permute(0,2,3,1).numpy()
  rec_out_tens = rec_out_tens[0:img_to_save, :, :, :]

  for i in range(0, rec_out_tens.shape[0], 1 ):
    rec_out_tens_el = rec_out_tens[i, : , :, :]
    plt.imshow(rec_out_tens_el) 
    rec_filename = f'{epoch_str}) {i}_sample_step_{step}_rec_out.png'
    rec_path = os.path.join(save_path_complete, rec_filename)
    plt.savefig(rec_path)

  warped_albedo_out_tens = warped_albedo_tens.cpu().detach().permute(0,2,3,1).numpy()
  warped_albedo_out_tens = warped_albedo_out_tens[0:img_to_save, :, :, :]

  for i in range(0, warped_albedo_out_tens.shape[0], 1 ):
    warped_albedo_out_tens_el = warped_albedo_out_tens[i, : , :, :]
    plt.imshow(warped_albedo_out_tens_el) 
    warped_albedo_filename = f'{epoch_str}) {i}_sample_step_{step}_warped_albedo.png'
    warped_albedo_path = os.path.join(save_path_complete, warped_albedo_filename)
    plt.savefig(warped_albedo_path)

#------------------------------------------------------------------------------------------------------------TRANSFORMAZIONI 
class Random_Horizontal_Flip_both(object):
  def __init__(self, p=0.5):
    self.p = p                           #probabilità di effettuare il flip

  def __call__(self, tupla):
    
    if random() < self.p:
      num = len(tupla)
      tupla = list(tupla)
      for i in range(0, num):
        tupla[i] = torch.flip(tupla[i], dims=[2])
      tupla = tuple(tupla)

    return tupla
    
  def __repr__(self):
    return self.__class__.__name__ + '(p={})'.format(self.p)


#------------------------------ Random Vertical flip
class Random_Vertical_Flip_both(object):
  def __init__(self, p=0.5):
    self.p = p                           #probabilità di effettuare il flip

  def __call__(self, tupla):

    if random() < self.p:
      num = len(tupla)
      tupla = list(tupla)
      for i in range(0, num):
        tupla[i] = torch.flip(tupla[i], dims=[1])
      tupla = tuple(tupla)

    return tupla
    
  def __repr__(self):
    return self.__class__.__name__ + '(p={})'.format(self.p)

#------------------------------ Rotazione n*90
class Random_Rotation_both(object):
  def __init__(self, p=0.5, n=1):
    self.p = p                        #probabilità di effettuare la rotazione
    self.n = n                        #numero di 90 gradi                    

  def __call__(self, tupla):

    if random() < self.p:
      num = len(tupla)
      tupla = list(tupla)
      for i in range(0, num):
        tupla[i] = torch.rot90(tupla[i],self.n,dims=[1,2])
      tupla = tuple(tupla)

    return tupla
    
  def __repr__(self):
    return self.__class__.__name__ + '(p={}, n={})'.format(self.p, self.n)

#------------------------------
class ToTensor_both(object):
  def __init__(self):
    self.info = 'tensor'

  def __call__(self, tupla):

    num = len(tupla)
    tupla = list(tupla)
    for i in range(0, num):
      tupla[i] =  transforms.ToTensor()(tupla[i])
    tupla = tuple(tupla)

    return tupla

  def __repr__(self):
    return self.__class__.__name__ + '()'  
    

def drawPlot(outputPath):
  plot_figure('AccTrainClassificationREAL.txt', 'AccVALClassificationREAL.txt', 'Training Vs Validation Accuracies', 'Accuracy', 'ACC_Train_val.png',outputPath)
  plot_figure('lossTrainClassificationREAL.txt', 'lossVALClassificationREAL.txt', 'Model: Training Vs Validation Losses', 'Loss', 'Loss_Train_val.png',outputPath)
  plot_figure('DAE_Pbpk_real.txt', 'DAE_Pbpk_real_val.txt', 'PBPK Losses', 'PBPK', 'PBPK_Train_val.png',outputPath)
  plot_figure('DAE_Rec_real.txt', 'DAE_Rec_real_val.txt', 'Reconstraction Loss', 'Reconstraction', 'Recon_Train_val.png',outputPath)
  
 
  
def plot_figure(file1, file2, title, metric, filename,outputPath):
  lossModel_Train = []
  lossModel_val = []
  
  accModel_Train = []
  accModel_val = []


  file = open(outputPath + file1, 'r')
  Testo = file.readlines()
  for element in Testo:
    lossModel_Train.append(float(element))

  file = open(outputPath + file2, 'r')
  Testo = file.readlines()
  for element in Testo:
    lossModel_val.append(float(element))

  plt.figure()
  plt.title(title)
  plt.xlabel('Epoch')
  plt.ylabel(metric)
  plt.plot(list(range(1,len(lossModel_Train)+1)), lossModel_Train, color='r', label="Train")  
  plt.plot(list(range(1, len(lossModel_val)+1)), lossModel_val, color='g', label="Validation")
  plt.legend()
  plt.savefig(outputPath + filename)
