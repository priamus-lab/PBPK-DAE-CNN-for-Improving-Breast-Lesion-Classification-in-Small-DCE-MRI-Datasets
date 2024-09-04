from Utils import*


parser = argparse.ArgumentParser()
parser.add_argument('--classe', default = 'benign', help='class to consider')
opt = parser.parse_args()


basePath = 'basic_data/' + opt.classe + '/'
savePath = 'FINAL_DATASET/' + opt.classe + '/'
savaPathimg = 'IMG_FIT/' + opt.classe + '/'

os.makedirs(savePath, exist_ok=True)
os.makedirs(savaPathimg, exist_ok=True)

list_files = os.listdir(basePath)

for patient_file in list_files:
  print('Working on ' + patient_file)
  new_data = fitting_mediano(patient_file, basePath, savePath)
  name = patient_file.split('.')[0]
  
  plot_curve(new_data, savaPathimg + patient_file + '.png')
  