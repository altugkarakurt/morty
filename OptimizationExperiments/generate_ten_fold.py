import json
import numpy as np
from sklearn import cross_validation

def get_ten_fold():
	makams = ['Acemasiran', 'Acemkurdi', 'Beyati', 'Bestenigar', 'Hicaz', 'Hicazkar', 'Huseyni', 'Huzzam', 'Karcigar', 'Kurdilihicazkar', 'Mahur', 'Muhayyer', 'Neva', 'Nihavent', 'Rast', 'Saba', 'Segah', 'Sultaniyegah', 'Suzinak', 'Ussak']
	
	all_annots = get_all_annots()

	makam_labels = [j for i in np.arange(1,21,1) for j in (np.ones(50) * i)]
	skf = cross_validation.StratifiedKFold(makam_labels, n_folds=10)

	fold_cnt = 1

	for train_idx, test_idx in skf:
		train_set = []
		test_set = []
		for tr in train_idx:
			train_set.append({'mbid':all_annots[tr]['mbid'], 'makam':all_annots[tr]['makam']})
		for te in test_idx:
			test_set.append({'mbid':all_annots[te]['mbid'], 'makam':all_annots[te]['makam']})
		save_fold({'train':train_set , 'test':test_set}, ('fold_' + str(fold_cnt)))
		fold_cnt+=1

def save_fold(fold, save_name, save_dir='./'):
	with open((save_dir+save_name+'.json'), 'w') as f:
		json.dump(fold, f, indent=2)
		f.close()
		
def get_all_annots(file_name='annotations.json'):
	with open(file_name, 'r') as f:
		all_annots = json.load(f)
		f.close()
	return all_annots

def load_fold(fold_name):
	with open(fold_name, 'r') as f:
		res = json.load(f)
		f.close()
	return res