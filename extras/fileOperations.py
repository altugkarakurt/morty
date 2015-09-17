import os

def getFileNamesInDir(dir_name, extension = '.mp3', skip_foldername = '', verbose = False):
	names = []
	folders = []
	fullnames = []

	if verbose:
		print dir_name

	# check if the folder exists
	if not os.path.isdir(dir_name):
		if verbose:
			print "> Directory doesn't exist!"
		return [], [], []

	# if the dir_name finishes with the file separator, remove it so os.walk works properly
	dir_name = dir_name[:-1] if dir_name[-1] == os.sep else dir_name

	# walk all the subdirectories
	for (path, dirs, files) in os.walk(dir_name):
		for f in files:
			if f.lower()[-len(extension):] == extension:
				if skip_foldername not in path.split(os.sep)[1:]:
					folders.append(unicode(path, 'utf-8'))
					names.append(unicode(f,'utf-8'))
					fullnames.append(os.path.join(path,f))

	if verbose:
		print "> Found " + str(len(names)) + " files."
	return fullnames, folders, names

def getModeNames(data_dir):
	# check if the folder exists
	if not os.path.isdir(data_dir):
		print "> Directory doesn't exist!"
		return []

	return [x[1] for x in os.walk(data_dir)][0]