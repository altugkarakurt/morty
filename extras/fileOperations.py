import os

def getFileNamesInDir(dir_name, extension = '.mp3', skip_foldername = ''):
	names = []
	folders = []
	fullnames = []
	print dir_name

	# check if the folder exists
	if not os.path.isdir(dir_name):
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

	print "> Found " + str(len(names)) + " files."
	return fullnames, folders, names
