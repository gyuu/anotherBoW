import os

def get_image_subpaths(top_folder):
	sub_folders = [ f for f in os.listdir(top_folder) if not f.startswith(".")]
	image_paths = []
	for folder in sub_folders:
		folder_path = os.path.join(top_folder, folder)
		for f in [ g for g in os.listdir(folder_path) if g.endswith(".jpg")]:
			image_path = os.path.join(folder, f)
			image_paths.append(image_path)
	return image_paths