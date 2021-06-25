import csv
import shutil

# dataset contains all images in one directory, and images.csv describes the classes.
# This script organize the images in 7 folders- by categories.
# Prepare data format for using - "ImageDataGenerator().flow_from_directory"
path = 'dataset path'

with open('images.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        # unclear labels - images under this label has been sorted manually
        if row[2] == "label" or row[2] == "Body" or row[2] == "Not sure" or row[2] == "Other" or row[2] == "Skip":
            continue
        if row[2] == 'Longsleeve' or row[2] == 'Polo' or row[2] == 'Shirt' or row[2] == 'T-Shirt' or row[
            2] == 'Blouse' or row[2] == 'Hoodie':
            full_path = path + '\\' + row[0] + ".jpg"
            full_dest = path + '\\' + "Shirt"
        elif row[2] == 'Outwear' or row[2] == 'Blazer':
            full_path = path + '\\' + row[0] + ".jpg"
            full_dest = path + '\\' + "Coat"
        elif row[2] == 'Dress':
            full_path = path + '\\' + row[0] + ".jpg"
            full_dest = path + '\\' + "Dress"
        elif row[2] == 'Hat':
            full_path = path + '\\' + row[0] + ".jpg"
            full_dest = path + '\\' + "Hat"
        elif row[2] == 'Pants' or row[2] == 'Shorts':
            full_path = path + '\\' + row[0] + ".jpg"
            full_dest = path + '\\' + "Pants"
        elif row[2] == 'Shoes':
            full_path = path + '\\' + row[0] + ".jpg"
            full_dest = path + '\\' + "Shoes"
        elif row[2] == 'Skirt':
            full_path = path + '\\' + row[0] + ".jpg"
            full_dest = path + '\\' + "Skirt"
        elif row[2] == 'Top' or row[2] == 'Undershirt':
            full_path = path + '\\' + row[0] + ".jpg"
            full_dest = path + '\\' + "Top"

        shutil.move(full_path, full_dest)
