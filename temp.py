# from PIL import Image, ExifTags
import glob
from PIL import Image
import pyexiv2 


for i in glob.glob("data\Jonathan-Saavedra\\" + "*.*"):
    try:
        image=Image.open(i)
        # k = pyexiv2.Image(i)
        # exif = k.read_exif()
        # print(i)
        # print (exif["Exif.Image.Orientation"])

        # if exif["Exif.Image.Orientation"] == 3:
        #     image=image.rotate(180, expand=True)
        # elif exif["Exif.Image.Orientation"] == 6:
        #     image=image.rotate(270, expand=True)
        # elif exif["Exif.Image.Orientation"] == 8:
        #     image=image.rotate(90, expand=True)
        image = image.rotate(90, expand= True)
        image.save(i)
        image.close()
    except (AttributeError, KeyError, IndexError):
        # cases: image don't have getexif
        pass