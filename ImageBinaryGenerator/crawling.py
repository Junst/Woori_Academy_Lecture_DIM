import os
from google_images_download import google_images_download   #importing the library

# python > 3.8 / pillow : 최신버전 / keras = 2.6.0

response = google_images_download.googleimagesdownload()   #class instantiation

pathDir = os.path.abspath(os.path.curdir)
pathDir += "\chromedriver.exe"
print(pathDir)


arguments = {
    "keywords":"tiger,lion",
    "limit":50,
    "print_urls":True,
    "chromedriver":pathDir,
    "format":"jpg"
}

paths = response.download(arguments)   #passing the arguments to the function
print(paths)   #printing absolute paths of the downloaded images