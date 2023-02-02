import requests
import shutil

class ImageFileOperator():
    def download(self, url: str, imagePath: str):
        res = requests.get(url, stream = True)
        if res.status_code == 200:
            with open(imagePath,'wb') as f:
                shutil.copyfileobj(res.raw, f)
                print('Image sucessfully Downloaded: ', imagePath)
        else:
            raise Exception(f'Cannot download image from {url}')

    def getFileName(self, url: str):
        urlSplitted = url.split('/')
        return urlSplitted[len(urlSplitted)-1]
