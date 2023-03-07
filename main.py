from io import BytesIO
from typing import Optional

from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from io import BytesIO
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from werkzeug.utils import secure_filename
from application.components.bg_remove import BGRemove
import time
import aiofiles
import urllib

from fastapi.middleware.cors import CORSMiddleware



app = FastAPI()
app.mount("/assets", StaticFiles(directory="assets"), name="assets")
ckpt_image = 'application/pretrained/modnet_photographic_portrait_matting.ckpt'
bg_remover = BGRemove(ckpt_image)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class NFTItem(BaseModel):
    url: str
    background: Optional[bool] = True



@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/create_new_nft_image", status_code=201)
async def create_file(nftItem: NFTItem ):
    #async def test_function(dict: dict[str, str]):
    # todo https://pypi.org/project/fastapi-easy-cache/
    background = nftItem.background
    file_url = nftItem.url
    print("file_url = " + file_url)

    #Fixme: assuming arweave & extension is in the url als last query param
    extension = file_url.split("=")[-1]
    if extension not in ("jpg", "jpeg", "png"):
        print(extension)
        raise HTTPException(status_code=422, detail="Image must be jpg or png format!")


    filename = create_filename(file_url, extension)
    print("filename = "+ filename)

    # todo check for network/http errors
    try:
        destination_file_path, headers = urllib.request.urlretrieve(file_url, "assets/input/" + filename)
        #todo switch models MODNet, DeepLabV3, for now just MODNet
        output_filename = bg_remover.image(
        destination_file_path, background=background, output="assets/output/", save=True)
        print("output file" + output_filename)
    except Exception as e:
        print(e)

    #todo return http error code
    if not output_filename:
        return "Something went wrong"

    return {"file_name": filename, "Result": "OK", "path": "/assets/output/" + filename}

@app.post("/upload-file", status_code=201)
async def create_upload_file(background: bool = True, file: UploadFile = File(...)):

    print("filename = ", file.filename) # getting filename
    print(background)
    extension = file.filename.split("=")[-1]
    if extension not in ("jpg", "jpeg", "png"):
        return "Image must be jpg or png format!"
    filename = create_secure_filename(file)

    destination_file_path = "assets/input/"+filename # location to store file
    async with aiofiles.open(destination_file_path, 'wb') as out_file:
        while content := await file.read(1024):  # async read file chunk
            await out_file.write(content)  # async write file chunk

    output_filename = bg_remover.image(
        destination_file_path, background=background, output="assets/output/", save=True)

    return {"file_name": file.filename, "Result": "OK", "path": "/assets/output/"+output_filename}

def read_imagefile(file) -> Image.Image:
    image = Image.open(BytesIO(file))
    return image

def create_secure_filename(file):
    if file:
        ts = time.time()
        filename = f"{str(ts)}-{file.filename}"
        return secure_filename(filename)

def create_filename(file_url: str, extension: str):
    url_path = file_url.split("/")[-1]
    file_name = url_path.split("?")[0]
    if file_name:
        ts = time.time()
        filename = f"{str(ts)}-{file_name}.{extension}"
        return secure_filename(filename)

def download_photo(self, img_url, filename):
    try:
        image_on_web = urllib.urlopen(img_url)
        if image_on_web.headers.maintype == 'image':
            buf = image_on_web.read()
            #path = os.getcwd() + DOWNLOADED_IMAGE_PATH
            #file_path = "%s%s" % (path, filename)
            #downloaded_image = file(file_path, "wb")
            #downloaded_image.write(buf)
            #downloaded_image.close()
            image_on_web.close()
        else:
            return False
    except:
        return False
    return True