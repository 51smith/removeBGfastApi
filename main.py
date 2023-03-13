import json
from typing import Optional
import os
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException, Path
from io import BytesIO
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from werkzeug.utils import secure_filename
from application.components.modnet_bg_remove import MODNetBGRemove
import time
import aiofiles
import aiofiles.os
import urllib

from fastapi.middleware.cors import CORSMiddleware

from dotenv import load_dotenv

load_dotenv()

#TODO set env vars in dockerfile & local

app = FastAPI()
app.mount("/assets", StaticFiles(directory="assets"), name="assets")
ckpt_image = 'application/pretrained/modnet_photographic_portrait_matting.ckpt'
MODNet_bg_remover = MODNetBGRemove(ckpt_image)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class NFTItem(BaseModel):
    url: str
    background: Optional[str] = "background.jpg"
    model: Optional[str] = "MODNet"



@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/get_backgrounds")
async def get_backgrounds():
    files = os.listdir(os.path.abspath("assets/background"))
    filesList = [file for file in files if file.endswith(".jpg")]
    jsonStr = json.dumps(filesList)
    return jsonStr


@app.post("/create_new_nft_image", status_code=201)
async def create_new_nft_image(nftItem: NFTItem ):

    # todo https://pypi.org/project/fastapi-easy-cache/
    background = nftItem.background
    file_url = nftItem.url

    #Fixme: assuming arweave & extension is in the url als last query param
    extension = file_url.split("=")[-1]
    if extension not in ("jpg", "jpeg", "png", "webp"):
        extension="";#fixme, better check for filetype somewhere else
    #    raise HTTPException(status_code=422, detail="Image must be jpg or png format!")


    filename = create_filename(file_url, extension)
    # todo check for network/http errors
    try:
        destination_file_path, headers = urllib.request.urlretrieve(file_url, "assets/input/" + filename)
        #todo switch models MODNet, DeepLabV3, for now just MODNet
        output_filename = MODNet_bg_remover.image(
        destination_file_path, background=background, output="assets/output/", save=True)
        print("output file " + output_filename)
        return {"file_name": output_filename, "Result": "OK", "path": "assets/output/" + output_filename}
    except Exception as e:
        print(e)

    #todo return http error code
    return HTTPException(status_code=500, detail="Something went wrong")




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
        #remove decimal from float, to prevent error in bgremove.safe filename when there is no extension
        filename = f"{str(ts).split('.')[0]}-{file_name}"
        print(filename)
        if extension:
            filename = f"{filename}.{extension}"
        return secure_filename(filename)