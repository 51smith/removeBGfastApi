from io import BytesIO

from PIL import Image
from fastapi import FastAPI,  File, UploadFile
from io import BytesIO
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from werkzeug.utils import secure_filename
from application.components.bg_remove import BGRemove
import time
import aiofiles

app = FastAPI()
app.mount("/assets", StaticFiles(directory="assets"), name="assets")
ckpt_image = 'application/pretrained/modnet_photographic_portrait_matting.ckpt'
bg_remover = BGRemove(ckpt_image)


@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/upload-file", status_code=201)
async def create_upload_file(background: bool = True, file: UploadFile = File(...)):
    print("filename = ", file.filename) # getting filename
    print(background)
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
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