# Setup instructions for this project
Create a new conda environment (in anaconda) for fastapi project, for Pyhton 3.8

Active the environment: 
```
conda activate /path/to/your/anaconda3/envs/fastApiProject
```

Install dependencies for this environment:
```
conda install -c conda-forge fastapi
conda install -c conda-forge OpenCV
conda install -c conda-forge python-multipart
conda install pytorch torchvision -c pytorch
conda install aiofiles
conda install werkzeug
conda install python-dotenv
```

start the server:
```
python main.py
```

