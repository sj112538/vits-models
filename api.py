import sys
from fastapi import FastAPI
from pydantic import BaseModel,Extra
import uvicorn
from app import Vits
from fastapi.middleware.cors import CORSMiddleware
import os
class TextToSpeech(BaseModel):
    text: str
    language: int
    noise_scale: float
    noise_scale_w: float
    length_scale: float
    modelName:str
    speakerId:int
    class Config:
        extra = Extra.ignore
class SwitchModels(BaseModel):
    configPath: str
    modelPath: str
    infoPath:str
    device:str
class SwitchModel(BaseModel):
    configPath: str
    modelPath: str
    device:str
    name:str
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/confirm")
async def confirm():
    return True

@app.get("/getModels/{modelPath:path}")
async def getModels(modelPath: str):
    return Vits.getModels(modelPath)

@app.post("/switchs")
async def switch_models(switch_model:SwitchModels):
    configPath = switch_model.configPath
    modelPath = switch_model.modelPath
    infoPath = switch_model.infoPath
    device = switch_model.device
    Vits.loadModels(device,configPath,modelPath,infoPath)
    return {"message": "Config and model switched successfully!"}

@app.post("/switch")
async def switch_models(switch_model:SwitchModel):
    configPath = switch_model.configPath
    modelPath = switch_model.modelPath
    device = switch_model.device
    name = switch_model.name
    Vits.loadModel(device,configPath,modelPath,name)
    return {"message": "Config and model switched successfully!"}

@app.post("/generate")
async def generate(generate:TextToSpeech):
    text = generate.text
    speakerId = generate.speakerId
    lang = generate.language
    name = generate.modelName
    noise_scale = generate.noise_scale
    noise_scale_w = generate.noise_scale_w
    length_scale = generate.length_scale
    audio = Vits.generate(text,lang,speakerId,name,noise_scale,noise_scale_w,length_scale)
    return {"audio": audio.tolist(), "randsample": 22050}

def run(port):
    uvicorn.run(app, host="127.0.0.1", port=port)

if __name__ == '__main__':
    run(sys.argv[1])
