from fastapi import FastAPI  
from pydantic import BaseModel
import uvicorn
from app import Vits
from fastapi.middleware.cors import CORSMiddleware
class TextToSpeech(BaseModel):
    txt: str
    emotion: str
    speakerId: int
class SwitchModel(BaseModel):
    configPath: str
    modelPath: str
    infoPath:str
    device:str
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/getModels/{modelPath:path}")
async def getModels(modelPath: str):
    return Vits.getModels(modelPath)

@app.post("/switchs")
async def switch_models(switch_model:SwitchModel):
    configPath = switch_model.configPath
    modelPath = switch_model.modelPath
    infoPath = switch_model.infoPath
    device = switch_model.device
    Vits.loadModels(device,configPath,modelPath,infoPath)
    return {"message": "Config and model switched successfully!"}
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
