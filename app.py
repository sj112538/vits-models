# coding=utf-8 
import os
import re

import torch
import utils
import commons
import json
import psutil as ps
from models import SynthesizerTrn
from text import text_to_sequence, _clean_text
from torch import no_grad, LongTensor
limitation = os.getenv("SYSTEM") == "spaces"  # limit text and audio length in huggingface spaces
class Vits:
  hps_ms = None
  device = None
  models = {}
  @classmethod
  def get_text(self,text, hps, is_symbol):
      text_norm, clean_text = text_to_sequence(text, hps.symbols, [] if is_symbol else hps.data.text_cleaners)
      if hps.data.add_blank:
          text_norm = commons.intersperse(text_norm, 0)
      text_norm = LongTensor(text_norm)
      return text_norm, clean_text
  @classmethod
  def create_tts_fn(self,net_g_ms, speaker_id):
      def tts_fn(text, language, noise_scale, noise_scale_w, length_scale, is_symbol):
          text = text.replace('\n', ' ').replace('\r', '').replace(" ", "")
          if limitation:
              text_len = len(re.sub("\[([A-Z]{2})\]", "", text))
              max_len = 100
              if is_symbol:
                  max_len *= 3
              if text_len > max_len:
                  return "Error: Text is too long", None
          if not is_symbol:
              if language == 0:
                  text = f"[ZH]{text}[ZH]"
              elif language == 1:
                  text = f"[JA]{text}[JA]"
              else:
                  text = f"{text}"
          print(self.hps_ms)
          stn_tst, clean_text = self.get_text(text, self.hps_ms, is_symbol)
          with no_grad():
              x_tst = stn_tst.unsqueeze(0).to(self.device)
              x_tst_lengths = LongTensor([stn_tst.size(0)]).to(self.device)
              sid = LongTensor([speaker_id]).to(self.device)
              audio = net_g_ms.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=noise_scale, noise_scale_w=noise_scale_w,
                                    length_scale=length_scale)[0][0, 0].data.cpu().float().numpy()
          return audio
      return tts_fn
  @classmethod
  def create_to_symbol_fn(hps):
      def to_symbol_fn(is_symbol_input, input_text, temp_lang):
          if temp_lang == 0:
              clean_text = f'[ZH]{input_text}[ZH]'
          elif temp_lang == 1:
              clean_text = f'[JA]{input_text}[JA]'
          else:
              clean_text = input_text
          return _clean_text(clean_text, hps.data.text_cleaners) if is_symbol_input else ''
      return to_symbol_fn
  @classmethod
  def change_lang(language):
      if language == 0:
          return 0.6, 0.668, 1.2
      elif language == 1:
          return 0.6, 0.668, 1
      else:
          return 0.6, 0.668, 1
  @classmethod
  def generate(self,text,lang,speakerId,name,noise_scale,noise_scale_w,length_scale):
    tts_fn = self.create_tts_fn(self.models[name],speakerId)
    return tts_fn(text,lang,noise_scale, noise_scale_w, length_scale, True)
  @classmethod
  def loadModels(self,device,config_path,models_path,info_path):
    self.hps_ms = utils.get_hparams_from_file(config_path)
    with open(info_path, "r", encoding="utf-8") as f:
        models_info = json.load(f)
        with open(info_path, "r", encoding="utf-8") as f:
            models_info = json.load(f)
        for i, info in models_info.items():
            net_g_ms = SynthesizerTrn(
            len(self.hps_ms.symbols),
            self.hps_ms.data.filter_length // 2 + 1,
            self.hps_ms.train.segment_size // self.hps_ms.data.hop_length,
            n_speakers=self.hps_ms.data.n_speakers if info['type'] == "multi" else 0,
            **self.hps_ms.model)
            utils.load_checkpoint(f'{models_path}/{i}/{i}.pth', net_g_ms, None)
            print(f'check loaded {i}')
            _ = net_g_ms.eval().to(device)
  @classmethod
  def loadModel(self, device, config_path, model_path,model_name):
      self.device = torch.device(device)
      if model_name in self.models:
        return
      self.hps_ms = utils.get_hparams_from_file(config_path)
      net_g_ms = SynthesizerTrn(
          len(self.hps_ms.symbols),
          self.hps_ms.data.filter_length // 2 + 1,
          self.hps_ms.train.segment_size // self.hps_ms.data.hop_length,
          n_speakers=self.hps_ms.data.n_speakers,
          **self.hps_ms.model
      )
      utils.load_checkpoint(model_path, net_g_ms, None)
      net_g_ms = net_g_ms.eval().to(device)
      self.models[model_name] = net_g_ms
