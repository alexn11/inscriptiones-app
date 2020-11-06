
import re


from model.model_list import lstm_models, transformer_models

from model.lstm_generator import lstm_generate
from model.transformer_generator import generate_text as transformer_generate


def romanize(text):
  return text.upper().replace('U','V').replace('J', 'I')


def get_transformer_model_number(model_name):
  number_match = re.search(r'transformer\-v([0-9]+)', model_name)
  return int(number_match.group(1))

def generate(model_name, prompt, options = None):

  if(prompt == ''):
    prompt = 'Classis'

  temperature = 1.
  if('temperature' in options):
    try:
      temperature = float(options['temperature'])
    except(ValueError):
      temperature = 1.
      
  if (model_name in transformer_models):
    model_number = get_transformer_model_number(model_name)
    output = transformer_generate(model_number, prompt, temperature = temperature)
  elif (model_name in lstm_models):
    output = lstm_generate(model_name, prompt, temperature = temperature, categories = options['categories'])
  elif (model_name == 'bunny'):
    output = 'bunny'
  else:
    output = '(Error: unknown model)'
    
  return romanize(output)
