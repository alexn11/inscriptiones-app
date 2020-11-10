import flask
from model.generate import generate
from model.model_list import model_list, model_uses_categories
from model.category_list import admissible_categories as category_list

app = flask.Flask(__name__, template_folder='templates')

def check_and_build_category_list(form_data):
  categories = []
  for category in category_list:
    if(category in form_data):
      categories.append(category)
  return categories


initial_form_state = {
 'model_name' : 'char-lstm',
 'prompt' : 'Classis',
 'temperature' : 0.5,
 'selected_categories' : [],
}

model_cartoons = {
 'lstm-char' : 'lstm-char.svg',
 'lstm-char-cond' : 'lstm-char-cond.svg',
 'lstm-word' : 'lstm-word.svg',
 'lstm-word-cond' : 'lstm-word-cond.svg',
 'transformer-v1' : 'transformer-model.svg',
 'transformer-v11' : 'transformer-model.svg',
 'transformer-v20' : 'transformer-model.svg',
 'bunny' : 'bunny.svg',
}

learning_curves = {
  model_name : model_name + '-lc.png' if(model_name not in [ 'transformer-v1', 'transformer-v20', 'bunny' ]) else 'dot.png' for model_name in model_cartoons
}



@app.route('/', methods=['GET', 'POST'])
def main():

  if(flask.request.method == 'GET'):
    return flask.render_template('main.html',
                                  model_list = model_list,
                                  model_cartoons = model_cartoons,
                                  learning_curves = learning_curves,
                                  category_list = category_list,
                                  form_state = initial_form_state, 
                                  model_uses_categories = model_uses_categories)
    
  if(flask.request.method == 'POST'):
  
    form_values = flask.request.form
    
    categories = check_and_build_category_list(form_values)
    form_state = {
      'model_name' : form_values['models'],
      'prompt' : form_values['prompt'],
      'temperature' : form_values['temperature'],
      'selected_categories' : categories,
    }
    print(f'app main: POST: New form state: {form_state}')
    
    model_name = form_state['model_name']
    options = {}
    options['categories'] = categories
    try:
      options['temperature'] = float(form_state['temperature'])
    except(ValueError):
      options['temperature'] = 0.5
    
    output = generate(model_name, form_state['prompt'], options)
    
    return flask.render_template('main.html',
                                 form_state = form_state,
                                 model_list = model_list,
                                 model_cartoons = model_cartoons,
                                 learning_curves = learning_curves,
                                 model_uses_categories = model_uses_categories,
                                 category_list = category_list,
                                 output = output)



if __name__ == '__main__':
    app.run()
    



















