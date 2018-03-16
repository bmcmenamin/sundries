import sys
sys.path.append('/home/mcmenamin/model_wrangler')

import os
from model_wrangler.model_wrangler import ModelWrangler

ROOT_DIR = '/home/mcmenamin/sundries/moby_sequel/'
MODEL_DIR  = os.path.join(ROOT_DIR, 'mw', 'moby_model_stride10')

param_file = os.path.join(MODEL_DIR, 'model_params.pickle')
restored_model = ModelWrangler.load(param_file)


pred_text = "You can still call me Ishmael. "
for _ in range(5000):
    pred_text += str(restored_model.predict([[pred_text]])[0], 'utf8')
print(pred_text)

