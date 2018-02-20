import sys; sys.path.append('/home/mcmenamin/model_wrangler')

import os
from model_wrangler.model_wrangler import ModelWrangler

MODEL_DIR  = '/home/mcmenamin/sundries/moby_sequel/moby_model'

param_file = os.path.join(MODEL_DIR, 'model_params.pickle')
restored_model = ModelWrangler.load(param_file)


pred_text = "You can still call me Ishmael. "
for _ in range(500):
    pred_text += str(restored_model.predict([[pred_text]])[0], 'utf8')
print(pred_text)

