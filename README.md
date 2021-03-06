**Summary**

I was tasked to do a chatbot that could read the user input and produce a response.

The downside, I had zero knowledge to on neural networks, data science and machine learning.

I am fortunate enough to be given time to learn and navigate my way around this space, while trying to produce some form of sample along the way... So below are the links, steps that helped me to get to where I am.


**References**
1) What the original plan was : https://github.com/lukalabs/cakechat

2) Like an idiot, I dove right into that and drowned.

3) After realising it went nowhere, I started right at the beginning, with tensorflow's own documentation : https://www.tensorflow.org/tutorials

4) Additional.. 'fun' bits to read and experiment within an hour: https://www.gradio.app

5) Finally came to chatbot + tensorflow : https://towardsdatascience.com/build-it-yourself-chatbot-api-with-keras-tensorflow-model-f6d75ce957a5. This comes with the github repo, so I mainly referred to the jupyter notebook files.

6) An interesting common bit between the projects seen in 1) and 5), they both use Flask to set up a simple localhost server, when you run the python script. 

**Troubles and Issues faced along the way**
1) Keras + Flask API : Not alot of references.

2) Keras + Tensorflow : Versions not compatible with one another. I've addressed this in the 'tensorflow_assistant_py3.ipynb' and 'flask_api_tensorflow.py' script :
- import tensorflow.compat.v1 as tf
- tf.disable_v2_behavior()

3) If you use 'pickle.dump', be warned that you could face a thread lock issue and you would need to work around it. Some claimed it was a tensorflow issue, some claimed it was a Python one.
For the tutorial shown in 5): 
- pickle.dump(model, open("katana-assistant-model.pkl", "wb")) // this was shown but produced a thread lock issue.
- I just saved it using model.save() and avoided pickle.dump to save a model.
- Funnily enough : pickle.dump(model.history, open("katana-assistant-model.pkl", "wb")) would work. ( https://stackoverflow.com/questions/59326551/cannot-pickle-tensorflow-object-in-python-typeerror-cant-pickle-thread-loc )
- Related to this : https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/History ( model.fit() RETURNS a history object, so take note of that )

**Running**
- Run it via : python3 flask_api_tensorflow.py
- While the server is up :
    - Access the url : http://0.0.0.0:5000/predict?a=i_need_help&b=none
    - This would produce the probability and determine the intent, for the command ( a = 'I need help' and b = 'None' )
    - The python script 'GETs' the a and b values in the URL ( which is 'I need help' ) and checks it with the model and produces the probability alongside it.
    - The intents.json shows a couple of commands and data I fed into it, which is then used in the "tensorflow_assistant_py3.ipynb" to build the model.
- Try another URL : http://0.0.0.0:5000/predict?a=hello&b=none
    - This is under the intent : greeting, which can be referred to in the Intents.json.

**Building and defining your own model** 
- Change the intents in intents.json however you will. ( Remove, Upgrade etc ).
- Happy? Run the "tensorflow_assistant_py3.ipynb", this will build the model. ( both .pkl and .h5 )
- Then, run "python3 flask_api_tensorflow.py" in your terminal, to test your change. ( Provided youve installed the necessary files )

**Whats next?**
- Look into Flask examples and build a simple form or HTML to display your change.
- Maybe use gradio as well.
- I'll include it into this project when I clean up the build on my end... It's a mess.

