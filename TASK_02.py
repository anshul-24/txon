{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "00b6d3ce-7b12-4e2f-9fb7-59a1525932c0",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "WARNING:tensorflow:From C:\\Users\\Desktop\\anaconda3\\lib\\site-packages\\tensorflow\\python\\compat\\v2_compat.py:107: disable_resource_variables (from tensorflow.python.ops.variable_scope) is deprecated and will be removed in a future version.\n",
      "Instructions for updating:\n",
      "non-resource variables are not supported in the long term\n"
     ]
    }
   ],
   "source": [
    "#Imports\n",
    "import nltk\n",
    "import os\n",
    "from nltk.stem.lancaster import LancasterStemmer\n",
    "import numpy as np\n",
    "import tflearn\n",
    "import tensorflow as tf\n",
    "import random\n",
    "import json\n",
    "import pickle"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "f31fa13a-d24e-49c9-b7c0-6680a9779f0a",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Loading Data\n",
    "with open(\"C:/Users/Desktop/task2chatbot/Lintents.json\") as file:\n",
    "\tdata = json.load(file)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "2bdc7d87-a786-4419-9269-056fe2ca42a1",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Initializing empty lists\n",
    "words = []\n",
    "labels = []\n",
    "docs_x = []\n",
    "docs_y = []\n",
    "\n",
    "#Looping through our data\n",
    "for intent in data['intents']:\n",
    "\tfor pattern in intent['patterns']:\n",
    "\t\tpattern = pattern.lower()\n",
    "    \t\t#Creating a list of words\n",
    "\t\twrds = nltk.word_tokenize(pattern)\n",
    "\t\twords.extend(wrds)\n",
    "\t\tdocs_x.append(wrds)\n",
    "\t\tdocs_y.append(intent['tag'])\n",
    "\n",
    "\tif intent['tag'] not in labels:\n",
    "\t    labels.append(intent['tag'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "90b65cd3-ee1d-404f-a3b5-3318c1450c55",
   "metadata": {},
   "outputs": [],
   "source": [
    "stemmer = LancasterStemmer()\n",
    "words = [stemmer.stem(w.lower()) for w in words if w not in \"?\"]\n",
    "words = sorted(list(set(words)))\n",
    "labels = sorted(labels)\n",
    "\n",
    "training = []\n",
    "output = []\n",
    "\n",
    "out_empty = [0 for _ in range(len(labels))]\n",
    "for x,doc in enumerate(docs_x):\n",
    "\tbag = []\n",
    "\twrds = [stemmer.stem(w) for w in doc]\n",
    "\tfor w in words:\n",
    "\t\tif w in wrds:\n",
    "\t\t\tbag.append(1)\n",
    "\t\telse:\n",
    "\t\t\tbag.append(0)\n",
    "\toutput_row = out_empty[:]\n",
    "\toutput_row[labels.index(docs_y[x])] = 1\n",
    "\ttraining.append(bag)\n",
    "\toutput.append(output_row)\n",
    "#Converting training data into NumPy arrays\n",
    "training = np.array(training)\n",
    "output = np.array(output)\n",
    "\n",
    "#Saving data to disk\n",
    "with open(\"data.pickle\",\"wb\") as f:\n",
    "\tpickle.dump((words, labels, training, output),f)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "2317286f-a273-4587-a8dd-933d688da971",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Training Step: 3011  | total loss: \u001b[1m\u001b[32m1.27302\u001b[0m\u001b[0m | time: 0.054s\n",
      "| Adam | epoch: 251 | loss: 1.27302 - acc: 0.8765 -- iter: 88/91\n"
     ]
    }
   ],
   "source": [
    "#tf.reset_default_graph()\n",
    "\n",
    "net = tflearn.input_data(shape = [None, len(training[0])])\n",
    "net = tflearn.fully_connected(net,8)\n",
    "net = tflearn.fully_connected(net,8)\n",
    "net = tflearn.fully_connected(net,len(output[0]), activation = \"softmax\")\n",
    "net = tflearn.regression(net)\n",
    "\n",
    "model = tflearn.DNN(net)\n",
    "\n",
    "model.fit(training, output, n_epoch = 1000, batch_size = 8, show_metric = True)\n",
    "model.save(\"model.tflearn\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "1b11b8ae-1406-472f-b949-edc2068eec11",
   "metadata": {},
   "outputs": [],
   "source": [
    "def bag_of_words(s, words):\n",
    "    bag = [0 for _ in range(len(words))]\n",
    "\n",
    "    s_words = nltk.word_tokenize(s)\n",
    "    s_words = [stemmer.stem(word.lower()) for word in s_words]\n",
    "\n",
    "    for se in s_words:\n",
    "        for i, w in enumerate(words):\n",
    "            if w == se:\n",
    "                bag[i] = 1\n",
    "            \n",
    "    return np.array(bag)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "cea008eb-e382-4be6-886f-eb111426c015",
   "metadata": {},
   "outputs": [],
   "source": [
    "def chat():\n",
    "    print(\"Start conversation with the bot (type quit to stop)!\")\n",
    "    while True:\n",
    "        inp = input(\"You: \")\n",
    "        if inp.lower() == \"quit\":\n",
    "            break\n",
    "            \n",
    "        results = model.predict([bag_of_words(inp, words)])\n",
    "        results_index = np.argmax(results)\n",
    "        tag = labels[results_index]\n",
    "\n",
    "        for tg in data[\"intents\"]:\n",
    "            if tg['tag'] == tag:\n",
    "                responses = tg['responses']\n",
    "\n",
    "        print(random.choice(responses))\n",
    "\n",
    "chat()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.9"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
