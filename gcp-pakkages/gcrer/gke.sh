#!/bin/sh
gcloud container clusters get-credentials cluster-1 --zone us-central1-c --project mlops-308101
>>>>
Fetching cluster endpoint and auth data.
kubeconfig entry generated for cluster-1.


python3 -m venv venv && source venv/bin/activate
git clone https://github.com/madara-tribe/flask_on_kubernetes
cd flask*
pip3 install -r requirement.txt
pip3 install google-cloud-storage
pip3 install 'h5py==2.10.0' --force-reinstall
python3 -c 'import tensorflow as tf; print(tf.__version__)'  # for Python 3
>>> 1.14.1
