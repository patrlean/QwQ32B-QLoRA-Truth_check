git clone https://github.com/NVIDIA/apex.git
cd apex
pip install -v --disable-pip-version-check --global-option="--cpp_ext" --global-option="--cuda_ext" .
cd ..
pip install nltk
git clone https://github.com/NVIDIA/Megatron-LM
