# Dawify
Turn any mp4 to DAW file (We'll focus on midi for now)

# Install
<span style="color:red">1. **make a conda environment**</span>  
2. run the below
```bash
# at the root project directory, run:
python -m pip install .

# python removed setup.py; need this for fancy dependencies
bash setup.sh
```

# Usage
```bash
# run under the project root directory:
# NOTE: modify the parameters in run.py before hand if using it as a script
python dawify/run.py

# results are saved under the directory, outputs.
```

### To see all options when using client:
```bash
python dawify/run.py -h
```

# references
[YourMT3](https://github.com/mimbres/YourMT3) GPL2, need to replace in future  
[demucs](https://github.com/facebookresearch/demucs)  
[Apollo](https://github.com/JusperLee/Apollo) This is not commercial liscenece, replace in future  