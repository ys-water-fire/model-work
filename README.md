📌 **Dependency Note:**

If you want to run these scripts, please download the corresponding dataset (Flickr8kCN) first. 

⚠️ ** I used absolute paths in the code.** 
I sincerely apologize for the inconvenience. Please feel free to change them to relative paths according to your local folder structure.

⚠️ **Pre-trained Weights Notice**
I didn't upload the large pre-trained weight files to this repo. Please follow these steps to use the code:
1. First, run my `write_loss` function (this is the full training script) to train your own weights. After training, a `.pth` file will be saved to the `./opencv/` folder automatically.
2. Once training finishes, open the corresponding `.py` file (e.g. `blip.py`), find the `torch.load` line, and change the weight file name to match the one you just generated (for example, change `blip_1.checkpoint.pth` to `blip_10.checkpoint.pth` if you trained for 10 epochs).
3. If you're using absolute paths in the code, adjust the path in `torch.load()` to the correct location on your own computer.

4. checkpoint=torch.load(f'./(your own flie which save the weight)/blip_1.checkpoint.pth',device=device)
