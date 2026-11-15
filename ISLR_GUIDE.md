# Adding the Google ISLR (Kaggle) dataset to SignSpeak

What you get: many more signers for your signs (hello, goodbye, thank you, yes, no, please ...) and a big
pool of *other* signs for the "unknown" class. Your own recordings stay the test set, so the accuracy
you measure is still the accuracy on YOUR webcam.

Files in this update: `islr_extract.py` (step 1), `islr_convert.py` (step 2), `train.py` (new `--extra-data`
option and class weights), `requirements.txt`.

## Before you start
1. Copy the four files into your project folder (replace `train.py` and `requirements.txt`).
   If you changed `train.py` yourself, tell me and I will merge instead.
2. Back up `models/` (copy the folder), so you can compare old and new.
3. Open `config.py` and check `SIGNS` lists every sign you want. The importer only keeps signs that are in this list.
4. You need your own recordings for each sign (even about 15 each) from at least two signers in `data/`.
   They are used for calibration, validation and testing.

## Step 1: get the clips (use a free Kaggle Notebook, no 40 GB download)
1. Sign in at kaggle.com. Open the competition "Google - Isolated Sign Language Recognition" and accept its rules
   (button "Join" / "I understand and accept").
2. Click **Code -> New Notebook**. On the right choose **Add Input**, search for that competition, and add it.
   The files appear under `/kaggle/input/asl-signs`.
3. In the first cell, paste the whole of `islr_extract.py` below a first line `%%writefile islr_extract.py`, and run it.
4. In a second cell run:
       !python islr_extract.py --root /kaggle/input/asl-signs --list-signs
   Check which of your signs were found (the script prints matches such as `bye -> goodbye`).
5. Then run:
       !python islr_extract.py --root /kaggle/input/asl-signs --out /kaggle/working/islr_raw.npz --signs hello,goodbye,thank_you,yes,no,please
   (edit the list to your signs; keep `unknown` out of it. Signs the dataset does not have are reported, and you keep recording those yourself).
6. When it finishes, open the notebook's **Output** panel and download `islr_raw.npz` (about 10-20 MB).
   Put it in your project folder.

If you downloaded the whole dataset instead, run step 5 locally with `--root D:\path\to\asl-signs`
(needs `python -m pip install pandas pyarrow`).

## Step 2: convert to SignSpeak format (on your laptop)
    python islr_convert.py --raw islr_raw.npz

It compares the clips with your recordings and prints a table. The first line, the smallest distance, is used
(the mirror setting must be "yes" if the app mirrors your webcam, which is the default). It then writes
`data_islr/<sign>/...npy`.

## Step 3: train and compare
    python train.py --test-person YOURNAME --extra-data data_islr

`YOURNAME` is the signer name in your own files (the part of the file name before the underscore).
Look at the printed "Training sequences per class" and at the test accuracy and confusion list.
Compare with your old model, for example with the same `--test-person` and no `--extra-data`.

## Step 4: check it in the app
Run the app, use the signer page and try each sign several times. If a sign is recognised worse than before,
copy the backed-up `models/` folder back.

## Things to know
- Signs the dataset does not contain (probably "i love you", "sorry", "help") are not helped by this data.
  `train.py` gives them extra weight, but they still rely on your own recordings.
- The dataset was filmed on phones, so accuracy on your webcam can differ. Keep your own signers in the test set.
- If the calibration table shows nearly equal distances for several rows, tell me and send the table.
- Useful options: `--max-per-sign 200` and `--unknown 600` on step 1 (more clips or fewer),
  `--aspect 0.75 --flip yes` on step 2 to skip auto-calibration, `--no-padded` to drop the second copy of each clip.
- Check the Kaggle competition rules/licence before you publish anything that contains this data.
