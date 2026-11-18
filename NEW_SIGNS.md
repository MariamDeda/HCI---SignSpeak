# Adding 10 everyday signs that are in the Google ISLR (PopSign) dataset

New signs: **mom, dad, water, food, drink, hungry, happy, sad, bad, where**.
All ten are in the dataset's 250-sign list (checked against the PopSign vocabulary page), so you can train
them from the Kaggle clips without recording yourself. (Not in the dataset: sorry, help, I love you.)

## What is already done (files in this update)
- `signs_ref.py`: animation + written steps + spoken/typed words for all 10 (the Speaker side works straight
  away, even before you retrain).
- `view_hearing.py`: the Speaker's word buttons and word list show every sign it can animate.
- `train.py`: new `--islr-holdout` option (see step 4) so the new signs are tested on signers the model never saw.
- `islr_extract.py`: default sign list now includes the 10.

## Step 1: add the names to config.py (do this yourself, your file has your own settings)
Open `config.py`, find the line that starts with `SIGNS = [`, and add the new names **before "unknown"**
(keep every name you already have). Example:

    SIGNS = ["hello", "goodbye", "thank_you", "i_love_you", "yes", "no",
             "mom", "dad", "water", "food", "drink", "hungry", "happy", "sad", "bad", "where",
             "unknown"]

`DISPLAY` needs no change (the names are plain words).

## Step 2: get the clips (Kaggle notebook, as before)
In the notebook run (all signs you want from the dataset, existing ones too):

    !python islr_extract.py --root /kaggle/input/asl-signs --out /kaggle/working/islr_raw.npz --signs hello,goodbye,thank_you,yes,no,please,mom,dad,water,food,drink,hungry,happy,sad,bad,where

It should print `match: kaggle 'mom' -> ours 'mom'` for each new sign. Download `islr_raw.npz` into your project.

## Step 3: convert
    python islr_convert.py --raw islr_raw.npz

## Step 4: train
    python train.py --test-person YOURNAME --extra-data data_islr

For the 10 new signs you have no recordings, so the script keeps about 20% of the dataset's signers out of
training (see the line "held out ... clips from unseen signers"). The accuracy report then includes those
unseen signers, so you get an honest number for the new signs too. Change the share with `--islr-holdout 0.3`.

## Step 5: try it
Run the app. The Signer page now recognises the new signs; the Speaker page already animates them.

## Honest limits (worth a sentence in your report)
- The model only sees ONE hand (65 numbers per frame), not the face. Signs that differ mainly by where the hand
  touches the face (mom vs dad: same hand, chin vs forehead) rely on the hand's height in the picture, so
  expect confusion between them. The confusion list printed after training will show it.
- happy and sad are two-handed; only one hand is used for recognition.
- The clips were filmed on phones by many people. Accuracy on your webcam can be lower than the test number.
  If you can, record 5-10 clips of any sign that performs badly (`collect_data.py`) and retrain.
- The animations were written from written descriptions. Check each one against an ASL reference
  (for example Handspeak or ASL University) before showing the app to participants.
