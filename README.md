# foreign-accent-mos
Official repository for our paper:
> **How Should We Evaluate Naturalness for Synthetic Speech with a Foreign Accent?**  
> Kentaro Onda (The University of Tokyo/AIST), Satoru Fukayama (AIST), Daisuke Saito, Nobuaki Minematsu (The University of Tokyo)
> SLT2026

This repository contains resources for our listening study on foreign-accented speech.

In this study, speech from the ERJ corpus was analyzed and resynthesized using multiple speech processing systems. Listening tests were then conducted under multiple evaluation instructions and with different rater groups.

## Stimuli

To reproduce the listening-test stimuli:

1. Obtain the ERJ corpus from the official distribution site:
[ERJ corpus](https://www.nii.ac.jp/dsc/idr/speech/submit/UME-ERJ.html)
2. Place the corpus at `ERJ/` in this repository, so the paths in
   `original.scp` (for example `ERJ/wav/AE/F02/S_PH_B_1_182.wav`) exist.
3. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Clone the source repositories:

   ```bash
   git clone https://github.com/sarulab-speech/Sidon external/Sidon
   git clone https://github.com/NVIDIA/BigVGAN external/BigVGAN
   ```
5. Generate the stimuli:
   ```bash
   bash prep_stimuli.sh
   ```


## Evaluation Results
Listening-test responses are provided in `results/I{1,2,3,4}_{EN,JP}.json`.

