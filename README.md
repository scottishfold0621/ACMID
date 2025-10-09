# ACMID: Automatic Curation of Musical Instrument Dataset for 7-Stem Music Source Separation
This repository hosts the official code and pre-trained models for the paper "ACMID: Automatic Curation of Musical Instrument Dataset for 7-Stem Music Source Separation" (submitted to ICASSP 2026). The project addresses key limitations in music source separation (MSS) by constructing a high-quality, high-granularity 7-stem dataset through automated web crawling and data cleaning.

## 1. Project Overview

we propose ACMID (Automatic Curation of Musical Instrument Dataset), a pipeline that:

<div align="center">
  <img src="https://github.com/scottishfold0621/ACMID/blob/main/image/main_procedure1.png">
</div>

- Crawls target instrument raw video from YouTube using multilingual queries.
- Cleans the raw data with a pre-trained audio encoder-based binary classifier, resulting in a high-precision 7-stem data (ACMID-Cleaned).
- Expands MSS from 4-stem to 7-stem, enabling fine-grained separation of individual instruments.

**Key Contributions**
- **Automated Data Processing Pipeline**: Open-source web crawling and cleaning code to replicate the dataset.
- **High-Granularity**: 7-stem taxonomy (Piano/Drums/Bass/Acoustic Guitar/Electric Guitar/Strings/Wind-Brass) fills the gap in fine-grained MSS research.
- **Proven Effectiveness**: ACMID-Cleaned improves MSS performance by 2.39dB (vs. ACMID-Uncleaned) and boosts SOTA model (SCNet) performance by 1.16dB when combined with existing datasets.

## 2. Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/acmid.git
    cd acmid
    ```

2.  **Install dependencies:**
    It is recommended to use a virtual environment (e.g., venv or conda).
    ```bash
    pip install -r requirements.txt
    ```

## 3. Usages

### 3.1 Web-crawling with Multi-lingual Queries

Use `search_youtube.py` to do web-crawling for different musical instruments. You can edit the script to change the search queries and target instrument. The results of the web-crawling will be saved as a `.csv` file. For example, `piano_solo.csv` is provided as an example.

```bash
python search_youtube.py
```

### 3.2 Data Cleaning Model with pretrained checkpoints

(1) The pretrained Dasheng model will be downloaded automatically. If the automatic download fails, you can manually download it from here: https://zenodo.org/records/13315686/files/dasheng_audioset_mAP497.pt?download=1 and place it in the root directory of this project.
[https://github.com/XiaoMi/dasheng]

(2) We open-source 7-stem of our instruments cleaning models: Acoustic Guitar, Bass, Drums, Electric Guitar, Piano, Strings, Wind
`Instruments_cleaner_dasheng_{stem_name}.pth`
Here we give the last MLP weights of our checkpoints, and the Dasheng part should be downloaded with 2.2(1).

(3) Use `inference.py` to perform inference on the data cleaning model. Select instrument type with the `args.instr`.

**Example Usage:**

To clean piano audio files located in `/path/to/your/piano_audio_files` and save the cleaned files to `./cleaned_piano`, run the following command:

```bash
python inference.py --instr piano --input_path /path/to/your/piano_audio_files --output_path ./cleaned_piano --threshold 0.995
```

- `--instr`: The instrument type to process.
- `--input_path`: Path to the directory containing the audio files to be cleaned.
- `--output_path`: Path to the directory where the cleaned audio files will be saved.
- `--threshold`: The confidence threshold for the cleaning model.

## 4. Citation
If you find this work useful, please cite our paper:
```bibtex
@inproceedings{ji2026acmid,
  title={ACMID: Automatic Curation of Musical Instrument Dataset for 7-Stem Music Source Separation},
  author={Ji, Yu and Yang, shuo and Xu, Yuetonghui and Liu, Mengmei and Ji, Qiang and Han, Zerui},
  booktitle={IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year={2026},
  organization={IEEE}
}
```

