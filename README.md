
### **Krok 1: Logowanie na PLGrid**

✅ Instrukcja zgodna.

```bash
ssh login@athena.plgrid.pl
cd $HOME
```

---

### **Krok 2: Skopiowanie repozytorium z GitHub**

✅ Instrukcja zgodna.

```bash
git clone https://github.com/uzytkownik/moj_projekt.git
cd moj_projekt
```

---

### **Krok 3: Utworzenie środowiska wirtualnego w $SCRATCH**


```bash
cd $SCRATCH
mkdir -p projekty/moj_projekt
cd projekty/moj_projekt

module load Python/3.10.4 CUDA/12.3.0 cuDNN/8.9.7.29-CUDA-12.3.0
python -m venv my_env
source my_env/bin/activate

pip install --upgrade pip
pip install --no-cache-dir -r $HOME/moj_projekt/requirements.txt
```

---

### **Krok 4: Pobranie datasetu narad/ravdess do $SCRATCH**



```python
import os
from datasets import load_dataset

dataset_root = os.path.join(os.getenv('SCRATCH'), 'ravdess_dataset')
dataset = load_dataset("narad/ravdess", cache_dir=dataset_root)
```

* 🔹 Jeśli dataset wymaga tokena Hugging Face, ustaw zmienną środowiskową:

```bash
export HUGGINGFACE_HUB_TOKEN='twój_token'
```

---


### **Krok 6: Utworzenie pliku batch dla SLURM**


```bash
#!/bin/bash -l

#SBATCH --job-name=ravdess-exp
#SBATCH --account=plgkdistgnn-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output="output-%A_%a.out"
#SBATCH --error="error-%A_%a.err"

module load Python/3.10.4 CUDA/12.3.0 cuDNN/8.9.7.29-CUDA-12.3.0
# 1. Wejście do katalogu scratch projektu
cd $SCRATCH/projekty/moj_projekt

# 2. Aktywacja środowiska wirtualnego
source my_env/bin/activate

# 3. Uruchomienie skryptu
python $HOME/moj_projekt/fineTuningRavdess.py
```



---

### **Krok 7: Wysłanie zadania do kolejki**



```bash
sbatch run_job.sh
squeue -u $USER
```

* Logi: `output-<jobID>_*.out` i `error-<jobID>_*.err`.

---

### **Krok 8: Przechowywanie wyników**


```bash
cp -r $SCRATCH/wav2vec2_checkpoints $HOME/moj_projekt_results_backup
```


### Opcjonalnie: Uruchamianie interaktywnie

Do szybkich testów CPU:
```bash
srun --time=1:00:00 --mem=8G -N 1 -n 4 --partition=plgrid-now --pty /bin/bash
source $SCRATCH/my_env/bin/activate
python $HOME/moj_projekt/fineTuningRavdess.py
```

Do testów GPU:
```bash
srun --time=2:00:00 --mem=32G --ntasks=1 --gres=gpu:1 --partition plgrid-gpu-a100 --pty /bin/bash
source $SCRATCH/my_env/bin/activate
python $HOME/moj_projekt/fineTuningRavdess.py
```
