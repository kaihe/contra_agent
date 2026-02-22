---
description: How to add a new experiment write-up to the GitHub Pages site
---

# Add New Experiment to GitHub Pages

## Steps

1. **Run test.py** to generate the gameplay GIF:
   ```bash
   cd /home/kaihe/code/contra_agent/main
   python test.py --model trained_models/<model_name>.zip
   ```
   This saves a GIF to `main/recordings/<model_name>.gif`.

2. **Copy the GIF** to the docs assets folder:
   ```bash
   cp main/recordings/<model_name>.gif docs/assets/recordings/<model_name>.gif
   ```

3. **Create the experiment write-up** at `docs/experiments/<experiment_name>.md` using this template:
   ```markdown
   ---
   layout: default
   title: "<Experiment Title>"
   parent: Experiments
   nav_order: <next number>
   ---

   # Experiment: <Title>
   {: .no_toc }

   **Date:** YYYY-MM-DD · **Model:** `<model_file>.zip`
   {: .fs-5 .fw-300 }

   ---

   <details open markdown="block">
     <summary>Table of Contents</summary>
     {: .text-delta }
   1. TOC
   {:toc}
   </details>

   ---

   ## Training Configuration
   <!-- table of hyperparameters -->

   ## Reward Structure
   <!-- any changes from baseline -->

   ## Results
   ### Gameplay Recording
   ![Gameplay]({{ site.baseurl }}/assets/recordings/<model_name>.gif)

   ## Analysis
   <!-- what worked, what didn't, why -->
   ```

4. **Update the experiment index** in `docs/index.md` — add a row to the table:
   ```markdown
   | YYYY-MM-DD | [Experiment Name](experiments/<experiment_name>) | One-line summary |
   ```

5. **Commit and push**:
   ```bash
   git add docs/
   git commit -m "Add <experiment_name> experiment"
   git push
   ```
