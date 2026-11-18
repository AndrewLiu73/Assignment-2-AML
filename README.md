# COMP41810 Assignment 2 - Learning to Walk *With Style*

PPO (stable-baselines3) agents for the `BipedalWalkerUCD` environment, trained with different reward mixes to see how the reward changes the gait.

## What's in here

| File / folder | What it is |
|---|---|
| `COMP41810_Assignment_2_Going_The_Distance.ipynb` | The completed notebook (template headings kept in order) |
| `COMP41810_Assignment_2_Going_The_Distance.html` | HTML export of the executed notebook (all outputs) |
| `bipedal_walker_ucd.py` | The course's modified environment (unchanged) |
| `agents/` | Saved PPO agents: `Baseline`, `Reward_Exploration` (Task 1), `Custom_Reward` (Task 2, 1M steps) and `Custom_Reward_500000_steps` (Task 2 at 500k, for a like-for-like comparison). A re-run also writes `Custom_Reward_1000000_steps.zip` (checkpoint taken just before the last update), which isn't used |
| `videos/` | One video per agent, all on the same terrain (seed 100), plus frame-strip summaries |
| `NOTES.md` | Study notes: what each part of the code does to the learning, and what the results mean |
| `requirements.txt` | Pinned package versions used for the run |
| `COMP41810_Assignment_2_Going_The_Distance_Template.ipynb`, `Advanced Machine Learning Assignment 2.pdf`, `bipedal_walker.gif` | Provided files, untouched |

Not tracked by git (they get regenerated when the notebook runs, and are big): `eval_videos/` (one video per evaluation), `log_tb_bipedal/` (TensorBoard logs).

## Setup (Windows, Python 3.13)

```bash
python -m venv .venv
.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python -m pip install -r requirements.txt
```

Box2D installs from a prebuilt wheel on Python 3.13, so swig is not needed. `gymnasium[box2d]` pulls in `pygame-ce`; don't also install plain `pygame` (both use the same `pygame` folder and clash).

## Running

Open the notebook with the venv's Python so the kernel has the packages. Call the module directly (`-m notebook`, `-m nbconvert`): `python -m jupyter <cmd>` looks the subcommand up on PATH and can end up starting a different Python's Jupyter, whose kernel won't have gymnasium.

```bash
.venv\Scripts\python -m notebook COMP41810_Assignment_2_Going_The_Distance.ipynb
```

Run all cells top to bottom. The three training runs took about 95 minutes in total on a laptop CPU (8-core Ryzen 7). To watch training live:

```bash
.venv\Scripts\python -m tensorboard.main --logdir ./log_tb_bipedal/
```

then open http://localhost:6006/. Delete `log_tb_bipedal/` and `eval_videos/` before a full re-run, otherwise the old and new eval curves get mixed in the Task 3 plot.

To run headless and export the HTML in one go:

```bash
.venv\Scripts\python -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 COMP41810_Assignment_2_Going_The_Distance.ipynb
```

```bash
.venv\Scripts\python -m nbconvert --to html COMP41810_Assignment_2_Going_The_Distance.ipynb
```

## Before submitting

1. Fill in names and student numbers in the first cell.
2. Write the three reflections (the brief says these must be your own writing, not AI).
3. Re-export the HTML (second command above, or File > Save and Export Notebook As > HTML) so it includes the reflections.
4. Zip: notebook, HTML, `bipedal_walker_ucd.py`, `bipedal_walker.gif`, `videos/`, `agents/`.
