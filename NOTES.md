# Study notes - what the code does to the learning, and what the results mean

These are notes to help understand the code and results. **They are not the reflections.** The brief says the reflections must be your own writing ("For the reflections that must be provided it is not acceptable to use AI tools"), so the reflection cells in the notebook are left for you. Use these notes as a reference for what happened and why, then write the reflections in your own words from what you see in the videos, plots and tables.

The brief also says you may be asked to explain any line of the notebook, so section 1 walks through every moving part.

---

## 1. How the pieces fit together

```
BipedalWalkerUCD  ->  CustomRewardWrapper  ->  TimeLimit(5000)  ->  PPO (SB3)
  physics + base reward    adds weighted extra terms    cuts long episodes     learns from the TOTAL reward
                                                                               VideoEvalCallback every 5000 steps
```

**The environment (`bipedal_walker_ucd.py`).** Box2D physics, 50 steps per second. Each step the agent sends 4 numbers in [-1, 1]: hip 1, knee 1, hip 2, knee 2. The sign picks the motor direction and the size sets how much torque the motor may use (`maxMotorTorque = 80 * |a|`). The observation is 24 numbers: hull angle, angular velocity, x/y velocity, then per leg (hip angle, hip speed, knee angle + 1, knee speed, foot contact), then 10 lidar distances.

How the UCD version differs from the normal gym `BipedalWalker-v3` (compare `step()` with the gym source):
- The base reward is **only** forward progress, `130 * x / SCALE` as a difference each step (about 4.33 per unit of distance, about 364 for the whole course), plus **-100** if the hull touches the ground.
- The normal gym version also subtracts a torque cost and `5 * |hull angle|` inside the reward. Here both are removed and come back as optional terms in the wrapper (`action_penalty`, `stability`). So "all weights 0" is an easier, pure-distance reward than the standard one.

**`CustomRewardWrapper`.** `gym.Wrapper` sits between the agent and the env. Its `step()` calls the real env, then computes the extra terms from the robot's state and returns `base + sum(weight * term)`. **PPO only ever sees this total**; it never sees the terms separately. `info["reward_components"]` is just for logging (the callback sums it per episode).

**`TimeLimit(5000)`.** Ends an episode after 5000 steps (100 seconds of sim time) even if the robot hasn't fallen or finished. Without it, an agent that learns to stand still would run forever. It reports this as `truncated`, not `terminated`, so PPO still bootstraps the value there and doesn't treat it as a real ending.

**PPO.** Collects 2048 steps with the current policy, then works out for each step how much better than expected the outcome was (the *advantage*, using GAE). It then does 10 passes over those 2048 steps in minibatches of 64. Each pass nudges the actor to make good actions more likely, but the probability ratio is clipped to [0.8, 1.2] (`clip_range`) so the policy can't jump too far from the one that collected the data. The critic learns to predict the discounted return. "MlpPolicy" here means two separate networks (actor and critic), each 24 -> 64 -> 64 with tanh. The actor outputs the mean of a Gaussian per joint; during training actions are sampled (exploration), and in evaluation `deterministic=True` uses the mean.

| Setting | Value | What it does here |
|---|---|---|
| `n_steps` | 2048 | data per update; 500k steps = 245 updates |
| `batch_size` | 64 | minibatch size, 32 minibatches per epoch |
| `n_epochs` | 10 | passes over each batch of data, so each sample is used 10 times |
| `learning_rate` | 2.5e-4 | Adam step size |
| `gamma` | 0.99 | discount. Effective horizon about 1/(1-0.99) = 100 steps = 2 s, so the agent cares about the next couple of seconds, not the whole course |
| `gae_lambda` | 0.95 | how far the advantage looks ahead before trusting the critic (bias/variance tradeoff) |
| `clip_range` | 0.2 | the "proximal" bit, a trust region on each update |
| `ent_coef` | 0.001 | tiny bonus for keeping the action distribution wide (exploration) |

**`VideoEvalCallback`.** Every 5000 training steps (`n_calls % eval_freq`; with one env, 1 call = 1 step), it plays 3 episodes with the deterministic policy, logs the mean total reward and the mean of each component to TensorBoard, and saves the first episode as an mp4 (up to 1000 frames, 30 fps). The 3 eval episodes are on new random terrain each time, so eval numbers are noisy.

---

## 2. The reward terms, what each one pushes the agent towards

Per-step sizes, so you can compare them with the base reward. A decent walker moves about 2-4 units/s, which is about **0.17-0.35 base reward per step**.

| Term | Formula (per step) | Range per step | Over an episode | Pushes towards |
|---|---|---|---|---|
| base | 130·Δx/30, -100 on fall | ~0 to 0.4 | ~364 for the full course | moving right, not falling |
| action_penalty | -0.028·Σ\|a\| | -0.112 to 0 | ~-60 to -110 for 1000 steps | using less torque: lighter, less forceful motor use |
| stability | -\|hull angle\| | usually -0.02 to -0.3 | grows with episode length | keeping the hull level |
| smoothness | -Σ\|a_t - a_(t-1)\| | -8 to 0 | can be huge (PPO policies often flip ±1 a lot) | actions that don't flip around, so less shaking |
| speed | hull x-velocity | ~-1 to 5 | ≈ 50 × distance (Σv·dt = distance, dt = 1/50) | same as base, more distance reward, see below |
| custom (Task 2) | style × max(0, vx) | 0 to ~5 | ≈ 50 × (distance covered "in style") | the new gait feature |

Things worth understanding (useful for reflections and the viva):

1. **Per-step penalties make long episodes expensive.** The action, stability and smoothness terms are always ≤ 0 and paid every step. An agent that hasn't learned to walk yet can lose more by standing for 5000 steps (e.g. -0.2 × 5000 = -1000) than by falling straight away (-100). So big penalty weights can teach it to fall over on purpose. The normal gym version avoids this for the angle term by making it *potential-based* (it pays the *change* in -5·|angle| between steps, so it doesn't add up over time).
2. **Positive per-step bonuses invite loitering.** If you paid, say, +0.5 every step the robot was in the air, hopping on the spot for 5000 steps would earn 2500, far more than the ~364 for finishing the course. That's why the Task 2 term is multiplied by forward speed: it becomes "points per metre travelled in style", which you can't farm by standing still.
3. **The speed term is really a distance term.** Summed over an episode, Σ vx = 50 × distance, so `speed_weight = 0.1` adds about 5 per unit of distance on top of the base 4.33 (more than doubling it). What changes is the balance against the fixed -100 fall penalty. With more reward per metre, falling costs relatively less, so the agent takes more risks. The discount (γ = 0.99) also makes getting there sooner worth slightly more.
4. **Scale is everything.** A weight only means something relative to the per-step size of its term vs the base reward. `smoothness_weight = 0.05` sounds tiny, but the term can be -2 to -4 per step for a bang-bang policy, so it costs -0.1 to -0.2 per step, about the same size as the whole progress reward.
5. **Shaped total reward isn't comparable across agents.** Each agent was trained on a different total, so "agent B got 500 and agent A got 300" means nothing. Compare on the shared base reward and on behaviour (Task 3 section in the notebook).

---

## 3. Results

All numbers are from the executed notebook (Task 3 table: 10 fixed terrains, seeds 100-109, plain env reward, deterministic policy). Training took 23 min (baseline), 25 min (Task 1) and 45 min (Task 2) on the laptop CPU.

| agent | reached end | fell | base reward | speed | torque | jerkiness | hull tilt | bounce | airborne | hip corr | leg balance |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Baseline (all 0, 500k) | 50% | 50% | 214 | 2.57 | 2.91 | 0.76 | 0.46 | 0.21 | 0.44 | -0.59 | 0.46 |
| Task 1 (1.0 / 0.5 / 0.05 / 0, 500k) | 40% | 60% | 202 | 3.22 | 2.66 | 0.73 | **0.09** | **0.09** | 0.35 | -0.17 | **0.22** |
| Task 2 scissor @ 500k | 0% | 100% | -94 | - | 3.17 | 4.61 | 0.80 | - | - | - | - |
| Task 2 scissor @ 1M | **80%** | 20% | **283** | 2.99 | 3.39 | **4.26** | 0.42 | 0.13 | 0.26 | -0.56 | **0.78** |

### How the Task 2 term was chosen
Three candidate terms were piloted for 1M steps each, with the same seed and code as the notebook: "bounding" (forward distance while both feet are off the ground), "high knees" (forward distance with the swing thigh lifted past 0.5 rad), and "scissor" (forward distance while the hips swing in opposite directions).
- Bounding looked almost identical to the baseline. The baseline is already airborne about 44% of the time (it's a bounding run, not a walk), so there was little left to change.
- High knees did nothing visible to thigh lift. Every agent already pushes a hip to its joint limit (`hip_p95` ≈ 1.13 for all of them).
- Scissor changed the leg pattern the most, and it's the only one that doesn't rely on the unreliable contact flag. That's why it's in the notebook.

### What the outcomes mean

**Baseline (plain env reward).**
- The hull pitches a lot (mean tilt 0.46 rad, about 26°) and bounces (largest height std).
- One leg (hip 2) is held near its forward limit most of the time. The other swings between about -0.8 and 0, and there are long flight phases. Basically a lunging, bounding gait.
- Nothing in the reward cares about posture, only x-distance and not touching the hull, so leaning far forward is "free" until it falls. It reaches the end half the time.

**Task 1 (torque cost 1.0, stability 0.5, smoothness 0.05).**
- The stability term worked: hull tilt dropped 5× (0.46 → 0.09) and bounce more than halved. In the video the body stays almost flat.
- It also moves faster (3.2 vs 2.6 units/s) with a bit less torque (2.66 vs 2.91).
- The way it stays level is surprising, though. The hip-angle plot shows hip 1 **locked at the joint limit (1.1 rad) the whole episode**: one leg is held stiffly forward like a prop/ski, and the other leg does all the stepping with short quick hops. That's why leg balance is 0.22 and hip correlation is near 0.
- The optimiser found the cheapest way to satisfy "keep the hull level", and that was a strange, not-very-natural gait. It's a good example of a reward getting what you asked for rather than what you meant.
- The smoothness weight (0.05) barely changed jerkiness (0.73 vs 0.76). At that weight it's only about -0.04 per step, too small to matter next to the rest.
- Success was 40% vs the baseline's 50%. With 10 episodes and one seed, that difference is within noise. The posture change is the real effect.

**Task 2 (scissor term, weight 0.2, other weights 0).**
- The intended effect happened: both legs now share the work (leg balance 0.78, the most even of all agents), and the hips move in opposite directions (correlation -0.56).
- It is the most reliable agent: 80% reach the end, highest base reward (283).
- Reward hacking: jerkiness is 4.26, about 5.6× the baseline. The hip plot shows a sawtooth where the hip direction flips almost every step. The term pays for `-hip1_speed × hip2_speed`, i.e. for the hips *moving* in opposite directions, so the cheapest way to earn it is to vibrate them against each other very fast rather than take big strides. The reward measured hip *speed*, but "natural walking" is about hip *angle over a whole stride*. A version based on stride amplitude over a window (the template's `hip1_history`/`hip2_history` deques hint at this), or combined with the smoothness penalty, would probably fix it.
- Tilt went back up to 0.42 because the stability weight was 0 here.
- Airborne time is the lowest (0.26): the feet stay on the ground more and shuffle/vibrate.

**The 500k checkpoint of the Task 2 agent fell on all 10 terrains (tilt 0.8 rad, falls after ~70 steps).**
- That isn't "it hadn't learned yet". The learning curve shows it was already reaching the end before 200k. 500k just landed in one of the dips.
- All three learning curves swing between about 364 (reached the end) and -100 (fell) many times. PPO with these settings on one environment is unstable, and policies collapse and recover.
- The per-run panels (the thick line is a rolling mean over 10 evals) make the differences clearer:
  - **Baseline** collapsed completely and fell on every eval from about 150k to 250k steps before recovering.
  - **Task 1** learned fastest and most steadily, holding a rolling mean around 220-250 from 150k on. The stability/torque terms seem to have made learning more stable, even though the final 10-terrain test doesn't show it.
  - **Task 2** rises and falls in long waves.
- So the final agent you get depends a lot on *when* you stop. A fairer comparison would use the best checkpoint, or average over many checkpoints or seeds.

### For Task 3 ("which is best?")
- On the task itself (reaching the end, base reward), the Task 2 agent at 1M is best. But it trained twice as long, and at the same 500k budget it was the worst.
- On posture and efficiency (tilt, bounce, torque, speed), Task 1 is best, but its gait is odd (one leg held as a stiff prop).
- On "natural" two-legged leg use, Task 2 is best, but with vibrating hips that would be terrible for real motors.
- The plain shaped totals in the training printouts can't be compared across agents (different reward functions). Only the shared base reward and the behaviour metrics can.
- The evidence is one seed and 10 test terrains, and the learning curves show how noisy that is, so treat small differences (50% vs 40%) as noise.

Files: `videos/Baseline.mp4`, `videos/Reward_Exploration.mp4`, `videos/Custom_Reward.mp4`, `videos/Custom_Reward_500000_steps.mp4` (all on the same terrain, seed 100).

---

## 4. Things in the template that were broken or misleading (and what was changed)

| Where | Problem | Change |
|---|---|---|
| imports | `gc.collect()` used in the clean-up cells but `gc` never imported | `import gc` |
| Task 1 + 2 wrappers | leg-2 observation indices off by one: `hip2_angle = obs[10]` is hip-2 *speed*, `hip2_vel = obs[11]` is knee-2 *angle*, `knee2_vel = obs[13]` is the foot *contact* flag | leg 2 is `obs[9..13]`: angle 9, speed 10, knee 11, knee speed 12, contact 13 (doesn't change Task 1's reward, those variables aren't used there) |
| Task 2 wrapper cell | `modified_reward = (` never closed (SyntaxError). `__init__` reads `custom_component_weight` without taking it as an argument (it would silently pick up the global from the weights cell). The env cells pass `symmetry_weight` / `hop_height_weight` / `hop_velocity_weight`, which are defined nowhere (NameError) and which the class ignores anyway | rewritten so `__init__` takes `custom_component_weight`, matching the weights cell and env cells |
| callback cells | every run's eval writer logged to the same `./log_tb_bipedal/` folder, so eval curves from different runs land in one TensorBoard run | `tensorboard_log = log_tb_bipedal/<run_name>_eval`; the folder (and the run's eval videos) are wiped at the start of the cell so a re-run doesn't mix old and new curves |
| eval env | never seeded, so eval terrains (and the eval curves) change every run | `eval_env.reset(seed = seed + 1)` once after building it |
| callback (TensorBoard video) | `add_video` needs moviepy 1.x (`moviepy.editor`), which isn't installed; on Windows that route fails anyway (torch can't reopen its temp GIF) | `tb_video_every = 0` (the mp4s are still saved) |
| Task 2 callback comment | "evaluate every 20k steps" but `eval_freq = 5000` | comment corrected |

**Env quirk worth knowing (not changed, the env file is the course's):** the foot-contact flags in the observation (`obs[8]`, `obs[13]`) aren't fully reliable. `ContactDetector.EndContact` sets `ground_contact = False` whenever *any* contact of the lower leg ends. The ground is built from many separate edge pieces, so when a foot slides from one piece to the next, the flag drops to 0 while the foot is still down. It stays 0 until a new contact begins. On the walking pilot agents the flag was wrong on 2-4% of leg-steps (it says "in the air" when it isn't, never the other way around). For a robot standing still on a join it can be wrong almost all the time. The Task 3 analysis therefore asks Box2D directly (`edge.contact.touching` on the lower leg's contacts).

---

## 5. Judging which agent is "best" (for Task 3)

There's no single right answer, and that's the point of Task 3. Some angles:

- **Task success**: does it reach the end, how often (out of the 10 fixed terrains), and how often does it fall? This is the brief's own definition of "completing the task".
- **Shared reward**: the plain env (base) reward is the only number all agents were partly trained on. It's basically distance minus falls.
- **Efficiency**: torque used per step (energy), steps taken to finish.
- **Quality / style**: hull tilt, jerkiness, bounce, how natural the leg pattern looks (hip correlation, video). "Best" for a real robot (energy, wear on motors, stability) is not the same as "best" for the brief's "with style" theme.
- **Reliability**: one seed per agent and 3 eval episodes per checkpoint is very little data. RL results vary a lot between seeds, so a difference you see once might not hold. The learning curves also show that PPO can get worse after being good (policy collapse), so "best" depends on *when* you stop.
- **Fairness**: the Task 2 agent trained for 1M steps, the others for 500k. Compare the curves at equal steps too.

---

## 6. Limitations / honest caveats

- One seed (`seed = 0`) per configuration. The weights and the custom term were picked after short pilot runs of a few candidates (same seed, same code), so the configs that ended up in the notebook are ones that trained OK on this seed. A different seed could behave differently.
- Eval during training uses 3 episodes on random terrain, which is noisy.
- `deterministic=True` in eval uses the policy mean. Training used sampled actions, so eval behaviour can differ a bit from what the agent did during training.
