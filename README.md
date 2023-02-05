# Using the cross-entropy method to solve Frozen Lake

This project solves the famous [Frozen Lake](https://www.gymlibrary.dev/environments/toy_text/frozen_lake/) environment using a reinforcement learning (RL) method known as **cross-entropy**. Frozen Lake is an [OpenAI Gym](https://www.gymlibrary.dev) environment in which an agent is rewarded for traversing a frozen surface from a start position to a goal position without falling through any perilous holes in the ice. See my [blog post](https://dev.to/exactful/using-the-cross-entropy-method-to-solve-frozen-lake-3cea) for more details.

![Trained actions](https://raw.githubusercontent.com/exactful/cross-entropy-rl/master/images/trained-actions.gif)

## Read more

Blog post: [Using the cross-entropy method to solve Frozen Lake](https://dev.to/exactful/using-the-cross-entropy-method-to-solve-frozen-lake-3cea) on [DEV](https://dev.to)

## Installation and usage

```
git clone https://github.com/exactful/cross-entropy-rl.git
cd cross-entropy-rl
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
python3 cross-entropy-rl-train.py
```

## Strategy

1. Create a neural network
2. Use the neural network to play a batch of episodes using the states as inputs and the outputs as actions
3. Set a reward threshold and filter out "bad" episodes that have episode rewards below the threshold
4. Train the neural network on the remaining "good" episodes using the states as inputs and the actions taken as desired outputs; our objective will be to minimise the loss between the neural network output and the desired output
5. Repeat from step 2 until the mean episode reward of a batch is good enough

## Results

Running the code leads to a solution being found after around 120 episodes:

```
0:      Loss: 1.3657    Mean ep reward: 0.01    Mean disc ep reward: 0.0048
1:      Loss: 1.3814    Mean ep reward: 0.03    Mean disc ep reward: 0.0088
2:      Loss: 1.3742    Mean ep reward: 0.0     Mean disc ep reward: 0.0
3:      Loss: 1.3726    Mean ep reward: 0.02    Mean disc ep reward: 0.0055
4:      Loss: 1.3654    Mean ep reward: 0.01    Mean disc ep reward: 0.0035
5:      Loss: 1.3597    Mean ep reward: 0.03    Mean disc ep reward: 0.0078
6:      Loss: 1.3542    Mean ep reward: 0.02    Mean disc ep reward: 0.0042
7:      Loss: 1.3504    Mean ep reward: 0.0     Mean disc ep reward: 0.0
8:      Loss: 1.3466    Mean ep reward: 0.0     Mean disc ep reward: 0.0
9:      Loss: 1.3447    Mean ep reward: 0.01    Mean disc ep reward: 0.0021
10:     Loss: 1.3438    Mean ep reward: 0.04    Mean disc ep reward: 0.0116
11:     Loss: 1.3393    Mean ep reward: 0.04    Mean disc ep reward: 0.0122
12:     Loss: 1.3352    Mean ep reward: 0.01    Mean disc ep reward: 0.0028
13:     Loss: 1.3291    Mean ep reward: 0.01    Mean disc ep reward: 0.0023
14:     Loss: 1.323     Mean ep reward: 0.04    Mean disc ep reward: 0.011
...
...
...
111:    Loss: 0.5678    Mean ep reward: 0.62    Mean disc ep reward: 0.3
112:    Loss: 0.4479    Mean ep reward: 0.67    Mean disc ep reward: 0.3268
113:    Loss: 0.3356    Mean ep reward: 0.6     Mean disc ep reward: 0.2909
114:    Loss: 0.3315    Mean ep reward: 0.7     Mean disc ep reward: 0.3418
116:    Loss: 0.4725    Mean ep reward: 0.64    Mean disc ep reward: 0.3193
117:    Loss: 0.3119    Mean ep reward: 0.77    Mean disc ep reward: 0.3836
118:    Loss: 0.3133    Mean ep reward: 0.67    Mean disc ep reward: 0.3338
120:    Loss: 0.4974    Mean ep reward: 0.71    Mean disc ep reward: 0.3516
121:    Loss: 0.3878    Mean ep reward: 0.68    Mean disc ep reward: 0.3379
122:    Loss: 0.3064    Mean ep reward: 0.72    Mean disc ep reward: 0.3603
124:    Loss: 0.4871    Mean ep reward: 0.7     Mean disc ep reward: 0.3469
125:    Loss: 0.3627    Mean ep reward: 0.7     Mean disc ep reward: 0.3479
126:    Loss: 0.2901    Mean ep reward: 0.76    Mean disc ep reward: 0.3791
128:    Loss: 0.4612    Mean ep reward: 0.71    Mean disc ep reward: 0.3554
129:    Loss: 0.2774    Mean ep reward: 0.79    Mean disc ep reward: 0.3982
Environment solved!
```