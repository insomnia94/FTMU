
import gym, os
from itertools import count
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy
from PIL import Image
from torchvision import transforms
import torchvision.models as models
import copy


class Actor(nn.Module):
    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.linear1 = nn.Linear(self.state_size, 2048)
        self.linear2 = nn.Linear(2048, 512)
        self.linear3 = nn.Linear(512, self.action_size)

    def forward(self, state):
        output = F.relu(self.linear1(state))
        output = F.relu(self.linear2(output))
        output = self.linear3(output)
        distribution = Categorical(F.softmax(output, dim=-1))
        return distribution


class Critic(nn.Module):
    def __init__(self, state_size, action_size):
        super(Critic, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.linear1 = nn.Linear(self.state_size, 2048)
        self.linear2 = nn.Linear(2048, 512)
        self.linear3 = nn.Linear(512, 1)

    def forward(self, state):
        output = F.relu(self.linear1(state))
        output = F.relu(self.linear2(output))
        value = self.linear3(output)
        return value


def compute_returns(next_value, rewards, masks, gamma=0.99):
    R = next_value
    # next_value is the estimated value(return) of the next state of step(-1)
    returns = []
    # step(0) -> step(1) -> step(2) -> ... -> step(-2) -> step(-1)
    # step(1) is the next step of step(0), step(-1) is the next step of step(-2)
    # return(-1) = reward(-1) + gamma * next_value   (if step(-1) is the last step, the next step of step(-1) is the beginning step, the return of the beginning step is 0)
    # return(-2) = reward(-2) + gamma * return(-1)
    # return(-3) = reward(-3) + gamma * return(-2)
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * masks[step]
        # add the return on the first element of the list, finally it will be the last one
        returns.insert(0, R)
    return returns

def state_generate(PIL_img, extract_model):

    #feature_sequential = extract_model.features

    img = normalization(PIL_img)
    img.unsqueeze_(dim=0)
    input_feature = copy.deepcopy(img)
    input_feature = torch.Tensor(input_feature).cuda().float()

    output = extract_model.conv1(input_feature)
    output = extract_model.bn1(output)
    output = extract_model.relu(output)
    output = extract_model.maxpool(output)
    output = extract_model.layer1(output)
    output = extract_model.layer2(output)
    output = extract_model.layer3(output)
    output = extract_model.layer4(output)
    state = extract_model.avgpool(output)

    state = state.reshape(-1).detach()

    return state

# program starts here

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = gym.make("CartPole-v0").unwrapped

#state_size = 25088  # the size of the state after flatten
state_size = 2048  # the size of the state after flatten
action_size = 2  # the number of actions of actor
e_iters = 20000  # the number of training iteration
save_iters = 100  # the frequency of saving a model
lr_decay_iters = 20000 # the frequency of decay the learning rate
actor_lr = 0.00001  # learning rate of actor model
critic_lr = 0.00005   # learning rate of critic model
first_train = False  # whether initialize a new model or load a existing model
vgg_layer = 30  # the output layer index of pre-trained vgg model
save_path_actor = "./weights/RL_model/actor.pkl"
save_path_critic = "./weights/RL_model/critic.pkl"

if first_train == True:
    actor = Actor(state_size, action_size).cuda()
    critic = Critic(state_size, action_size).cuda()
    print("model initialized")
else:
    actor = torch.load(save_path_actor).cuda()
    critic = torch.load(save_path_critic).cuda()
    print("model loaded")

optimizerA = optim.Adam(actor.parameters(), lr=actor_lr)
optimizerC = optim.Adam(critic.parameters(), lr=critic_lr)

resnet50 = models.resnet50(pretrained=True).cuda()
resnet50.eval()

normalization = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])
])

score = 0  # show the final score of this iteration

for iter in range(e_iters):

    ####### need to modify ##################
    env.reset()
    ob = env.render(mode="rgb_array")

    #########  generate the first state ############
    ob = Image.fromarray(ob)
    state = state_generate(ob, resnet50)

    log_probs = []
    values = []
    rewards = []
    masks = []
    entropy = 0

    for i in count():
        # everything in this loop is in one iteration

        # generate the currrent action
        dist, value = actor(state), critic(state)
        action = dist.sample()
        torch.cuda.empty_cache()

        ####### need to modify ##################
        # execute the action
        # the observation returned by env.step is the observation after the action has been executed, this is why it is called next state
        _, reward, done, _ = env.step(action.cpu().numpy())
        # the RGB observation is still the one after the action has been executed 
        next_ob = env.render(mode="rgb_array")

        ######## generate the state of the next frame #################
        next_ob = Image.fromarray(next_ob)
        next_state = state_generate(next_ob, resnet50)

        log_prob = dist.log_prob(action).unsqueeze(0)
        entropy += dist.entropy().mean()

        # record the log_prob, value, reward, mask
        log_probs.append(log_prob)
        values.append(value)
        rewards.append(torch.tensor([reward], dtype=torch.float, device=device))
        masks.append(torch.tensor([1 - done], dtype=torch.float, device=device))

        # convert the state
        state = next_state

        # condition to leave the loop of this iteration
        if done:
            score = int(0.9 * score + 0.1 * i)
            print('Iteration: {}, Score: {}'.format(iter, score))
            break

    # this iteration is finished here (done==True, game is over, etc.)
    # next_value is used as the estimated return value of the next state of the states(-1), to calculate the return of states(-1) 
    next_value = critic(next_state)
    returns = compute_returns(next_value, rewards, masks)

    # [n, 1] all of these 4 variables below
    log_probs = torch.cat(log_probs)
    returns = torch.cat(returns).detach()
    values = torch.cat(values)
    advantage = returns - values

    # single value
    actor_loss = -(log_probs * advantage.detach()).mean()
    critic_loss = advantage.pow(2).mean()

    optimizerA.zero_grad()
    optimizerC.zero_grad()
    actor_loss.backward(retain_graph=True)
    critic_loss.backward(retain_graph=True)
    optimizerA.step()
    optimizerC.step()
    torch.cuda.empty_cache()

    if (iter % save_iters == 0) and (iter > 0):
        torch.save(actor, save_path_actor)
        torch.save(critic, save_path_critic)
        print("model is saved")

    if (iter % lr_decay_iters == 0) and (iter > 0):
        print("learning rate decayed, ", end="")
        for g in optimizerA.param_groups:
            g["lr"] = g["lr"] * 0.99
            print("actor: " + str(g["lr"]) + ", ", end="")

        for g in optimizerC.param_groups:
            g["lr"] = g["lr"] * 0.99
            print("critic: " + str(g["lr"]), end="")

env.close(
