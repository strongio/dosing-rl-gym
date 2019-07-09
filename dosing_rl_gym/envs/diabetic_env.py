#!/usr/bin/env python
# -*- coding: utf-8 -*-

# core modules
import os
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from scipy.integrate import odeint
from scipy.stats import truncnorm
from scipy.signal import savgol_filter


class Diabetic0Env(gym.Env):

    """
    Description:
        See dosing_rl/resources/Diabetic Background.ipynb
    Source:
        See dosing_rl/resources/Diabetic Background.ipynb
    Observation:
        Type: Box(9)
                                                                Min         Max
        0	Blood Glucose                                       0           Inf
        1	Remote Insulin                                      0           Inf
        2	Plasma Insulin                                      0           Inf
        3	S1                                                  0           Inf
        4	S2                                                  0           Inf
        5	Gut blood glucose                                   0           Inf
        6	Meal disturbance                                    0           Inf
        7	Previous Blood glucose                              0           Inf
        8   Previous meal disturbance                           0           Inf

    Actions:
        Type: Continuous
        Administered Insulin pump [mU/min]

    Reward:
        smooth function centered at 80 (self.target)
    Starting State:
        http://apmonitor.com/pdc/index.php/Main/DiabeticBloodGlucose
    Episode Termination:
        self.episode_length reached
    """

    def __init__(self):
        self.__version__ = "0.0.1"

        # General variables defining the environment
        self.base_path = os.path.dirname(os.path.abspath(__file__))
        self.seed()

        # Lists that will hold episode data
        self.opt_states = None
        self.G = None
        self.X = None
        self.I = None
        self.d = None
        self.u = None

        # Defining possible actions
        self.action_space = spaces.Box(0.0, 10.0, shape=(1,))

        # Defining observation space
        lows = np.zeros(9)
        highs = np.ones(9) * np.inf
        self.observation_space = spaces.Box(lows, highs)

        # Reward definitions
        self.target = 80.
        self.lb = 65.
        self.ub = 105.

        # Store what the agent tried
        self.curr_step = 0
        self.are_we_done = False

    def set_episode_length(self, minute_interval):
        """
        :param minute_interval: how often we will record information, make a recommendation
        The smaller this is, the longer an episode (patient trajectory) is
        :return:
        """
        self.minute_interval = minute_interval
        ns = int((24 * 60) / self.minute_interval) + 1
        # Final Time (hr)
        tf = 24  # simulate for 24 hours
        # Time Interval (min)
        self.t = np.linspace(0, tf, ns)
        self.episode_length = len(self.t)

    def step(self, action):
        """
        The agent takes a step in the environment.

        Parameters
        ----------
        action : list of length 1

        Returns
        -------
        observation (state), reward, episode_over, info : tuple
            ob (object) :
                an environment-specific object representing your observation of
                the environment.
            reward (float) :
                amount of reward achieved by the previous action. The scale
                varies between environments, but the goal is always to increase
                your total reward.
            episode_over (bool) :
                whether it's time to reset the environment again. Most (but not
                all) tasks are divided up into well-defined episodes, and done
                being True indicates the episode has terminated. (For example,
                perhaps the pole tipped too far, or you lost your last life.)
            info (dict) :
                 diagnostic information useful for debugging. It can sometimes
                 be useful for learning (for example, it might contain the raw
                 probabilities behind the environment's last state change).
                 However, official evaluations of your agent are not allowed to
                 use this for learning.
        """

        #if self.are_we_done:
            #raise RuntimeError("Episode is done")

        if self.opt_states is None:
            raise Exception("You need to reset() the environment before calling step()!")

        # add new action to dose list
        self.u.append(action[0])

        # check if we're done
        if self.curr_step >= self.episode_length - 2:
            self.are_we_done = True

        # get updated time step
        ts = [self.t[self.curr_step], self.t[self.curr_step + 1]]

        # simulate
        y = odeint(diabetic, self.opt_states, ts, args=(self.u[self.curr_step + 1], self.d[self.curr_step + 1]))

        # update lists with updated state values
        self.G.append(y[-1][0])
        self.X.append(y[-1][1])
        self.I.append(y[-1][2])
        self.opt_states = y[-1]

        state = np.array([self.opt_states[0],
                          self.opt_states[1],
                          self.opt_states[2],
                          self.opt_states[3],
                          self.opt_states[4],
                          self.opt_states[5],
                          self.d[self.curr_step + 1],
                          self.G[-2],
                          self.d[self.curr_step]
                          ])

        reward = self._get_reward()

        # increment episode
        self.curr_step += 1

        return state, reward, self.are_we_done, {}

    def _get_reward(self):
        """
        Reward is based on smooth function.
        Target blood glucose level: 80
        g parameter will change slope: 0.7
        """
        g = .7
        r = 1 - np.tanh(np.abs((self.G[-1] - self.target) / g) * .1) ** 2
        if (self.G[-1] < self.lb) or (self.G[-1] > self.ub):
            r = -1.

        return r

    def reset(self):
        """
        Reset the state of the environment and returns an initial observation (state)
        """

        self.curr_step = 0
        self.are_we_done = False

        # Steady State Initial Conditions for the States
        self.opt_states = np.array([self.np_random.uniform(self.lb, self.ub),
                                    self.np_random.normal(33.33, 5),
                                    self.np_random.normal(33.33, 5),
                                    self.np_random.normal(25., 5),
                                    self.np_random.normal(25., 5),
                                    self.np_random.normal(250., 5)])

        # Steady State Initial Condition for the Control
        first_dose = 3.0  # mU/min

        # Steady State for the Disturbance
        d_ss = 1000.0  # mmol/L-min

        # Store results for plotting
        self.G = [self.opt_states[0]]
        self.X = [self.opt_states[1]]
        self.I = [self.opt_states[2]]
        self.d = np.ones(self.episode_length) * d_ss
        self.u = [first_dose]

        # Constant meal disturbance vector
        meals = [1259,1451,1632,1632,1468,1314,1240,1187,1139,1116,\
                  1099,1085,1077,1071,1066,1061,1057,1053,1046,1040,\
                  1034,1025,1018,1010,1000,993,985,976,970,964,958,\
                  954,952,950,950,951,1214,1410,1556,1603,1445,1331,\
                  1226,1173,1136,1104,1088,1078,1070,1066,1063,1061,\
                  1059,1056,1052,1048,1044,1037,1030,1024,1014,1007,\
                  999,989,982,975,967,962,957,953,951,950,1210,1403,\
                  1588,1593,1434,1287,1212,1159,1112,1090,1075,1064,\
                  1059,1057,1056,1056,1056,1055,1054,1052,1049,1045,\
                  1041,1033,1027,1020,1011,1003,996,986]

        for i in range(len(meals)):
            self.d[i+43] = meals[i]

        state, _, _, _ = self.step(self.u)

        return state

    def seed(self, seed=None):
        self.np_random, _ = seeding.np_random(seed)


class Diabetic1Env(gym.Env):

    """
    Description:
        See dosing_rl/resources/Diabetic Background.ipynb
    Source:
        See dosing_rl/resources/Diabetic Background.ipynb
    Observation:
        Type: Box(9)
                                                                Min         Max
        0	Blood Glucose                                       0           Inf
        1	Remote Insulin                                      0           Inf
        2	Plasma Insulin                                      0           Inf
        3	S1                                                  0           Inf
        4	S2                                                  0           Inf
        5	Gut blood glucose                                   0           Inf
        6	Meal disturbance                                    0           Inf
        7	Previous Blood glucose                              0           Inf
        8   Previous meal disturbance                           0           Inf

    Actions:
        Type: Continuous
        Administered Insulin pump [mU/min]

    Reward:
        smooth function centered at 80 (self.target)
    Starting State:
        http://apmonitor.com/pdc/index.php/Main/DiabeticBloodGlucose
    Episode Termination:
        self.episode_length reached
    """

    def __init__(self):
        self.__version__ = "0.0.1"

        # General variables defining the environment
        self.base_path = os.path.dirname(os.path.abspath(__file__))
        self.seed()

        # Lists that will hold episode data
        self.opt_states = None
        self.G = None
        self.X = None
        self.I = None
        self.d = None
        self.u = None

        # Defining possible actions
        self.action_space = spaces.Box(0.0, 10.0, shape=(1,))

        # Defining observation space
        lows = np.zeros(9)
        highs = np.ones(9) * np.inf
        self.observation_space = spaces.Box(lows, highs)

        # Reward definitions
        self.target = 80.
        self.lb = 65.
        self.ub = 105.

        # Store what the agent tried
        self.curr_step = 0
        self.are_we_done = False

    def set_episode_length(self, minute_interval):
        """
        :param minute_interval: how often we will record information, make a recommendation
        The smaller this is, the longer an episode (patient trajectory) is
        :return:
        """
        self.minute_interval = minute_interval
        ns = int((24 * 60) / self.minute_interval) + 1
        # Final Time (hr)
        tf = 24  # simulate for 24 hours
        # Time Interval (min)
        self.t = np.linspace(0, tf, ns)
        self.episode_length = len(self.t)

    def step(self, action):
        """
        The agent takes a step in the environment.

        Parameters
        ----------
        action : list of length 1

        Returns
        -------
        observation (state), reward, episode_over, info : tuple
            ob (object) :
                an environment-specific object representing your observation of
                the environment.
            reward (float) :
                amount of reward achieved by the previous action. The scale
                varies between environments, but the goal is always to increase
                your total reward.
            episode_over (bool) :
                whether it's time to reset the environment again. Most (but not
                all) tasks are divided up into well-defined episodes, and done
                being True indicates the episode has terminated. (For example,
                perhaps the pole tipped too far, or you lost your last life.)
            info (dict) :
                 diagnostic information useful for debugging. It can sometimes
                 be useful for learning (for example, it might contain the raw
                 probabilities behind the environment's last state change).
                 However, official evaluations of your agent are not allowed to
                 use this for learning.
        """

        #if self.are_we_done:
            #raise RuntimeError("Episode is done")

        if self.opt_states is None:
            raise Exception("You need to reset() the environment before calling step()!")

        # add new action to dose list
        self.u.append(action[0])

        # check if we're done
        if self.curr_step >= self.episode_length - 2:
            self.are_we_done = True

        # get updated time step
        ts = [self.t[self.curr_step], self.t[self.curr_step + 1]]

        # simulate
        y = odeint(diabetic, self.opt_states, ts, args=(self.u[self.curr_step + 1], self.d[self.curr_step + 1]))

        # update lists with updated state values
        self.G.append(y[-1][0])
        self.X.append(y[-1][1])
        self.I.append(y[-1][2])
        self.opt_states = y[-1]

        state = np.array([self.opt_states[0],
                          self.opt_states[1],
                          self.opt_states[2],
                          self.opt_states[3],
                          self.opt_states[4],
                          self.opt_states[5],
                          self.d[self.curr_step + 1],
                          self.G[-2],
                          self.d[self.curr_step]
                          ])

        reward = self._get_reward()

        # increment episode
        self.curr_step += 1

        return state, reward, self.are_we_done, {}

    def _get_reward(self):
        """
        Reward is based on smooth function.
        Target blood glucose level: 80
        g parameter will change slope: 0.7
        """
        g = .7
        r = 1 - np.tanh(np.abs((self.G[-1] - self.target) / g) * .1) ** 2
        if (self.G[-1] < self.lb) or (self.G[-1] > self.ub):
            r = -1.

        return r

    def reset(self):
        """
        Reset the state of the environment and returns an initial observation (state)
        """

        self.curr_step = 0
        self.are_we_done = False

        # Steady State Initial Conditions for the States
        self.opt_states = np.array([self.np_random.uniform(self.lb, self.ub),
                                    self.np_random.normal(33.33, 5),
                                    self.np_random.normal(33.33, 5),
                                    self.np_random.normal(25., 5),
                                    self.np_random.normal(25., 5),
                                    self.np_random.normal(250., 5)])

        # Steady State Initial Condition for the Control
        first_dose = self.np_random.choice([0., 1., 2., 3., 4., 5., 6.]) # mU/min

        # Steady State for the Disturbance
        d_ss = self.np_random.uniform(800.0, 1200.0) # mmol/L-min

        # Store results for plotting
        self.G = [self.opt_states[0]]
        self.X = [self.opt_states[1]]
        self.I = [self.opt_states[2]]

        self.u = [first_dose]

        # Probabilistic meal disturbance vector
        self.d = get_meal_data(iterations=1,
                               base_value=d_ss,
                               meal_length=self.episode_length,
                               rs=self.np_random,
                               minute_interval=self.minute_interval)

        state, _, _, _ = self.step(self.u)

        return state

    def seed(self, seed=None):
        self.np_random, _ = seeding.np_random(seed)


def get_meal_data(iterations=1, base_value=1000., meal_length=1, rs=None, minute_interval=10):
    """
    Create smoothed meal disturbance vector based on probability distribution over time

    :param iterations: how many times to sample from meal distribution
    :param base_value: steady state value for meal disturbance
    :param meal_length: how long is meal disturbance vector
    :param rs: numpy random state object
    :return: vector of meal disturbances
    """

    scenarios = []
    for i in range(iterations):
        random_gen = rs

        scenario = {'meal': {'time': [], 'amount': []}}

        # Probability of taking each meal
        # [breakfast, snack1, lunch, snack2, dinner, snack3]
        prob = [0.95, 0.2, 0.95, 0.2, 0.95, 0.2]
        time_lb = np.array([5, 9, 11, 14, 16.5, 20]) * int(60 / minute_interval) + 1
        time_ub = np.array([9, 10, 13, 16, 29.5, 22]) * int(60 / minute_interval) + 1
        time_mu = np.array([7, 9.5, 12, 15, 18, 21]) * int(60 / minute_interval) + 1
        time_sigma = np.array([10, 10, 10, 10, 10, 10])
        amount_mu = [2500, 1250, 2800, 1250, 2800, 1250]
        amount_sigma = [100, 50, 100, 50, 100, 50]

        for p, tlb, tub, tbar, tsd, mbar, msd in zip(prob, time_lb, time_ub, time_mu, time_sigma, amount_mu,
                                                     amount_sigma):
            if random_gen.rand() < p:
                tmeal = np.round(truncnorm.rvs(a=(tlb - tbar) / tsd,
                                               b=(tub - tbar) / tsd,
                                               loc=tbar,
                                               scale=tsd,
                                               random_state=rs))

                scenario['meal']['time'].append(int(tmeal))
                scenario['meal']['amount'].append(max(round(rs.normal(mbar, msd)), 0))

        scenarios.append(scenario)
    # pre fill meal over time list
    meall = list(np.ones(meal_length) * int(base_value))

    for scenario in scenarios:
        for i in range(len(scenario['meal']['time'])):
            # avoid out of index meal on last time
            if scenario['meal']['time'][i] >= meal_length - 2:
                scenario['meal']['time'][i] = meal_length - 2
            # store meal from sampled scenario
            meall[scenario['meal']['time'][i]] = scenario['meal']['amount'][i]

    # smooth meal over time distribution
    tr_meall = savgol_filter(meall, 11, 3)
    tr_meall[tr_meall < base_value] = base_value

    return tr_meall


def diabetic(y, t, ui, dhat):
    """
    Expanded Bergman Minimal model to include meals and insulin
    Parameters for an insulin dependent type-I diabetic

    States (6):
    In non-diabetic patients, the body maintains the blood glucose
    level at a range between about 3.6 and 5.8 mmol/L (64.8 and
    104.4 mg/dL with 1:18 conversion between mmol/L and mg/dL)

    :param y: input state
    :param t: time step
    :param ui: Insulin infusion rate (mU/min)
    :param dhat: Meal disturbance (mmol/L-min)

    :return: change in states
    """

    g = y[0]            # blood glucose (mg/dL)
    x = y[1]            # remote insulin (micro-u/ml)
    i = y[2]            # plasma insulin (micro-u/ml)
    q1 = y[3]           # S1
    q2 = y[4]           # S2
    g_gut = y[5]        # gut blood glucose (mg/dl)

    # Parameters:
    gb = 291.0     # (mg/dL)                    Basal Blood Glucose
    p1 = 3.17e-2   # 1/min
    p2 = 1.23e-2   # 1/min
    si = 2.9e-2    # 1/min * (mL/micro-U)
    ke = 9.0e-2    # 1/min                      Insulin elimination from plasma
    kabs = 1.2e-2  # 1/min                      t max,G inverse
    kemp = 1.8e-1  # 1/min                      t max,I inverse
    f = 8.00e-1    # L
    vi = 12.0      # L                          Insulin distribution volume
    vg = 12.0      # L                          Glucose distibution volume

    # Compute ydot:
    dydt = np.empty(6)

    dydt[0] = -p1 * (g - gb) - si * x * g + f * kabs / vg * g_gut + f / vg * dhat  # (1)
    dydt[1] = p2 * (i - x)  # remote insulin compartment dynamics (2)
    dydt[2] = -ke * i + ui  # plasma insulin concentration  (3)
    dydt[3] = ui - kemp * q1  # two-part insulin absorption model dS1/dt
    dydt[4] = -kemp * (q2 - q1)  # two-part insulin absorption model dS2/dt
    dydt[5] = kemp * q2 - kabs * g_gut

    # convert from minutes to hours
    dydt = dydt * 60
    return dydt
