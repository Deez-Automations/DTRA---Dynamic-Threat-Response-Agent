# -*- coding: utf-8 -*-
"""
DTRA Decider Module
A* search and Q-Learning agent for autonomous threat response decisions.
"""

import numpy as np
import random
import os

from config import BUSINESS_COSTS, RL_CONFIG, Q_TABLE_PATH


class AStarDecider:
    """
    A* algorithm-based decision maker.
    Balances business cost (g) with security risk (h) to find optimal action.
    """
    
    def __init__(self, costs=None):
        self.costs = costs or BUSINESS_COSTS
        
    def calculate_security_risk(self, action, danger_score):
        """Calculate h(n) - the security risk heuristic."""
        if action == 'Ignore':
            return danger_score * 100  # High risk if ignoring danger
        elif action == 'Log':
            return danger_score * 90   # Slightly lower risk
        elif action == 'Block':
            return (1.0 - danger_score) * 20  # Risk of false positive
        elif action == 'Isolate':
            return 0  # No security risk - threat contained
        return 0
    
    def decide(self, danger_score, verbose=False):
        """
        Find the best action using A* search.
        
        Args:
            danger_score: float 0.0 - 1.0 representing threat probability
            verbose: print decision reasoning
            
        Returns:
            tuple: (best_action, total_cost)
        """
        if verbose:
            print(f"--- A* Decider: Evaluating {danger_score*100:.1f}% danger ---")
        
        results = []
        
        for action, g_cost in self.costs.items():
            h_cost = self.calculate_security_risk(action, danger_score)
            f_cost = g_cost + h_cost
            
            if verbose:
                print(f"  {action:<8} | f={f_cost:>6.2f} (g={g_cost:<2} + h={h_cost:<6.2f})")
            
            results.append((f_cost, action))
        
        best_cost, best_action = min(results)
        
        if verbose:
            print(f"  => Best Action: {best_action} (Cost: {best_cost:.2f})")
        
        return best_action, best_cost


class QLearningAgent:
    """
    Reinforcement Learning agent using Q-Learning.
    Learns optimal response policy through experience.
    """
    
    def __init__(self, config=None):
        cfg = config or RL_CONFIG
        
        self.num_states = cfg['num_states']
        self.actions = cfg['actions']
        self.num_actions = len(self.actions)
        self.alpha = cfg['alpha']
        self.gamma = cfg['gamma']
        self.epsilon = cfg['epsilon']
        
        # Initialize Q-Table (state x action matrix)
        self.q_table = np.zeros((self.num_states, self.num_actions))
        self.reward_history = []
        
    def get_reward(self, state, action_index):
        """
        Calculate reward for taking an action in a state.
        Encodes SOC best practices.
        """
        action = self.actions[action_index]
        danger_level = state + 1  # Convert 0-9 to 1-10
        
        # Low Danger (0-30%): Should ignore or log
        if danger_level <= 3:
            if action in ['Ignore', 'Log']:
                return 0   # Correct - no alert fatigue
            else:
                return -20  # Penalty - false positive
        
        # Medium Danger (40-70%): Should block
        elif danger_level <= 7:
            if action == 'Block':
                return 10   # Good cautious response
            elif action == 'Isolate':
                return -50  # Too aggressive
            else:
                return -10  # Too passive
        
        # High Danger (80-100%): Must block or isolate
        else:
            if action in ['Block', 'Isolate']:
                return 50   # Correct - threat contained
            elif action == 'Ignore':
                return -100  # Critical failure
            else:
                return -30   # Insufficient response
        
        return -1
    
    def train(self, num_episodes=None, verbose=True):
        """Train the agent through simulated episodes."""
        episodes = num_episodes or RL_CONFIG['num_episodes']
        
        if verbose:
            print(f"🎮 Training Q-Learning Agent ({episodes:,} episodes)...")
        
        for episode in range(episodes):
            # Random alert arrives
            state = random.randint(0, self.num_states - 1)
            
            # Epsilon-greedy action selection
            if random.uniform(0, 1) < self.epsilon:
                action = random.randint(0, self.num_actions - 1)  # Explore
            else:
                action = np.argmax(self.q_table[state, :])  # Exploit
            
            # Get reward and update Q-table
            reward = self.get_reward(state, action)
            
            old_q = self.q_table[state, action]
            new_q = old_q + self.alpha * (
                reward + self.gamma * np.max(self.q_table[state, :]) - old_q
            )
            self.q_table[state, action] = new_q
            
            self.reward_history.append(reward)
        
        if verbose:
            print("✅ Training complete!")
            self.print_policy()
    
    def print_policy(self):
        """Print the learned optimal policy."""
        print("\n📋 Learned Optimal Policy:")
        print("-" * 40)
        for state in range(self.num_states):
            best_action_idx = np.argmax(self.q_table[state, :])
            danger_range = f"{state*10}%-{state*10+10}%"
            print(f"  {danger_range:<15} → {self.actions[best_action_idx]}")
        print("-" * 40)
    
    def decide(self, danger_score):
        """
        Get the optimal action for a given danger score.
        
        Args:
            danger_score: float 0.0 - 1.0
            
        Returns:
            str: action name ('Ignore', 'Log', 'Block', 'Isolate')
        """
        state = min(int(danger_score * 10), self.num_states - 1)
        action_idx = np.argmax(self.q_table[state, :])
        return self.actions[action_idx]
    
    def save(self, path=None):
        """Save Q-table to disk."""
        save_path = path or Q_TABLE_PATH
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.save(save_path, self.q_table)
        print(f"✅ Saved Q-table to {save_path}")
    
    def load(self, path=None):
        """Load Q-table from disk."""
        load_path = path or Q_TABLE_PATH
        if not os.path.exists(load_path):
            raise FileNotFoundError(
                f"Q-table not found at {load_path}. "
                "Please train the agent first using train.py"
            )
        self.q_table = np.load(load_path)
        print("✅ Loaded Q-table")
        return self


# Combined decider that uses both A* and RL
class HybridDecider:
    """
    Combines A* reasoning with Q-Learning policy.
    Uses RL for fast decisions, A* for explainability.
    """
    
    def __init__(self):
        self.astar = AStarDecider()
        self.rl_agent = QLearningAgent()
        
    def load(self):
        """Load pre-trained RL agent."""
        self.rl_agent.load()
        return self
    
    def decide(self, danger_score, method='rl'):
        """
        Get action recommendation.
        
        Args:
            danger_score: float 0.0 - 1.0
            method: 'rl' for Q-Learning, 'astar' for A*, 'both' for comparison
            
        Returns:
            str or dict: action recommendation(s)
        """
        if method == 'rl':
            return self.rl_agent.decide(danger_score)
        elif method == 'astar':
            action, cost = self.astar.decide(danger_score)
            return action
        else:
            return {
                'rl': self.rl_agent.decide(danger_score),
                'astar': self.astar.decide(danger_score)[0]
            }
