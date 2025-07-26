# 🤖 Q-Learning for Robot Navigation in Static Environment

This project implements the **Q-Learning algorithm** on a physical **E-Puck robot** to learn optimal navigation strategies in static environments. It is based on the undergraduate thesis titled:  
**"Implementation of Q-Learning Algorithm on Wheeled Robot for Navigation Strategy in Static Environment"**  
by **Haekal Fadhilah Ardandi**, under the supervision of **Dr. Eng. Gembong Edhi Setyawan, S.T., M.T.**

---

## 📌 Research Motivation

Classical robot navigation algorithms like A*, Dijkstra, and Greedy face limitations in adaptability and computational efficiency. This research explores the use of **Reinforcement Learning**, specifically **Q-Learning**, to allow autonomous robots to learn navigation strategies through interaction with their environment.

The **E-Puck robot** was selected due to its flexible mobility (differential wheels) and easy communication via Bluetooth.

---

## 🎯 Objectives

- Evaluate the effectiveness of Q-Learning for optimal pathfinding in static environments.
- Compare learned paths against theoretical minimum steps.
- Analyze the effect of environmental obstacles on learning speed and strategy formation.

---

## 🧪 Methodology

- **Robot Platform**: E-Puck  
- **Environment**: 8x8 Grid Arena  
- **Scenarios**:
  1. Empty grid (no obstacles)
  2. Grid with scattered obstacles
  3. Grid with narrow path (obstacles forming a maze)

### Q-Learning Setup

- **States**: 6x6 grid positions  
- **Actions**: Move North, South, East, West  
- **Rewards**:
  - +100: Goal reached  
  - -100: Collision with obstacle  
  - -1: Normal step  

- **Training Episodes**: 500  
- **Learning Rate (α)**: 0.1  
- **Discount Factor (γ)**: 0.9  
- **Policy**: Epsilon-Greedy

---

## 📊 Results & Analysis

| Environment        | Convergence | Success Rate | Actions (avg) | Reward (avg) |
|--------------------|-------------|---------------|----------------|---------------|
| No Obstacles       | Ep. 70–75   | 99.2% (496/500) | 9–11           | 76–88         |
| Scattered Obstacles| Ep. 75–80   | 97.6% (488/500) | 10–12          | 86–91         |
| Narrow Path        | Ep. 131–136 | 90.0% (450/500) | 13–17          | 84–88         |

- **Q-Learning** achieved high success rates (90–99.2%) across all environments.
- The number of actions required matched theoretical minimums:
  - Env 1: 9 steps  
  - Env 2: 10 steps  
  - Env 3: 13 steps

---

## 🧠 Insights

- **Obstacle Complexity Impacts Learning**:
  - No obstacles → fast learning, few errors.
  - Scattered obstacles → more early mistakes, but sufficient exploration space.
  - Narrow path → slow early learning due to dense obstacles, but still converges.

- The learned strategies successfully converged to optimal paths, even when navigational routes varied from theoretical shortest paths.

---

## 🛠️ Tech Stack

- Python (Q-table simulation and visualization)  
- Image processing tools for map generation (e.g., Gaussian Blur, HSV Thresholding, Canny Edge Detection, Hough Transform)  
- E-Puck Robot with Bluetooth interface

---
