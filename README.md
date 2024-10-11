# Battleship AI Game

This project is a digital version of the classic Battleship game with advanced AI strategies implemented using **Genetic Algorithms**, **Fuzzy Logic**, **Minimax Algorithm**, and basic search techniques. The AI plays against a human player or other AI opponents, offering three levels of difficulty: **Easy**, **Medium**, and **Hard**.

## Features

- **Play against AI**: The game allows the human player to compete against AI opponents that use different search and optimization strategies.
- **AI Strategies**:
  - **Minimax Algorithm**: Utilized in the Easy mode, with depth-limited search and decision-making based on potential future game states.
  - **Basic AI**: In Medium mode, the AI uses simple heuristic and pattern-based search techniques to optimize moves.
  - **Fuzzy Logic**: The Hard mode uses fuzzy logic to adapt dynamically to uncertain game states and optimize targeting.
  - **Genetic Algorithm**: AI strategically places its ships using genetic algorithms for effective ship placement optimization.
- **AI vs. AI Battles**: Demonstrations of AI modes competing against each other (e.g., Basic AI vs. Fuzzy Logic AI).
- **User Interface**: Includes a drag-and-drop interface for the human player to place ships and intuitive grids for gameplay.
- **Gameplay Modes**:
  - Easy: Minimax AI
  - Medium: Basic AI
  - Hard: Fuzzy Logic AI

## Game Interface

- **Player Grids**:
  - **Attacking Grid**: Displays the player's attempts to hit AI ships.
  - **Own Grid**: Displays the player's ship placements.
- **AI Grids**:
  - **Computer's Attacking Grid**: Displays the AI's attempts to hit the player's ships.
- **Results Section**: Displays statistics like hits, misses, and sinks for both players.

## AI Techniques

1. **Genetic Algorithm**: Used for strategic ship placement by optimizing based on fitness functions like ship distribution and distance from edges/corners.
2. **Basic Search**: Implements heuristic search techniques to prioritize moves around known hits.
3. **Fuzzy Logic**: Adds adaptability by calculating probabilities based on the number of neighboring hits and unknowns.
4. **Minimax Algorithm**: Provides depth-limited decision-making, alternating between maximizing gains and minimizing losses.

## Installation and Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/Battleship-AI-Game.git
