# Riftbound Sim

## Project Overview
Genetic algorithm + ML + deep RL simulator for the Riftbound TCG that finds the optimal competitive deck. Uses island model evolution, expert heuristic AI, and PPO self-play training.

## Key Rules
- 2 battlefields (Left, Right), victory at 8 points
- Energy 1-12, rune slots 12 (split 6/6 between legend's two domains)
- Draw 4, mulligan up to 2, D20 for turn order
- P1 channels 2 runes turn 1, P2 channels 3 (going-second compensation), then 2/turn after
- Runes accumulate (not refresh)
- 5-step combat: cleanup → deathknell/damage triggers → results → control established → combat ends
- Signature spells/gears locked to their legend
- Banned: Draven - Vanquisher, Scrapheap, Called Shot, Fight or Flight

## Running
- `python run_massive.py` — full pipeline
- `python -m streamlit run app.py --server.headless true` — dashboard
- `python fetch_cards.py` — re-fetch cards from API
