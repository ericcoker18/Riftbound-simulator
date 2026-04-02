"""
Riftbound Sim — Streamlit Dashboard

Launch with:  python -m streamlit run app.py --server.headless true
"""

import streamlit as st
import os
import time as _time
from collections import Counter
from streamlit_autorefresh import st_autorefresh

st.set_page_config(
    page_title="Riftbound Sim",
    page_icon="https://riftbound.leagueoflegends.com/favicon.ico",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Theme colors
# ---------------------------------------------------------------------------

ACCENT      = "#d4a539"    # warm gold
ACCENT2     = "#8b6914"    # dark gold
BG_DARK     = "#111318"    # near-black
BG_CARD     = "#181b22"    # card background
BG_CARD2    = "#1e222b"    # lighter card
BORDER      = "#2a2e38"    # subtle border
TEXT_DIM    = "#7a7f8d"    # muted text
TEXT_BRIGHT = "#e8e8ec"    # bright text

# Domain colors
DOMAIN_COLORS = {
    "Fury": "#cf4444", "Order": "#d4a539", "Chaos": "#9b59b6",
    "Body": "#3daa5c", "Calm": "#4a9bd9", "Mind": "#2ec4b6",
}

PIE_COLORS = ["#d4a539", "#cf4444", "#4a9bd9"]

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------

st.markdown(f"""
<style>
    [data-testid="stSidebar"] {{ display: none; }}
    .stApp {{ background-color: {BG_DARK}; }}

    .card {{
        background: {BG_CARD};
        border: 1px solid {BORDER};
        border-radius: 10px;
        padding: 1.4rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 12px rgba(0,0,0,0.4);
    }}
    .card h3 {{
        color: {ACCENT};
        margin-top: 0;
        font-size: 1rem;
        font-weight: 600;
    }}

    .stat-box {{
        background: {BG_CARD};
        border: 1px solid {BORDER};
        border-radius: 10px;
        padding: 1.1rem 0.8rem;
        text-align: center;
        box-shadow: 0 2px 10px rgba(0,0,0,0.3);
        border-top: 3px solid {ACCENT};
    }}
    .stat-box .number {{
        font-size: 1.8rem;
        font-weight: 700;
        color: {TEXT_BRIGHT};
    }}
    .stat-box .label {{
        font-size: 0.75rem;
        color: {TEXT_DIM};
        text-transform: uppercase;
        letter-spacing: 1.2px;
        margin-top: 0.3rem;
    }}

    .main-title {{
        font-size: 2rem;
        font-weight: 800;
        color: {TEXT_BRIGHT};
        margin-bottom: 0.1rem;
    }}
    .main-title span {{
        color: {ACCENT};
    }}
    .subtitle {{
        color: {TEXT_DIM};
        font-size: 0.95rem;
        margin-bottom: 1.5rem;
    }}

    .section-header {{
        font-size: 1.15rem;
        font-weight: 700;
        color: {TEXT_BRIGHT};
        border-left: 3px solid {ACCENT};
        padding-left: 0.7rem;
        margin: 1.5rem 0 0.8rem 0;
    }}

    /* Status panel */
    .status-panel {{
        background: {BG_CARD2};
        border: 1px solid {BORDER};
        border-left: 4px solid {ACCENT};
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin-bottom: 1rem;
    }}
    .status-phase {{
        font-size: 1rem;
        font-weight: 700;
        color: {ACCENT};
    }}
    .status-detail {{
        font-size: 0.85rem;
        color: {TEXT_DIM};
        margin-top: 0.3rem;
    }}
    .status-stats {{
        display: flex;
        gap: 1.5rem;
        margin-top: 0.6rem;
        flex-wrap: wrap;
    }}
    .status-stat {{
        font-size: 0.8rem;
        color: {TEXT_DIM};
    }}
    .status-stat strong {{
        color: {TEXT_BRIGHT};
    }}

    .status-complete {{
        border-left-color: #3daa5c;
    }}
    .status-complete .status-phase {{
        color: #3daa5c;
    }}

    /* Progress bar override */
    .stProgress > div > div > div {{
        background-color: {ACCENT};
    }}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Auto-refresh when simulation is running
# ---------------------------------------------------------------------------

def _is_sim_running():
    """Check if a simulation is actively running."""
    import json
    if os.path.exists("results/sim_status.json"):
        try:
            with open("results/sim_status.json", "r") as f:
                status = json.load(f)
            phase = status.get("phase", "")
            ts = status.get("timestamp", 0)
            age = _time.time() - ts
            # Running if not complete and updated within last 60 seconds
            return phase != "Complete" and age < 60
        except Exception:
            return False
    return False

IS_LOCAL = os.path.exists("data/cards.json")

if _is_sim_running():
    # Auto-refresh every 3 seconds while sim is running
    st_autorefresh(interval=3000, limit=None, key="sim_refresh")


# ---------------------------------------------------------------------------
# Top navigation
# ---------------------------------------------------------------------------

nav_cols = st.columns(5)
pages = ["Dashboard", "Card Pool", "Meta Decks", "Run Simulation", "Results"]

for i, p in enumerate(pages):
    if nav_cols[i].button(p, use_container_width=True):
        st.session_state["page"] = p

page = st.session_state.get("page", "Dashboard")
st.markdown(f'<div style="border-bottom: 1px solid {BORDER}; margin-bottom: 1.2rem;"></div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Shared loaders
# ---------------------------------------------------------------------------

@st.cache_data
def load_cards():
    import json
    with open("data/cards.json", "r", encoding="utf-8") as f:
        return json.load(f)

@st.cache_data
def load_legends_data():
    import json
    with open("data/legends.json", "r", encoding="utf-8") as f:
        return json.load(f)

@st.cache_data
def load_meta_decks_data():
    import json
    try:
        with open("data/meta_decks.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return []

def load_results():
    import json
    if os.path.exists("results/best_deck.json"):
        with open("results/best_deck.json", "r") as f:
            return json.load(f)
    return None

def load_sim_status():
    import json
    if os.path.exists("results/sim_status.json"):
        with open("results/sim_status.json", "r") as f:
            return json.load(f)
    return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def stat_box(label, value):
    st.markdown(f"""
    <div class="stat-box">
        <div class="number">{value}</div>
        <div class="label">{label}</div>
    </div>
    """, unsafe_allow_html=True)

def section_header(text):
    st.markdown(f'<div class="section-header">{text}</div>', unsafe_allow_html=True)

def chart_layout(fig, height=None):
    """Apply consistent dark theme to plotly charts."""
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color=TEXT_DIM,
        margin=dict(t=40, b=20, l=20, r=20),
        **({"height": height} if height else {}),
    )
    return fig

def sim_status_panel():
    """Render the simulation status panel. Always shows something."""
    status = load_sim_status()

    if not status:
        # No status file — show idle state
        result = load_results()
        if result:
            score = result.get("score", 0)
            legend = result.get("legend", "Unknown")
            st.markdown(f"""
            <div class="status-panel status-complete">
                <div class="status-phase">&#10003; Last Run Complete</div>
                <div class="status-detail">Best deck: {legend} at {score:.0%} win rate</div>
                <div class="status-stats">
                    <span class="status-stat">Run <code>python run_massive.py</code> to start a new simulation</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="status-panel">
                <div class="status-phase">&#9679; Ready</div>
                <div class="status-detail">No simulation has been run yet</div>
                <div class="status-stats">
                    <span class="status-stat">Run <code>python run_massive.py</code> or use the Run Simulation tab to start</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        return

    phase = status.get("phase", "Unknown")
    detail = status.get("detail", "")
    progress = status.get("progress", 0)
    ts = status.get("timestamp", 0)
    age = _time.time() - ts

    is_complete = phase == "Complete"

    # If stale (>60s without update) and not complete, show as stalled
    if age > 60 and not is_complete:
        st.markdown(f"""
        <div class="status-panel">
            <div class="status-phase">&#9888; Simulation Stalled</div>
            <div class="status-detail">Last update: {phase} — {detail} ({int(age)}s ago)</div>
        </div>
        """, unsafe_allow_html=True)
        return

    css_class = "status-panel status-complete" if is_complete else "status-panel"

    extra_html = ""
    stats = []
    if status.get("best_score"):
        stats.append(f'<span class="status-stat">Best: <strong>{status["best_score"]:.3f}</strong></span>')
    if status.get("avg_score"):
        stats.append(f'<span class="status-stat">Avg: <strong>{status["avg_score"]:.3f}</strong></span>')
    if status.get("stability") is not None and status.get("stability") > 0:
        stats.append(f'<span class="status-stat">Stability: <strong>{status["stability"]:.0%}</strong></span>')
    if status.get("legend"):
        stats.append(f'<span class="status-stat">Legend: <strong>{status["legend"][:25]}</strong></span>')
    if status.get("games_per_sec"):
        stats.append(f'<span class="status-stat">Speed: <strong>{status["games_per_sec"]:,}/s</strong></span>')
    if status.get("total_time_min"):
        stats.append(f'<span class="status-stat">Time: <strong>{status["total_time_min"]} min</strong></span>')
    if stats:
        extra_html = f'<div class="status-stats">{"".join(stats)}</div>'

    icon = "&#10003;" if is_complete else "&#9654;"

    st.markdown(f"""
    <div class="{css_class}">
        <div class="status-phase">{icon} {phase}</div>
        <div class="status-detail">{detail}</div>
        {extra_html}
    </div>
    """, unsafe_allow_html=True)

    if not is_complete and progress > 0:
        st.progress(progress)


# ===================================================================
# PAGE: Dashboard
# ===================================================================

if page == "Dashboard":
    st.markdown('<div class="main-title">Riftbound <span>Meta Deck Finder</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Genetic algorithm + ML + deep RL simulator for competitive deck optimization</div>', unsafe_allow_html=True)

    # Sim status
    sim_status_panel()

    cards = load_cards()
    legends = load_legends_data()
    meta = load_meta_decks_data()
    result = load_results()

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1: stat_box("Cards", len(cards))
    with col2: stat_box("Legends", len(set(l["champion_tag"] for l in legends)))
    with col3: stat_box("Meta Decks", len(meta))
    with col4: stat_box("RL Model", "Ready" if os.path.exists("models/rl_policy.pt") else "---")
    with col5: stat_box("Best Win Rate", f"{result['score']:.0%}" if result else "---")

    st.markdown("<br>", unsafe_allow_html=True)

    section_header("Card Pool Breakdown")
    type_counts = Counter(c.get("card_type", "Unit") for c in cards)
    domain_counts = Counter(c.get("domain", "None") for c in cards if c.get("domain"))

    col_a, col_b = st.columns(2)
    with col_a:
        import plotly.express as px
        fig = px.pie(names=list(type_counts.keys()), values=list(type_counts.values()),
                     title="By Card Type", hole=0.45, color_discrete_sequence=PIE_COLORS)
        st.plotly_chart(chart_layout(fig), width='stretch')

    with col_b:
        fig = px.bar(x=list(domain_counts.keys()), y=list(domain_counts.values()),
                     title="By Domain", labels={"x": "Domain", "y": "Count"},
                     color=list(domain_counts.keys()), color_discrete_map=DOMAIN_COLORS)
        chart_layout(fig)
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, width='stretch')

    if result:
        section_header("Current Best Deck")
        col1, col2 = st.columns([1, 3])
        with col1:
            stat_box("Legend", result.get("legend", "?").split(" - ")[0])
            st.markdown("<br>", unsafe_allow_html=True)
            stat_box("Win Rate", f"{result.get('score', 0):.0%}")
        with col2:
            deck_items = result.get("deck", [])
            if deck_items:
                import pandas as pd
                df = pd.DataFrame(deck_items, columns=["Card", "Count"])
                st.dataframe(df.sort_values("Count", ascending=False),
                             width='stretch', hide_index=True, height=300)


# ===================================================================
# PAGE: Card Pool
# ===================================================================

elif page == "Card Pool":
    st.markdown('<div class="main-title">Card Pool <span>Explorer</span></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="subtitle">Browse and filter all {len(load_cards())} cards</div>', unsafe_allow_html=True)

    cards = load_cards()

    with st.expander("Filters", expanded=True):
        col1, col2, col3, col4 = st.columns([1, 1, 1, 2])
        with col1:
            type_filter = st.multiselect("Card Type", ["Unit", "Spell", "Gear"], default=["Unit", "Spell", "Gear"])
        with col2:
            domains = sorted(set(c.get("domain", "") for c in cards if c.get("domain")))
            domain_filter = st.multiselect("Domain", domains, default=domains)
        with col3:
            rarities = sorted(set(c.get("rarity", "") for c in cards if c.get("rarity")))
            rarity_filter = st.multiselect("Rarity", rarities, default=rarities)
        with col4:
            search = st.text_input("Search", placeholder="Card name...")

    filtered = [
        c for c in cards
        if c.get("card_type", "Unit") in type_filter
        and c.get("domain", "") in domain_filter
        and c.get("rarity", "") in rarity_filter
        and (not search or search.lower() in c["name"].lower())
    ]

    section_header(f"{len(filtered)} / {len(cards)} cards")

    import pandas as pd
    df = pd.DataFrame(filtered)
    display_cols = ["name", "card_type", "domain", "cost", "rune_cost", "health", "rarity", "champion", "max_copies"]
    available_cols = [c for c in display_cols if c in df.columns]
    if not df.empty:
        st.dataframe(df[available_cols].sort_values("name"),
                     width='stretch', hide_index=True, height=500)

    if filtered:
        section_header("Card Detail")
        card_name = st.selectbox("Select a card", [c["name"] for c in filtered])
        card = next(c for c in filtered if c["name"] == card_name)

        col_a, col_b = st.columns(2)
        with col_a:
            ic = st.columns(3)
            ic[0].metric("Cost", card.get("cost", 0))
            ic[1].metric("Rune", card.get("rune_cost", 0))
            ic[2].metric("Might", card.get("health", 0))
            st.write(f"**{card.get('card_type', '?')}** | **{card.get('domain', '?')}** | **{card.get('rarity', '?')}**")
            if card.get("champion"): st.write("Champion unit")
            if card.get("tags"): st.write(f"Tags: {', '.join(card['tags'])}")
        with col_b:
            if card.get("ability"):
                st.markdown(f'<div class="card"><h3>Ability</h3><p>{card["ability"]}</p></div>', unsafe_allow_html=True)


# ===================================================================
# PAGE: Meta Decks
# ===================================================================

elif page == "Meta Decks":
    st.markdown('<div class="main-title">Tournament <span>Meta Decks</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Vegas & Bologna Regional Qualifier top decklists</div>', unsafe_allow_html=True)

    meta = load_meta_decks_data()
    if not meta:
        st.warning("No meta decks found.")
    else:
        events = Counter(d.get("event", "Unknown") for d in meta)
        evt_cols = st.columns(len(events))
        for i, (event, count) in enumerate(events.most_common()):
            with evt_cols[i]:
                short_name = event.replace("Regional Qualifier 2026", "RQ '26")
                stat_box(short_name, f"{count} decks")

        st.markdown("<br>", unsafe_allow_html=True)
        deck_names = [d.get("name", f"Deck {i}") for i, d in enumerate(meta)]
        selected = st.selectbox("Select a deck", deck_names)
        deck = meta[deck_names.index(selected)]

        col1, col2 = st.columns([2, 1])
        with col1:
            section_header(deck.get("name", ""))
            st.write(f"**{deck.get('event', '')}** | **{deck.get('placement', '')}**")
            import pandas as pd
            cards_df = pd.DataFrame(deck.get("cards", []))
            if not cards_df.empty:
                cards_df = cards_df.sort_values("count", ascending=False)
                st.write(f"**{cards_df['count'].sum()} cards**")
                st.dataframe(cards_df, width='stretch', hide_index=True)

        with col2:
            all_cards = load_cards()
            pool_lookup = {c["name"]: c for c in all_cards}
            types = Counter()
            deck_domains = Counter()
            for slot in deck.get("cards", []):
                card = pool_lookup.get(slot["name"])
                if card:
                    types[card.get("card_type", "?")] += slot["count"]
                    deck_domains[card.get("domain", "?")] += slot["count"]

            import plotly.express as px
            if types:
                fig = px.pie(names=list(types.keys()), values=list(types.values()),
                            title="Types", hole=0.45, color_discrete_sequence=PIE_COLORS)
                st.plotly_chart(chart_layout(fig, 250), width='stretch')
            if deck_domains:
                fig = px.pie(names=list(deck_domains.keys()), values=list(deck_domains.values()),
                            title="Domains", hole=0.45, color=list(deck_domains.keys()),
                            color_discrete_map=DOMAIN_COLORS)
                st.plotly_chart(chart_layout(fig, 250), width='stretch')

        section_header("Most Common Cards Across All Meta Decks")
        card_freq = Counter()
        for d in meta:
            for slot in d.get("cards", []):
                card_freq[slot["name"]] += slot["count"]
        import pandas as pd
        freq_df = pd.DataFrame(
            [(n, c, round(c / len(meta), 1)) for n, c in card_freq.most_common(30)],
            columns=["Card", "Total Copies", "Avg/Deck"]
        )
        st.dataframe(freq_df, width='stretch', hide_index=True)


# ===================================================================
# PAGE: Run Simulation
# ===================================================================

elif page == "Run Simulation":
    st.markdown('<div class="main-title">Run <span>Simulation</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Configure and launch training runs</div>', unsafe_allow_html=True)

    # Live status
    sim_status_panel()

    tab1, tab2, tab3 = st.tabs(["Genetic Algorithm", "RL Self-Play", "Full Pipeline"])

    with tab1:
        section_header("Quick GA Evolution")
        col1, col2 = st.columns(2)
        with col1:
            pop_size = st.slider("Population Size", 10, 100, 30,
                help="Number of decks competing each generation. Larger = more diversity but slower.")
            generations = st.slider("Generations", 5, 50, 10,
                help="How many breeding cycles to run. More generations = more refined decks.")
            mutation_rate = st.slider("Mutation Rate", 0.01, 0.30, 0.10,
                help="Chance each card gets randomly swapped per generation. Higher = more exploration, lower = more stability.")
        with col2:
            opp_pool = st.slider("Opponent Pool Size", 4, 20, 8,
                help="Number of unique opponents each deck is tested against per generation.")
            games_per_opp = st.slider("Games per Opponent", 3, 20, 5,
                help="Games played against each opponent. More games = more accurate win rates but slower.")
            coevo = st.slider("Hall of Fame %", 0, 100, 30,
                help="Percentage of opponents drawn from the Hall of Fame (best decks from past generations). Higher = tougher competition.") / 100

        if st.button("Run Genetic Algorithm", type="primary", use_container_width=True):
            with st.spinner("Evolving decks..."):
                from game.loader import load_card_pool
                from ai.genetic import run_genetic_algorithm, summarize_deck, genome_legend, genome_cards
                pool = load_card_pool()
                best, score = run_genetic_algorithm(
                    card_pool=pool, population_size=pop_size, deck_size=40,
                    generations=generations, keep_top=max(pop_size // 2, 5),
                    mutation_rate=mutation_rate, opponent_pool_size=opp_pool,
                    games_per_opponent=games_per_opp, coevo_ratio=coevo,
                )
                st.success(f"Best score: {score:.3f}")
                col_r1, col_r2 = st.columns([1, 2])
                with col_r1:
                    stat_box("Legend", genome_legend(best).split(" - ")[0])
                    st.markdown("<br>", unsafe_allow_html=True)
                    stat_box("Win Rate", f"{score:.0%}")
                with col_r2:
                    st.code(summarize_deck(best))

                import json
                os.makedirs("results", exist_ok=True)
                with open("results/best_deck.json", "w") as f:
                    json.dump({"legend": genome_legend(best), "score": score,
                              "deck": list(Counter(genome_cards(best)).items())}, f, indent=2)

    with tab2:
        section_header("Deep RL Self-Play Training (PPO)")
        col1, col2 = st.columns(2)
        with col1:
            rl_gens = st.slider("Generations", 10, 500, 100, key="rl_gens",
                help="Training cycles of self-play. The agent plays itself each generation and learns from wins/losses.")
            rl_games = st.slider("Games per Gen", 10, 100, 30, key="rl_games",
                help="Games played per generation. More games = more data to learn from each cycle, but slower.")
        with col2:
            rl_hidden = st.selectbox("Network Size", [128, 256, 512], index=1,
                help="Hidden layer size in the neural network. Larger = more capacity to learn complex strategies, but needs more training data.")
            rl_lr = st.select_slider("Learning Rate", [1e-4, 3e-4, 1e-3], value=3e-4,
                help="How fast the network updates its weights. Too high = unstable training, too low = slow learning.")

        if st.button("Train RL Agent", type="primary", use_container_width=True):
            from game.loader import load_card_pool
            from ai.self_play import SelfPlayTrainer, play_self_play_game, ppo_update, benchmark_vs_expert
            pool = load_card_pool()
            trainer = SelfPlayTrainer(pool, hidden=rl_hidden, lr=rl_lr)
            progress_bar = st.progress(0)
            status = st.empty()
            chart_placeholder = st.empty()
            chart_data = {"gen": [], "loss": []}
            import time
            for gen in range(1, rl_gens + 1):
                progress = gen / rl_gens
                temperature = 1.5 * (1 - progress) + 0.3 * progress
                trajectories = []
                for _ in range(rl_games):
                    t1, t2, _ = play_self_play_game(trainer.net, pool, 40, temperature)
                    trajectories.append(t1)
                    trajectories.append(t2)
                loss = ppo_update(trainer.net, trainer.optimizer, trajectories)
                chart_data["gen"].append(gen)
                chart_data["loss"].append(loss)
                progress_bar.progress(progress)
                if gen % 25 == 0 or gen == 1:
                    wr = benchmark_vs_expert(trainer.net, pool, games=100)
                    status.write(f"Gen {gen}/{rl_gens} | Loss: {loss:.4f} | vs Expert: {wr:.0%}")
                else:
                    status.write(f"Gen {gen}/{rl_gens} | Loss: {loss:.4f}")
                if gen % 10 == 0:
                    import plotly.graph_objects as go
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=chart_data["gen"], y=chart_data["loss"],
                                            name="Loss", line=dict(color=ACCENT)))
                    chart_layout(fig)
                    fig.update_layout(title="Training Loss", xaxis_title="Generation", yaxis_title="Loss")
                    chart_placeholder.plotly_chart(fig, width='stretch')
            trainer.save("models/rl_policy.pt")
            st.success("RL training complete! Model saved.")

    with tab3:
        section_header("Full Pipeline")
        st.markdown("""
        Runs the complete pipeline end-to-end:

        **Phase 1a** — ML Agent Training (REINFORCE)
        **Phase 1b** — Deep RL Training (PPO self-play)
        **Phase 2** — Massive GA Evolution (parallel, all opponent types)
        **Phase 3** — Meta Analytics Report
        """)
        st.code("python run_massive.py", language="bash")
        st.markdown(f'<div class="card"><h3>Tip</h3><p>Run from terminal for parallel workers. The status panel above updates live as the simulation runs.</p></div>', unsafe_allow_html=True)


# ===================================================================
# PAGE: Results
# ===================================================================

elif page == "Results":
    st.markdown('<div class="main-title">Simulation <span>Results</span></div>', unsafe_allow_html=True)

    # Sim status
    sim_status_panel()

    result = load_results()
    if not result:
        st.markdown('<div class="subtitle">No results yet. Run a simulation to see the best deck.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="subtitle">Best deck found by the genetic algorithm</div>', unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        with col1: stat_box("Legend", result.get("legend", "?").split(" - ")[0])
        with col2: stat_box("Win Rate", f"{result.get('score', 0):.0%}")
        with col3:
            deck_items = result.get("deck", [])
            stat_box("Cards", sum(c for _, c in deck_items) if deck_items else 0)

        st.markdown("<br>", unsafe_allow_html=True)

        if deck_items:
            import pandas as pd
            cards = load_cards()
            pool_lookup = {c["name"]: c for c in cards}
            rows = []
            for card_name, count in sorted(deck_items, key=lambda x: -x[1]):
                card = pool_lookup.get(card_name, {})
                rows.append({
                    "Card": card_name, "Copies": count,
                    "Type": card.get("card_type", "?"), "Domain": card.get("domain", "?"),
                    "Cost": card.get("cost", 0), "Rune": card.get("rune_cost", 0),
                    "Might": card.get("health", 0), "Rarity": card.get("rarity", "?"),
                })
            enriched = pd.DataFrame(rows)

            col_t, col_c = st.columns([3, 2])
            with col_t:
                section_header("Decklist")
                st.dataframe(enriched, width='stretch', hide_index=True, height=500)
            with col_c:
                import plotly.express as px
                domain_counts = Counter()
                for _, r in enriched.iterrows():
                    domain_counts[r["Domain"]] += r["Copies"]
                fig = px.pie(names=list(domain_counts.keys()), values=list(domain_counts.values()),
                            title="Domain Split", hole=0.45, color=list(domain_counts.keys()),
                            color_discrete_map=DOMAIN_COLORS)
                st.plotly_chart(chart_layout(fig, 280), width='stretch')

                type_counts = Counter()
                for _, r in enriched.iterrows():
                    type_counts[r["Type"]] += r["Copies"]
                fig = px.pie(names=list(type_counts.keys()), values=list(type_counts.values()),
                            title="Card Types", hole=0.45, color_discrete_sequence=PIE_COLORS)
                st.plotly_chart(chart_layout(fig, 280), width='stretch')

            section_header("Mana Curve")
            cost_dist = Counter()
            for _, r in enriched.iterrows():
                cost_dist[r["Cost"]] += r["Copies"]
            costs = sorted(cost_dist.keys())
            fig = px.bar(x=costs, y=[cost_dist[c] for c in costs],
                        title="Energy Cost Distribution",
                        labels={"x": "Energy Cost", "y": "Cards"},
                        color_discrete_sequence=[ACCENT])
            chart_layout(fig)
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, width='stretch')
