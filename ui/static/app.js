/* ==========================================================================
   STRATDECK : PLAYABLE CARD GAME & AI RESEARCH CONTROLLER (PHASE 8.2)
   ========================================================================== */

let CURRENT_SESSION_ID = null;
let CURRENT_GAME_STATE = null;
let SELECTED_PLAYER_COUNT = 2;
let SELECTED_DIFFICULTY = "auto";

// CARD METADATA DEFINITIONS
const CARD_INFO = {
    "HITMAN":  { icon: "🎯", type: "Elimination", desc: "Lethal Card: Eliminates player unless defused by SHIELD" },
    "ATTACK":  { icon: "⚔️", type: "Aggressive",  desc: "Ends turn without drawing; forces next player to take 2 turns" },
    "SHIELD":  { icon: "🛡️", type: "Defensive",   desc: "Auto-defuse: Protects against drawing a lethal HITMAN card" },
    "SKIP":    { icon: "⏭️", type: "Tactical",    desc: "Pass turn immediately to next player without drawing" },
    "REVERSE": { icon: "🔄", type: "Tactical",    desc: "Reverse turn direction (acts as Skip in 2-player match)" },
    "SPY":     { icon: "👁️", type: "Information", desc: "Peek top 3 deck cards without ending your turn" },
    "SWAP":    { icon: "🔀", type: "Disruptive",  desc: "Swap a random card from your hand with opponent" },
    "TRAP":    { icon: "🪤", type: "Disruptive",  desc: "Set forced double draw on opponent on next draw" },
    "STEAL":   { icon: "💰", type: "Disruptive",  desc: "Steal 1 random card directly from opponent's hand" },
    "DRAW":    { icon: "🎴", type: "Core",         desc: "Draw top card from deck to end your turn (Risk HITMAN!)" }
};

// ==========================================================================
// TUTORIAL STEPS & STEP-BY-STEP EXPLANATIONS (PRACTICE / TUTORIAL MODE)
// ==========================================================================
let CURRENT_TUTORIAL_STEP = 0;
const TUTORIAL_STEPS = [
    {
        badge: "Step 1 of 4: Survival Objective",
        title: "🎯 Objective: Outsurvive the Lethal HITMAN",
        text: "StratDeck is a survival card game modeled after Russian Roulette and Exploding Kittens. In the deck is 1 lethal <strong>HITMAN</strong> card. If you draw it without holding a <strong>SHIELD</strong>, you lose! You start with 1 SHIELD and 4 random cards. Every turn ends with a draw unless you play an action card.",
        hint: "💡 <strong>Coach Recommendation:</strong> You have the first turn! Try playing a Utility card like <strong>SPY</strong>, or click <strong>DRAW CARD</strong> to end your turn."
    },
    {
        badge: "Step 2 of 4: Multi-Action Combos",
        title: "⚡ Utility Cards Do NOT End Your Turn",
        text: "Cards like <strong>SPY</strong>, <strong>SWAP</strong>, and <strong>STEAL</strong> are Utility cards. When you play them, your turn continues! You can chain multiple utility moves (e.g. peek with SPY, then steal with STEAL) before finally drawing or attacking.",
        hint: "💡 <strong>Coach Recommendation:</strong> Both you and the AI have this multi-move superpower! Use it to gather intel and disrupt the opponent before drawing."
    },
    {
        badge: "Step 3 of 4: Offense & Turn Flow",
        title: "⚔️ How ATTACK, SKIP & TRAP Work",
        text: "Drawing from the deck is a risk. Playing <strong>SKIP</strong> or <strong>REVERSE</strong> ends your turn immediately without drawing. Playing <strong>ATTACK</strong> ends your turn and forces your opponent to take <strong>2 turns / draws</strong> from the lethal deck! <strong>TRAP</strong> forces them to draw 2 cards on their next draw.",
        hint: "💡 <strong>Coach Recommendation:</strong> When the deck count drops, playing ATTACK forces the opponent into double danger of hitting the HITMAN!"
    },
    {
        badge: "Step 4 of 4: Endgame Deck Risk",
        title: "💣 Managing Hitman Risk Percentage",
        text: "Check the <strong>Hitman Risk %</strong> badge in the top bar. Early in the game (Deck 22), risk is low (~4.5%). As cards are drawn and the deck shrinks to 5 cards, danger jumps above 20%! Hold your <strong>SHIELD</strong> as life insurance for the endgame.",
        hint: "💡 <strong>Coach Recommendation:</strong> Hover your mouse over any card below right now to read its detailed rule breakdown!"
    }
];

const TUTORIAL_CARD_EXPLANATIONS = {
    "HITMAN": {
        title: "🎯 HITMAN — Lethal Elimination Threat",
        text: "This card is hidden inside the draw deck! If drawn without a SHIELD in hand, the player is immediately eliminated and loses the game.",
        hint: "⚠️ <strong>Threat Warning:</strong> Cannot be held in hand; only drawn from deck. Watch the Hitman Risk % in the top header!"
    },
    "SHIELD": {
        title: "🛡️ SHIELD — Passive Defusal / Life Insurance",
        text: "Protects you from instant defeat. If you draw the HITMAN from the deck, 1 SHIELD is automatically consumed to defuse the HITMAN and shuffle it safely back into the deck.",
        hint: "💡 <strong>Coach Strategy:</strong> Never waste or trade your SHIELD early! It is your only shield against instant death."
    },
    "ATTACK": {
        title: "⚔️ ATTACK — Aggressive Double-Turn Transfer",
        text: "Immediately ends your turn <em>without drawing</em> and forces the opponent to take <strong>2 consecutive turns</strong> (forcing them to draw twice from the lethal deck). If they survive, they gain 2 cards, but faced double the risk of dying!",
        hint: "💡 <strong>Coach Strategy:</strong> Play ATTACK when the deck is small to force your opponent into lethal Russian-Roulette draws!"
    },
    "SKIP": {
        title: "⏭️ SKIP — Tactical Draw Avoidance",
        text: "Immediately concludes your turn without drawing a card from the deck. Safely passes play to the opponent.",
        hint: "💡 <strong>Coach Strategy:</strong> Use SKIP when you suspect the top card is a HITMAN or when deck danger is high."
    },
    "REVERSE": {
        title: "🔄 REVERSE — Direction & Turn Skip",
        text: "Reverses play direction. In a 2-player match, this acts as a tactical Skip, ending your turn without drawing from the deck.",
        hint: "💡 <strong>Coach Strategy:</strong> Acts as an extra SKIP card in 1v1 matches."
    },
    "SPY": {
        title: "👁️ SPY — Information Utility (Turn Continues!)",
        text: "Peeks at the top 3 cards of the draw deck in exact order. <strong>Crucial:</strong> SPY is a Utility card and does NOT end your turn! You can play another card right after.",
        hint: "💡 <strong>Coach Strategy:</strong> Always play SPY first to see if a HITMAN is on top before you decide whether to DRAW or ATTACK!"
    },
    "SWAP": {
        title: "🔀 SWAP — Disruptive Card Exchange (Turn Continues!)",
        text: "Exchanges a random card from your hand with a random card from the opponent. <strong>Crucial:</strong> SWAP is a Utility card and does NOT end your turn!",
        hint: "💡 <strong>Coach Strategy:</strong> Use SWAP to disrupt opponent strategy or fish for a SHIELD from their hand."
    },
    "STEAL": {
        title: "💰 STEAL — Direct Hand Theft (Turn Continues!)",
        text: "Steals 1 random card directly out of the opponent's hand and adds it to yours! <strong>Crucial:</strong> STEAL is a Utility card and does NOT end your turn.",
        hint: "💡 <strong>Coach Strategy:</strong> Reduces opponent options while increasing yours. Great when opponent has few cards."
    },
    "TRAP": {
        title: "🪤 TRAP — Forced Double-Draw Hazard",
        text: "Arms a hazard on the draw deck. The next player who draws must draw <strong>2 cards at once</strong> instead of 1, doubling their risk of drawing the HITMAN!",
        hint: "💡 <strong>Coach Strategy:</strong> Pairs destructively with SPY. If you know the 2nd card is a HITMAN, arm a TRAP!"
    },
    "DRAW": {
        title: "🎴 DRAW CARD — Core Turn Concluding Action",
        text: "Draws 1 card from the deck into your hand and ends your turn. If the card is not a HITMAN, it is added to your hand.",
        hint: "💡 <strong>Coach Strategy:</strong> Early in the game (20+ cards in deck), drawing is safe and expands your tactical options."
    }
};

function renderTutorialStep() {
    const step = TUTORIAL_STEPS[CURRENT_TUTORIAL_STEP];
    if (!step) return;
    const ind = document.getElementById("tutorialStepIndicator");
    const title = document.getElementById("tutorialCoachTitle");
    const text = document.getElementById("tutorialCoachText");
    const hint = document.getElementById("tutorialActionHint");
    if (ind) ind.innerText = step.badge;
    if (title) title.innerHTML = step.title;
    if (text) text.innerHTML = step.text;
    if (hint) hint.innerHTML = step.hint;
}

function nextTutorialStep() {
    CURRENT_TUTORIAL_STEP = (CURRENT_TUTORIAL_STEP + 1) % TUTORIAL_STEPS.length;
    renderTutorialStep();
}

function prevTutorialStep() {
    CURRENT_TUTORIAL_STEP = (CURRENT_TUTORIAL_STEP - 1 + TUTORIAL_STEPS.length) % TUTORIAL_STEPS.length;
    renderTutorialStep();
}

function showTutorialCardInfo(cardName) {
    if (!CURRENT_GAME_STATE || CURRENT_GAME_STATE.mode !== 'practice') return;
    const info = TUTORIAL_CARD_EXPLANATIONS[cardName];
    if (!info) return;
    const ind = document.getElementById("tutorialStepIndicator");
    const title = document.getElementById("tutorialCoachTitle");
    const text = document.getElementById("tutorialCoachText");
    const hint = document.getElementById("tutorialActionHint");
    if (ind) ind.innerText = "🔍 CARD BREAKDOWN";
    if (title) title.innerHTML = info.title;
    if (text) text.innerHTML = info.text;
    if (hint) hint.innerHTML = info.hint;
}

function resetTutorialCardInfo() {
    if (!CURRENT_GAME_STATE || CURRENT_GAME_STATE.mode !== 'practice') return;
    renderTutorialStep();
}

function updateTutorialCoachForGameState(state) {
    if (state.mode !== 'practice') return;
    const title = document.getElementById("tutorialCoachTitle");
    const text = document.getElementById("tutorialCoachText");
    const hint = document.getElementById("tutorialActionHint");
    const ind = document.getElementById("tutorialStepIndicator");

    if (state.game_over) {
        if (ind) ind.innerText = "🎓 TUTORIAL COMPLETE";
        if (state.winner === 0) {
            if (title) title.innerHTML = "🏆 Victory! Tutorial Mission Accomplished";
            if (text) text.innerHTML = "Congratulations! You successfully outsurvived the opponent. You now understand HITMAN deck risk, utility card chaining, and defensive preservation. You are ready for 1v1 Playable Mode against the real RL Agent!";
            if (hint) hint.innerHTML = "💡 <strong>Next Step:</strong> Click '🚪 Exit' in the top bar and try <strong>1v1 Playable Mode</strong>!";
        } else {
            if (title) title.innerHTML = "💀 Eliminated by the HITMAN";
            if (text) text.innerHTML = "You or the bot drew the lethal HITMAN without holding a SHIELD. Remember: as the deck shrinks, save your SHIELD, and force the opponent to take risky draws by using ATTACK or TRAP!";
            if (hint) hint.innerHTML = "💡 <strong>Try Again:</strong> Click 'Exit' to restart tutorial practice, or dive into 1v1 Mode!";
        }
        return;
    }

    const seqList = state.recent_ai_actions || [];
    if (seqList.length > 0) {
        if (ind) ind.innerText = "🤖 COACH BOT MOVE ANALYSIS";
        if (seqList.length > 1) {
            const moveNames = seqList.map(s => s.action).join(" ➔ ");
            if (title) title.innerHTML = `🎓 Bot Multi-Action Combo: ${moveNames}`;
            if (text) text.innerHTML = `The Coach Bot played <strong>${seqList.length} actions in one turn</strong> (${moveNames}). Because utility cards (SPY, SWAP, STEAL) do not end turns, it took multiple actions before ending its turn with ${seqList[seqList.length - 1].action}.`;
        } else {
            const singleAct = seqList[0].action;
            if (title) title.innerHTML = `🎓 Bot Action: Played ${singleAct}`;
            if (singleAct === "DRAW") {
                if (text) text.innerHTML = "The Coach Bot drew 1 card from the deck. Since it didn't hit the HITMAN, its turn ended safely and the card was added to its hand. Now it's your turn!";
            } else if (singleAct === "ATTACK") {
                if (text) text.innerHTML = "⚠️ <strong>Incoming Attack!</strong> The Coach Bot played ATTACK. It skipped its draw and forced 2 turns onto you! You must take consecutive draws or play an avoidance card (SKIP or ATTACK) to escape!";
            } else {
                if (text) text.innerHTML = `The Coach Bot played ${singleAct} and concluded its turn. Check your hand and choose your response!`;
            }
        }
        if (hint) hint.innerHTML = "💡 <strong>Your Turn:</strong> Hover over any card below to see its exact effect, or click <strong>DRAW CARD</strong> to conclude your turn.";
    } else {
        renderTutorialStep();
    }
}

// INITIALIZATION
document.addEventListener("DOMContentLoaded", () => {
    checkHealthStatus();
    loadResearchMetrics();

    // Direct screenshot/URL navigation support
    const urlParams = new URLSearchParams(window.location.search);
    const viewParam = urlParams.get("view");
    if (viewParam === "rules") {
        openModal("howItWorksModal");
    } else if (viewParam === "game") {
        startNewGame("1v1", 2);
    } else if (viewParam === "insights") {
        startNewGame("1v1", 2).then(() => {
            const drw = document.getElementById("aiInsightsDrawer");
            if (drw) {
                drw.style.transition = "none";
                drw.classList.add("open");
                drw.style.right = "0px";
            }
        });
    } else if (viewParam === "actionlog") {
        startNewGame("1v1", 2).then(() => {
            openModal("actionLogModal");
        });
    }
});

let IS_STARTING_GAME = false;
let WATCH_INTERVAL = null;
let WATCH_SPEED_MS = 1200;
let IS_WATCH_STEPPING = false;

// SCREEN SWITCHING CONTROLLER
function showScreen(screenId) {
    stopWatchAutoPlay();
    document.querySelectorAll(".screen-view").forEach(s => s.classList.remove("active"));
    
    if (screenId === 'lobby') {
        document.getElementById("lobbyScreen").classList.add("active");
    } else if (screenId === 'game') {
        document.getElementById("gameScreen").classList.add("active");
        if (!CURRENT_SESSION_ID && !IS_STARTING_GAME) {
            startNewGame('1v1', SELECTED_PLAYER_COUNT || 2);
        }
    } else if (screenId === 'research') {
        document.getElementById("researchScreen").classList.add("active");
    }
}

// SPECTATOR CONTROLLERS (RL AGENT VS RL AGENT WATCH MODE)
function setWatchSpeed(ms, btnEl) {
    WATCH_SPEED_MS = ms;
    document.querySelectorAll(".watch-speed-selector .spd-btn").forEach(b => b.classList.remove("active"));
    if (btnEl) btnEl.classList.add("active");
    if (WATCH_INTERVAL) {
        clearInterval(WATCH_INTERVAL);
        WATCH_INTERVAL = setInterval(stepWatchMatch, WATCH_SPEED_MS);
    }
}

async function stepWatchMatch() {
    if (!CURRENT_SESSION_ID || IS_WATCH_STEPPING) return;
    if (CURRENT_GAME_STATE && CURRENT_GAME_STATE.game_over) {
        stopWatchAutoPlay();
        return;
    }
    IS_WATCH_STEPPING = true;
    try {
        const res = await fetch("/api/game/watch/step", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ session_id: CURRENT_SESSION_ID })
        });
        const state = await res.json();
        if (state.error) {
            console.error("Watch step error:", state.error);
            stopWatchAutoPlay();
            return;
        }
        renderGameState(state);
        if (state.game_over) {
            stopWatchAutoPlay();
            showGameOverModal(state);
        }
    } catch (err) {
        console.error("Watch step failed:", err);
        stopWatchAutoPlay();
    } finally {
        IS_WATCH_STEPPING = false;
    }
}

function toggleWatchAutoPlay() {
    if (WATCH_INTERVAL) {
        stopWatchAutoPlay();
    } else {
        startWatchAutoPlay();
    }
}

function startWatchAutoPlay() {
    if (CURRENT_GAME_STATE && CURRENT_GAME_STATE.game_over) return;
    const btn = document.getElementById("watchAutoBtn");
    const icon = document.getElementById("watchAutoIcon");
    const text = document.getElementById("watchAutoText");
    if (btn) btn.classList.add("playing");
    if (icon) icon.innerText = "⏸";
    if (text) text.innerText = "Pause";
    
    // Execute immediate step then continue on interval
    stepWatchMatch();
    WATCH_INTERVAL = setInterval(stepWatchMatch, WATCH_SPEED_MS);
}

function stopWatchAutoPlay() {
    if (WATCH_INTERVAL) {
        clearInterval(WATCH_INTERVAL);
        WATCH_INTERVAL = null;
    }
    const btn = document.getElementById("watchAutoBtn");
    const icon = document.getElementById("watchAutoIcon");
    const text = document.getElementById("watchAutoText");
    if (btn) btn.classList.remove("playing");
    if (icon) icon.innerText = "▶";
    if (text) text.innerText = "Auto Play";
}

function restartWatchMatch() {
    stopWatchAutoPlay();
    startNewGame('watch', 2);
}

function selectPlayerCount(count, btnEl) {
    SELECTED_PLAYER_COUNT = count;
    document.querySelectorAll(".lobby-settings-panel .pill-opt").forEach(b => {
        if (b.parentElement && b.parentElement.previousElementSibling?.innerText?.includes("Player")) {
            b.classList.remove("active");
        }
    });
    if (btnEl) btnEl.classList.add("active");
}

function selectDifficulty(diff, btnEl) {
    SELECTED_DIFFICULTY = diff;
    const container = document.getElementById("lobbyDifficultySelector");
    if (container) {
        container.querySelectorAll(".pill-opt").forEach(b => b.classList.remove("active"));
    }
    if (btnEl) btnEl.classList.add("active");
}

async function changeInGameDifficulty(diff, btnEl) {
    if (!CURRENT_SESSION_ID) return;
    try {
        const res = await fetch("/api/game/difficulty", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                session_id: CURRENT_SESSION_ID,
                difficulty: diff
            })
        });
        const state = await res.json();
        if (state.error) {
            alert(state.error);
            return;
        }
        SELECTED_DIFFICULTY = diff;
        renderGameState(state);

        const inGameCont = document.getElementById("ingameDifficultySelector");
        if (inGameCont) {
            inGameCont.querySelectorAll(".pill-opt").forEach(b => b.classList.remove("active"));
        }
        if (btnEl) btnEl.classList.add("active");

        const dda = state.dda_status;
        const diffTitle = dda ? (dda.is_fixed ? `Tier ${dda.current_difficulty} (${dda.checkpoint || dda.checkpoint_name})` : "Auto (Adaptive DDA)") : diff;
        showGameToast("DIFFICULTY UPDATED", `Agent switched to: ${diffTitle}`, "⚙️", "normal");
    } catch (err) {
        console.error("Failed to change difficulty:", err);
    }
}

// HEALTH DIAGNOSTIC CHECK
async function checkHealthStatus() {
    const badge = document.getElementById("ollamaStatusBadge");
    try {
        const res = await fetch("/api/health");
        const data = await res.json();
        
        if (data.ollama_online && data.target_model_installed) {
            badge.className = "status-pill online";
            badge.innerText = "● Ollama qwen3.5:4b Ready";
        } else if (data.ollama_online) {
            badge.className = "status-pill offline";
            badge.innerText = "● Model qwen3.5:4b Missing";
        } else {
            badge.className = "status-pill offline";
            badge.innerText = "● Ollama Offline";
        }
    } catch (err) {
        badge.className = "status-pill offline";
        badge.innerText = "● Ollama Offline";
    }
}

// START NEW PLAYABLE GAME SESSION
async function startNewGame(mode = '1v1', playerCount = 2) {
    if (IS_STARTING_GAME) return;
    stopWatchAutoPlay();
    IS_STARTING_GAME = true;

    // Switch to game screen directly without triggering recursive call
    document.querySelectorAll(".screen-view").forEach(s => s.classList.remove("active"));
    document.getElementById("gameScreen").classList.add("active");

    const isWatchMode = (mode === 'watch');
    const watchBarEl = document.getElementById("watchControlsBar");
    if (watchBarEl) {
        watchBarEl.classList.toggle("hidden", !isWatchMode);
        watchBarEl.style.display = isWatchMode ? "flex" : "none";
    }

    const isPracticeMode = (mode === 'practice');
    const coachBarEl = document.getElementById("tutorialCoachBar");
    if (coachBarEl) {
        coachBarEl.classList.toggle("hidden", !isPracticeMode);
        coachBarEl.style.display = isPracticeMode ? "flex" : "none";
        if (isPracticeMode) {
            CURRENT_TUTORIAL_STEP = 0;
            renderTutorialStep();
        }
    }

    try {
        const res = await fetch("/api/game/start", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                mode: mode,
                player_count: playerCount,
                difficulty: SELECTED_DIFFICULTY,
                seed: Math.floor(Math.random() * 10000)
            })
        });
        const state = await res.json();
        CURRENT_SESSION_ID = state.session_id;
        CURRENT_GAME_STATE = state;
        renderGameState(state);
    } catch (err) {
        console.error("Failed to start game:", err);
        alert("Failed to initialize game session on backend.");
    } finally {
        IS_STARTING_GAME = false;
    }
}

// RENDER GAME STATE
function renderGameState(state) {
    CURRENT_GAME_STATE = state;

    // 1. HUD BAR
    document.getElementById("hudTurnBadge").innerText = `Turn ${state.turn_index}`;
    document.getElementById("hudDeckBadge").innerText = `Deck ${state.deck_size}`;
    document.getElementById("hudAliveBadge").innerText = `Alive ${state.alive_players_count}/2`;
    
    // HITMAN DANGER BADGE
    const dangerEl = document.getElementById("hudDangerBadge");
    if (dangerEl) {
        const danger = state.hitman_danger_pct || 0;
        dangerEl.innerText = `Hitman Risk: ${danger}%`;
        dangerEl.className = "hud-chip " + (danger > 25 ? "danger-high" : (danger > 10 ? "danger-med" : "danger-low"));
    }

    // DIFFICULTY BADGE IN HUD
    const diffBadge = document.getElementById("hudDifficultyBadge");
    if (diffBadge) {
        if (state.mode === 'practice') {
            diffBadge.innerText = "🎓 Mode: Step-by-Step Tutorial";
            diffBadge.className = "hud-chip diff-badge diff-tutorial";
            diffBadge.title = "Step-by-step tutorial practice against Rule-Based Coach Bot";
        } else if (state.dda_status) {
            const dda = state.dda_status;
            const ckpt = dda.checkpoint || dda.checkpoint_name || "DQN";
            if (dda.is_fixed) {
                diffBadge.innerText = `🤖 Tier ${dda.current_difficulty} (${ckpt})`;
                diffBadge.className = "hud-chip diff-badge diff-fixed";
                diffBadge.title = `Fixed Difficulty: Tier ${dda.current_difficulty} (${ckpt}). Click to change live.`;
            } else {
                diffBadge.innerText = `🤖 Auto (Tier ${dda.current_difficulty})`;
                diffBadge.className = "hud-chip diff-badge diff-auto";
                diffBadge.title = `Adaptive DDA Active (Tier ${dda.current_difficulty} - ${ckpt}). Target WR: 45–55%. Click to change.`;
            }
        }
    }

    const pendBadge = document.getElementById("hudPendingDrawBadge");
    pendBadge.innerText = `Pending Draw: ${state.pending_draw_count}`;
    if (state.pending_draw_count > 1) {
        pendBadge.classList.add("warning");
    } else {
        pendBadge.classList.remove("warning");
    }

    const isWatch = (state.mode === 'watch');
    const isPractice = (state.mode === 'practice');
    const isHumanTurn = (state.current_player === 0) && !state.game_over && !isWatch;

    // TUTORIAL COACH BAR (PRACTICE / TUTORIAL MODE ONLY)
    const coachBar = document.getElementById("tutorialCoachBar");
    if (coachBar) {
        coachBar.classList.toggle("hidden", !isPractice);
        coachBar.style.display = isPractice ? "flex" : "none";
        if (isPractice) {
            updateTutorialCoachForGameState(state);
        }
    }

    // SPECTATOR CONTROLS BAR & MATCHUP LABEL (WATCH MODE ONLY)
    const watchBar = document.getElementById("watchControlsBar");
    if (watchBar) {
        watchBar.classList.toggle("hidden", !isWatch);
        watchBar.style.display = isWatch ? "flex" : "none";
    }

    const watchMatchup = document.getElementById("watchMatchupLabel");
    if (watchMatchup) {
        watchMatchup.innerText = `RL Agent 0 (MaskablePPO)  VS  ${state.p1_name || 'RL Agent 1'}`;
    }

    const drawBtn = document.getElementById("drawCardBtn");
    const specBadge = document.getElementById("watchSpectatorBadge");
    if (drawBtn) {
        drawBtn.style.display = isWatch ? "none" : "inline-flex";
        drawBtn.onmouseenter = () => showTutorialCardInfo("DRAW");
        drawBtn.onmouseleave = () => resetTutorialCardInfo();
    }
    const deckPile = document.getElementById("deckPile");
    if (deckPile) {
        deckPile.onmouseenter = () => showTutorialCardInfo("DRAW");
        deckPile.onmouseleave = () => resetTutorialCardInfo();
    }
    if (specBadge) {
        specBadge.classList.toggle("hidden", !isWatch);
        if (isWatch) {
            const p0Count = (state.human_hand || []).length;
            specBadge.innerHTML = `<span>👁️ AGENT 0 HAND (${p0Count} CARDS)</span>`;
        }
    }

    // DYNAMIC PLAYER HEADERS
    const topAvatar = document.getElementById("topAvatarIcon");
    const topName = document.getElementById("topPlayerName");
    const topRole = document.getElementById("topPlayerRole");
    const oppStatus = document.getElementById("oppStatus");
    const bottomAvatar = document.getElementById("bottomAvatarIcon");
    const bottomName = document.getElementById("bottomPlayerName");
    const bottomRole = document.getElementById("bottomPlayerRole");

    if (isWatch) {
        if (topAvatar) topAvatar.innerText = "🤖";
        if (topName) topName.innerHTML = `RL AGENT 1 <span id="topPlayerRole" class="player-role">(${state.p1_name || 'MaskedDQN'})</span>`;
        if (oppStatus) oppStatus.innerText = `● Alive | Policy: ${state.p1_name || 'MaskedDQN'}`;

        if (bottomAvatar) bottomAvatar.innerText = "🤖";
        if (bottomName) bottomName.innerHTML = `RL AGENT 0 <span id="bottomPlayerRole" class="player-role">(MaskablePPO)</span>`;
    } else if (isPractice) {
        if (topAvatar) topAvatar.innerText = "🎓";
        if (topName) topName.innerHTML = `TUTORIAL COACH BOT <span id="topPlayerRole" class="player-role">(Rule-Based Coach)</span>`;
        if (oppStatus) oppStatus.innerText = "● Active | Guided Tutorial Mode";

        if (bottomAvatar) bottomAvatar.innerText = "👤";
        if (bottomName) bottomName.innerHTML = `YOUR HAND <span id="bottomPlayerRole" class="player-role">(Student)</span>`;
    } else {
        if (topAvatar) topAvatar.innerText = "🤖";
        if (topName) topName.innerHTML = `RL AGENT <span id="topPlayerRole" class="player-role">(Player 1)</span>`;
        if (oppStatus) oppStatus.innerText = "● Alive | Frozen MaskablePPO";

        if (bottomAvatar) bottomAvatar.innerText = "👤";
        if (bottomName) bottomName.innerHTML = `YOUR HAND <span id="bottomPlayerRole" class="player-role">(Player 0)</span>`;
    }

    // 2. TURN BANNER & OPPONENT PLAYED BANNER
    const banner = document.getElementById("turnBanner");
    const bannerText = document.getElementById("turnBannerText");
    const oppBannerTag = document.getElementById("oppBannerTag");
    const oppBannerText = document.getElementById("oppPlayedText");

    const latestAI = state.latest_ai_decision;
    const seqList = state.recent_ai_actions || [];

    if (isWatch) {
        if (state.game_over) {
            const winnerName = (state.winner === 0) ? (state.p0_name || "RL Agent 0 (MaskablePPO)") : (state.p1_name || "RL Agent 1 (MaskedDQN)");
            banner.className = "turn-banner your-turn";
            bannerText.innerText = `MATCH ENDED — ${winnerName.toUpperCase()} WON! 🏆`;
        } else {
            const currActor = (state.current_player === 0) ? (state.p0_name || "RL Agent 0 (MaskablePPO)") : (state.p1_name || "RL Agent 1 (MaskedDQN)");
            banner.className = "turn-banner " + (state.current_player === 0 ? "your-turn" : "ai-turn");
            bannerText.innerText = `🤖 ${currActor.toUpperCase()}'S TURN (Turn ${state.turn_index})`;
        }

        if (seqList.length > 0) {
            const lastAct = seqList[0];
            const inf = CARD_INFO[lastAct.action] || { icon: "🃏" };
            const isP0 = (lastAct.actor_id === 0);
            if (oppBannerTag) {
                oppBannerTag.className = "opp-banner-tag " + (isP0 ? "actor-tag-p0" : "actor-tag-p1");
                oppBannerTag.innerText = isP0 ? "🤖 AGENT 0 MOVED" : "🤖 AGENT 1 MOVED";
            }
            oppBannerText.innerHTML = `<strong>${lastAct.actor_label}</strong> played <strong class="played-card-highlight card-tag-${lastAct.action}">${inf.icon} ${lastAct.action}</strong> (Turn ${lastAct.turn_index})`;
        } else {
            if (oppBannerTag) {
                oppBannerTag.className = "opp-banner-tag start-tag";
                oppBannerTag.innerText = "👁️ SPECTATOR READY";
            }
            oppBannerText.innerText = "Match initialized! Press 'Auto Play' or 'Next Step' to watch the RL agents battle.";
        }
    } else {
        if (seqList.length > 1) {
            if (oppBannerTag) {
                oppBannerTag.className = "opp-banner-tag ai-tag";
                oppBannerTag.innerText = `🤖 MULTI-MOVE (${seqList.length} CARDS)`;
            }
            const actionBadges = seqList.map(s => {
                const inf = CARD_INFO[s.action] || { icon: "🃏" };
                return `<strong class="played-card-highlight card-tag-${s.action}">${inf.icon} ${s.action}</strong>`;
            }).join(" ➔ ");
            oppBannerText.innerHTML = `RL Agent executed multi-action sequence: ${actionBadges}`;
        } else if (latestAI) {
            const aiName = latestAI.action;
            const info = CARD_INFO[aiName] || { icon: "🤖", type: "AI Move", desc: aiName };
            if (oppBannerTag) {
                oppBannerTag.className = "opp-banner-tag ai-tag";
                oppBannerTag.innerText = "🤖 AGENT MOVED";
            }
            oppBannerText.innerHTML = `RL Agent played <strong class="played-card-highlight card-tag-${aiName}">${info.icon} ${aiName}</strong>`;
        } else {
            if (oppBannerTag) {
                oppBannerTag.className = "opp-banner-tag start-tag";
                oppBannerTag.innerText = "🎮 GAME START";
            }
            oppBannerText.innerText = "Match started. You have the first turn! Choose a card from your hand.";
        }

        if (state.game_over) {
            banner.className = "turn-banner your-turn";
            bannerText.innerText = state.winner === 0 ? "GAME ENDED — YOU WIN! 🏆" : "GAME ENDED — RL AGENT WON 💀";
        } else if (isHumanTurn) {
            banner.className = "turn-banner your-turn";
            const hasUtil = (state.human_hand || []).some(c => c === "SPY" || c === "SWAP" || c === "STEAL");
            const utilHint = hasUtil ? '<span style="color: #92400e; font-weight: 700; margin-left: 6px;">[⚡ You have FREE Utility moves!]</span>' : '';
            if (seqList.length > 1) {
                bannerText.innerHTML = `YOUR TURN <span class="ai-last-tag">(RL Agent completed ${seqList.length} moves)</span> ${utilHint}`;
            } else if (latestAI) {
                const aiName = latestAI.action;
                const info = CARD_INFO[aiName] || { icon: "🤖" };
                bannerText.innerHTML = `YOUR TURN <span class="ai-last-tag">(RL Agent played ${info.icon} ${aiName})</span> ${utilHint}`;
            } else {
                bannerText.innerHTML = `YOUR TURN ${utilHint || '— Play a card or click DRAW CARD to end turn'}`;
            }
        } else {
            banner.className = "turn-banner ai-turn";
            bannerText.innerText = "RL AGENT'S TURN — Thinking...";
        }
    }

    // 3. MULTI-ACTION ROUND SEQUENCE TRAY
    renderAIActionSequenceTray(seqList);

    // 4. AI HERO PLAYED CARD SLOT (CENTER BOARD WITH CARD DROP ANIMATION)
    const heroCard = document.getElementById("aiPlayedHeroCard");
    const heroIcon = document.getElementById("heroCardIcon");
    const heroTitle = document.getElementById("heroCardTitle");
    const heroDesc = document.getElementById("heroCardDesc");
    const heroBadge = document.getElementById("heroCardBadge");

    if (latestAI) {
        const aiName = latestAI.action;
        const info = CARD_INFO[aiName] || { icon: "🤖", type: "AI Move", desc: aiName };
        const probStr = (latestAI.probability * 100).toFixed(0);

        // Re-trigger card drop animation on every agent step
        heroCard.classList.remove("card-dropping");
        void heroCard.offsetWidth; // Force CSS reflow
        heroCard.className = `ai-hero-card card-${aiName} active-hero card-dropping`;

        if (isWatch && seqList.length > 0) {
            const lastAct = seqList[0];
            const isP0 = (lastAct.actor_id === 0);
            heroBadge.innerText = `${(lastAct.actor_label || 'AGENT').toUpperCase()} (${probStr}%)`;
            heroBadge.style.background = isP0 ? "#0284c7" : "#7c3aed";
            heroBadge.style.color = "#ffffff";
        } else {
            heroBadge.innerText = `CONFIDENCE: ${probStr}%`;
            heroBadge.style.background = "";
            heroBadge.style.color = "";
        }
        heroIcon.innerText = info.icon;
        heroTitle.innerText = aiName;
        heroDesc.innerText = info.desc;
    } else {
        heroCard.className = "ai-hero-card empty-hero";
        heroBadge.innerText = isWatch ? "SPECTATOR" : "READY";
        heroBadge.style.background = "";
        heroBadge.style.color = "";
        heroIcon.innerText = "🎴";
        heroTitle.innerText = isWatch ? "READY TO PLAY" : "FIRST MOVE";
        heroDesc.innerText = isWatch ? "Press Auto Play or Next Step" : "Play your first card to begin";
    }

    // 5. OPPONENT HAND (DUAL-HAND AWARE: FACE-UP IN SPECTATOR WATCH MODE)
    const oppHandCountEl = document.getElementById("oppHandCount");
    if (oppHandCountEl) oppHandCountEl.innerText = state.opponent_hand_count || 0;

    const topOppArea = document.getElementById("topOpponentArea");
    const topWatchBadge = document.getElementById("topWatchSpectatorBadge");
    const topOppContainer = document.getElementById("topOppHandContainer");
    const oppSpectatorGrid = document.getElementById("oppSpectatorCardsGrid");
    const oppBacksGrid = document.getElementById("oppCardsGrid");

    if (isWatch && state.p1_hand) {
        if (topOppArea) topOppArea.classList.add("watch-mode-area");
        if (topWatchBadge) {
            topWatchBadge.classList.remove("hidden");
            topWatchBadge.innerHTML = `<span>👁️ AGENT 1 HAND (${state.p1_hand.length} CARDS)</span>`;
        }
        if (topOppContainer) topOppContainer.classList.add("hidden");
        if (oppSpectatorGrid) {
            oppSpectatorGrid.classList.remove("hidden");
            oppSpectatorGrid.innerHTML = "";
            state.p1_hand.forEach(cardName => {
                const info = CARD_INFO[cardName] || { icon: "🃏", type: "Action", desc: cardName };
                const cardEl = document.createElement("div");
                cardEl.className = `playing-card card-${cardName} spectator-card`;
                cardEl.innerHTML = `
                    <div class="card-top">
                        <span class="card-type-badge">${info.type}</span>
                        <span class="card-icon-mini">${info.icon}</span>
                    </div>
                    <div class="card-symbol">${info.icon}</div>
                    <div class="card-title">${cardName}</div>
                    <div class="card-desc">${info.desc}</div>
                    <div class="card-status-tag spectator-tag">👁️ In Hand</div>
                `;
                oppSpectatorGrid.appendChild(cardEl);
            });
        }
    } else {
        if (topOppArea) topOppArea.classList.remove("watch-mode-area");
        if (topWatchBadge) topWatchBadge.classList.add("hidden");
        if (oppSpectatorGrid) {
            oppSpectatorGrid.classList.add("hidden");
            oppSpectatorGrid.innerHTML = "";
        }
        if (topOppContainer) topOppContainer.classList.remove("hidden");
        if (oppBacksGrid) {
            oppBacksGrid.innerHTML = "";
            const cardCount = state.opponent_hand_count || 4;
            for (let i = 0; i < cardCount; i++) {
                const cb = document.createElement("div");
                cb.className = "opponent-card-back";
                cb.innerText = "🎴";
                oppBacksGrid.appendChild(cb);
            }
        }
    }

    // 6. DECK & DISCARD
    document.getElementById("deckCountBadge").innerText = state.deck_size;
    
    const topDiscardEl = document.getElementById("topDiscardCard");
    if (state.public_discard && state.public_discard.length > 0) {
        const topName = state.public_discard[state.public_discard.length - 1];
        const info = CARD_INFO[topName] || { icon: "🃏", name: topName };
        topDiscardEl.className = `top-discard-card card-${topName}`;
        topDiscardEl.innerHTML = `
            <div class="discard-icon">${info.icon}</div>
            <div class="discard-name">${topName}</div>
        `;
    } else {
        topDiscardEl.className = "top-discard-card empty";
        topDiscardEl.innerHTML = `<div class="discard-icon">🗑️</div><div class="discard-name">EMPTY</div>`;
    }

    // 7. RECENT ACTION HUD
    if (state.action_log && state.action_log.length > 0) {
        const lastLog = state.action_log[state.action_log.length - 1];
        document.getElementById("recentMoveText").innerText = lastLog;
    }

    // 8. PROCESS GAME EVENTS (SPY PEEK, STEAL, SWAP, DEFUSED, TRAP)
    processGameEvents(state.last_events || []);

    // 9. HUMAN CARDS HAND GRID
    renderHumanHand(state);

    // 10. DRAW CARD BUTTON
    if (drawBtn) {
        const drawMasked = state.human_action_mask ? !state.human_action_mask[9] : false;
        drawBtn.disabled = !isHumanTurn || drawMasked || state.game_over;
    }

    // 11. SYNCHRONIZE IMMUTABLE LATEST AI DECISION
    renderAIDecisionEvidence(state.latest_ai_decision, state);

    // 12. CHECK GAME OVER RESULT MODAL
    if (state.game_over) {
        showGameOverModal(state);
    }
}

// RENDER MULTI-ACTION SEQUENCE TRAY
function renderAIActionSequenceTray(seqList) {
    const seqTray = document.getElementById("aiActionSequenceTray");
    const seqCards = document.getElementById("aiActionSequenceCards");
    if (!seqTray || !seqCards) return;

    if (seqList && seqList.length > 0) {
        seqTray.classList.remove("hidden");
        seqCards.innerHTML = "";
        seqList.forEach((step, idx) => {
            const sInfo = CARD_INFO[step.action] || { icon: "🃏", type: "Action" };
            const isLast = (idx === seqList.length - 1);
            let effectNote = "";
            if (step.action === "STEAL") effectNote = "Stole card from you";
            else if (step.action === "SWAP") effectNote = "Swapped cards";
            else if (step.action === "TRAP") effectNote = "Forces 2x Draw";
            else if (step.action === "ATTACK") effectNote = "Forces 2 Turns";
            else if (step.action === "SPY") effectNote = "Peeked deck";

            const stepCard = document.createElement("div");
            stepCard.className = `sequence-card-item card-${step.action} ${!step.is_utility ? 'is-core' : ''}`;
            stepCard.innerHTML = `
                <span class="sequence-step-badge ${!step.is_utility ? 'core-badge' : ''}">Step ${idx + 1}: ${step.is_utility ? 'UTIL' : 'CORE'}</span>
                <div class="sequence-card-icon">${sInfo.icon}</div>
                <div class="sequence-card-details">
                    <span class="sequence-card-title">${step.action}</span>
                    <span class="sequence-card-prob">${(step.probability * 100).toFixed(0)}% Conf</span>
                    ${effectNote ? `<span class="sequence-effect-badge" style="font-size: 0.68rem; color: #b91c1c; font-weight: 700; margin-top: 2px;">⚡ ${effectNote}</span>` : ''}
                </div>
            `;
            seqCards.appendChild(stepCard);

            if (!isLast) {
                const arrow = document.createElement("div");
                arrow.className = "sequence-arrow";
                arrow.innerText = "➔";
                seqCards.appendChild(arrow);
            }
        });
    } else {
        seqTray.classList.add("hidden");
    }
}

// PROCESS LAST GAME EVENTS
function processGameEvents(events) {
    if (!events || events.length === 0) return;
    const isWatch = (CURRENT_GAME_STATE && CURRENT_GAME_STATE.mode === 'watch');

    // Aggregate stolen cards so multiple thefts are clearly visible and not wiped
    const stolenEvents = events.filter(e => e.type === "AI_STOLE");
    const otherEvents = events.filter(e => e.type !== "AI_STOLE");

    if (stolenEvents.length > 0) {
        const stolenCards = stolenEvents.map(e => e.card).join(", ");
        showGameToast("CARDS STOLEN!", `⚠️ RL Agent played STEAL and took ${stolenEvents.length} card(s) from your hand: ${stolenCards}!`, "💰", "danger");
    }

    otherEvents.forEach((ev, idx) => {
        const delay = (stolenEvents.length > 0 ? (idx + 1) * 2200 : idx * 1500);
        setTimeout(() => {
            if (ev.type === "SPY_PEEK") {
                if (isWatch) {
                    showGameToast("SPY PEEK", (ev.actor || "RL Agent") + " peeked top deck cards.", "👁️", "normal");
                } else {
                    showSpyPeekModal(ev.cards || []);
                }
            } else if (ev.type === "STEAL_SUCCESS") {
                showGameToast("CARD STOLEN!", ev.message, "💰", "success");
            } else if (ev.type === "SWAP_SUCCESS") {
                showGameToast("CARD SWAP!", ev.message, "🔀", "success");
            } else if (ev.type === "AI_TRAP") {
                showGameToast("TRAP SET!", ev.message, "🪤", "warning");
            } else if (ev.type === "AI_ATTACK") {
                showGameToast("ATTACKED!", ev.message, "⚔️", "danger");
            } else if (ev.type === "DEFUSED") {
                showGameToast("HITMAN DEFUSED!", ev.message, "🛡️", "success");
            } else if (ev.type === "AI_STOLE_FAILED") {
                showGameToast("STEAL FAILED", ev.message, "🛡️", "normal");
            }
        }, delay);
    });
}

// RENDER HUMAN HAND CARDS
function renderHumanHand(state) {
    const grid = document.getElementById("humanHandGrid");
    grid.innerHTML = "";

    const isWatch = (state.mode === 'watch');
    const isHumanTurn = (state.current_player === 0) && !state.game_over;
    const hand = state.human_hand || [];
    const mask = state.human_action_mask || [];

    hand.forEach((cardName, idx) => {
        const cardActionIdx = getActionIndexForCard(cardName);
        const isLegal = mask[cardActionIdx] === 1 || mask[cardActionIdx] === true;
        const info = CARD_INFO[cardName] || { icon: "🃏", type: "Action", desc: cardName };

        const isUtility = (cardName === "SPY" || cardName === "SWAP" || cardName === "STEAL");
        let tagText, tagClass;
        if (isWatch) {
            tagText = '👁️ In Hand';
            tagClass = 'spectator-tag';
        } else if (!isLegal) {
            tagText = '✕ Masked';
            tagClass = 'masked-tag';
        } else if (isUtility) {
            tagText = '⚡ FREE MOVE (Turn Continues)';
            tagClass = 'utility-tag';
        } else {
            tagText = '⏳ ENDS TURN';
            tagClass = 'ends-turn-tag';
        }

        const cardEl = document.createElement("div");
        if (isWatch) {
            cardEl.className = `playing-card card-${cardName} spectator-card`;
        } else {
            cardEl.className = `playing-card card-${cardName} ${isLegal ? 'legal' : 'masked'} ${isUtility ? 'utility-card-border' : ''}`;
        }

        cardEl.innerHTML = `
            <div class="card-top">
                <span class="card-type-badge">${info.type}</span>
                <span class="card-icon-mini">${info.icon}</span>
            </div>
            <div class="card-symbol">${info.icon}</div>
            <div class="card-title">${cardName}</div>
            <div class="card-desc">${info.desc}</div>
            <div class="card-status-tag ${tagClass}">
                ${tagText}
            </div>
        `;

        if (!isWatch && isLegal && isHumanTurn) {
            cardEl.onclick = () => handleCardClick(cardName, cardActionIdx);
        }

        // TUTORIAL BREAKDOWN HOVER LISTENERS
        cardEl.addEventListener("mouseenter", () => showTutorialCardInfo(cardName));
        cardEl.addEventListener("mouseleave", () => resetTutorialCardInfo());

        grid.appendChild(cardEl);
    });
}

// MAP CARD NAME TO ACTION INDEX
function getActionIndexForCard(name) {
    const map = {
        "HITMAN": 0, "ATTACK": 1, "SHIELD": 2, "SKIP": 3, "REVERSE": 4,
        "SPY": 5, "SWAP": 6, "TRAP": 7, "STEAL": 8, "DRAW": 9
    };
    return map[name] !== undefined ? map[name] : 9;
}

let IS_PROCESSING_TURN = false;

// SUBMIT HUMAN ACTION TO BACKEND
async function playHumanAction(actionIdx) {
    if (!CURRENT_SESSION_ID || !CURRENT_GAME_STATE || IS_PROCESSING_TURN) return;
    if (CURRENT_GAME_STATE.mode === 'watch') {
        console.warn("Human moves are disabled in Watch Match mode.");
        return;
    }
    if (CURRENT_GAME_STATE.game_over) return;
    if (CURRENT_GAME_STATE.current_player !== 0) {
        console.warn("Not Player 0's turn!");
        return;
    }

    IS_PROCESSING_TURN = true;

    const drawBtn = document.getElementById("drawCardBtn");
    const handGrid = document.getElementById("humanHandGrid");
    const deckPile = document.getElementById("deckPile");
    if (drawBtn) drawBtn.disabled = true;
    if (handGrid) handGrid.classList.add("processing-locked");
    if (deckPile) deckPile.classList.add("processing-locked");

    try {
        const res = await fetch("/api/game/action", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ session_id: CURRENT_SESSION_ID, action_idx: actionIdx })
        });

        if (!res.ok) {
            const errData = await res.json();
            alert(`Turn Notice: ${errData.error || 'Action rejected.'}`);
            return;
        }

        const state = await res.json();
        renderGameState(state);
    } catch (err) {
        console.error("Action submission error:", err);
    } finally {
        IS_PROCESSING_TURN = false;
        if (handGrid) handGrid.classList.remove("processing-locked");
        if (deckPile) deckPile.classList.remove("processing-locked");
        const state = CURRENT_GAME_STATE;
        if (drawBtn && state) {
            const isHumanTurn = (state.current_player === 0) && !state.game_over;
            const drawMasked = state.human_action_mask ? !state.human_action_mask[9] : false;
            drawBtn.disabled = !isHumanTurn || drawMasked || state.game_over;
        }
    }
}

// HANDLE DRAW CARD CLICK
function handleDrawClick() {
    if (!CURRENT_GAME_STATE || CURRENT_GAME_STATE.mode === 'watch' || IS_PROCESSING_TURN) return;
    if (CURRENT_GAME_STATE.current_player !== 0 || CURRENT_GAME_STATE.game_over) return;
    playHumanAction(9); // 9 is DRAW index
}

// HANDLE CARD CLICK (WITH SHIELD PROTECTION WARNING)
let PENDING_SHIELD_ACTION_IDX = null;
function handleCardClick(cardName, actionIdx) {
    if (!CURRENT_GAME_STATE || CURRENT_GAME_STATE.mode === 'watch' || IS_PROCESSING_TURN) return;
    if (CURRENT_GAME_STATE.current_player !== 0 || CURRENT_GAME_STATE.game_over) return;

    // Verify card is actually still present in hand to prevent desync
    if (CURRENT_GAME_STATE.human_hand && !CURRENT_GAME_STATE.human_hand.includes(cardName)) {
        console.warn(`Card ${cardName} is no longer in hand.`);
        renderGameState(CURRENT_GAME_STATE);
        return;
    }

    if (cardName === "SHIELD") {
        PENDING_SHIELD_ACTION_IDX = actionIdx;
        openModal('shieldWarningModal');
    } else {
        playHumanAction(actionIdx);
    }
}

function confirmPlayShieldSkip() {
    closeModal('shieldWarningModal');
    if (IS_PROCESSING_TURN) return;
    if (PENDING_SHIELD_ACTION_IDX !== null) {
        const idx = PENDING_SHIELD_ACTION_IDX;
        PENDING_SHIELD_ACTION_IDX = null;
        playHumanAction(idx);
    }
}

// SPY DECK PEEK MODAL CONTROLLER
function showSpyPeekModal(cards) {
    const row = document.getElementById("spyCardsRow");
    if (!row) return;
    row.innerHTML = "";

    const hasHitman = cards.includes("HITMAN");
    const adviceBox = document.getElementById("spyTacticalAdvice");
    const adviceIcon = document.getElementById("spyAdviceIcon");
    const adviceText = document.getElementById("spyAdviceText");

    if (hasHitman) {
        const hitmanIndex = cards.indexOf("HITMAN");
        adviceBox.className = "spy-advice-box danger";
        adviceIcon.innerText = "🚨";
        adviceText.innerHTML = `<strong>CRITICAL DANGER!</strong> HITMAN detected at <strong>position #${hitmanIndex + 1}</strong>! If you draw without a SHIELD in hand, you will be eliminated immediately! Play ATTACK, SKIP, or REVERSE immediately to avoid drawing!`;
    } else {
        adviceBox.className = "spy-advice-box safe";
        adviceIcon.innerText = "✅";
        adviceText.innerHTML = `<strong>DECK SAFE:</strong> No HITMAN detected in the top 3 cards. You can safely draw or plan your utility moves.`;
    }

    cards.forEach((cName, idx) => {
        const cInfo = CARD_INFO[cName] || { icon: "🎴", type: "Core", desc: cName };
        const isHitman = (cName === "HITMAN");
        const orderText = idx === 0 ? "1st (Next Draw)" : (idx === 1 ? "2nd in Deck" : "3rd in Deck");

        const cardDiv = document.createElement("div");
        cardDiv.className = `peek-card card-${cName} ${isHitman ? 'danger-hitman' : ''}`;
        cardDiv.innerHTML = `
            <span class="peek-order-badge ${idx === 0 ? 'next-badge' : ''}">${orderText}</span>
            <div class="peek-icon">${cInfo.icon}</div>
            <div class="peek-title">${cName}</div>
            <div class="peek-desc">${cInfo.desc}</div>
            <span class="peek-tag">${cInfo.type}</span>
        `;
        row.appendChild(cardDiv);
    });

    openModal('spyPeekModal');
}

// FLOATING GAME EVENT TOAST CONTROLLER
let TOAST_TIMER = null;
function showGameToast(title, message, icon = "⚡", type = "normal") {
    const toast = document.getElementById("gameEventToast");
    if (!toast) return;
    clearTimeout(TOAST_TIMER);

    toast.className = `game-event-toast toast-${type}`;
    document.getElementById("toastIcon").innerText = icon;
    document.getElementById("toastTitle").innerText = title;
    document.getElementById("toastMessage").innerText = message;

    toast.classList.remove("hidden");
    TOAST_TIMER = setTimeout(() => {
        toast.classList.add("hidden");
    }, 5500);
}

function dismissToast() {
    const toast = document.getElementById("gameEventToast");
    if (toast) toast.classList.add("hidden");
}

// SYNCHRONIZE IMMUTABLE LATEST AI DECISION EVIDENCE
function renderAIDecisionEvidence(decision, state) {
    // 1. DDA & PROFILER STATUS (ALWAYS RENDER)
    if (state && state.cluster_id !== null && state.cluster_id !== undefined) {
        const clusterBadge = document.getElementById("clusterBadge");
        if (clusterBadge) clusterBadge.innerText = `Cluster ${state.cluster_id}`;
    }

    if (state && state.dda_status) {
        const dda = state.dda_status;
        const ckpt = dda.checkpoint || dda.checkpoint_name || "Checkpoint";
        const modeLabel = dda.is_fixed ? "Fixed (User Locked)" : "Auto (Adaptive DDA)";
        const ddaLvlEl = document.getElementById("ddaLevel");
        const ddaWrEl = document.getElementById("ddaWinRate");
        if (ddaLvlEl) ddaLvlEl.innerText = `L${dda.current_difficulty} (${ckpt}) — ${modeLabel}`;
        if (ddaWrEl) ddaWrEl.innerText = `${(dda.rolling_win_rate * 100).toFixed(1)}%`;

        // Synchronize live switcher buttons in drawer
        const inGameCont = document.getElementById("ingameDifficultySelector");
        if (inGameCont) {
            inGameCont.querySelectorAll(".pill-opt").forEach(b => {
                const isAutoBtn = b.innerText.includes("Auto");
                if (dda.is_fixed) {
                    const match = b.getAttribute("onclick")?.includes(String(dda.current_difficulty));
                    b.classList.toggle("active", Boolean(match));
                } else {
                    b.classList.toggle("active", isAutoBtn);
                }
            });
        }
    }

    // 2. LATEST AI DECISION
    if (!decision) {
        document.getElementById("aiActionName").innerText = "None Recorded";
        document.getElementById("aiActionProb").innerText = "0.0%";
        document.getElementById("aiProbBar").style.width = "0%";
        document.getElementById("aiActionRank").innerText = "#1";
        document.getElementById("aiStateVal").innerText = "-0.00";
        document.getElementById("aiAlternativesList").innerHTML = `<li class="empty-msg">Play a turn to observe policy evidence.</li>`;
        return;
    }

    document.getElementById("aiActionName").innerText = decision.action;
    const probPct = (decision.probability * 100).toFixed(1) + "%";
    document.getElementById("aiActionProb").innerText = probPct;
    document.getElementById("aiProbBar").style.width = probPct;
    document.getElementById("aiActionRank").innerText = `#${decision.rank}`;
    document.getElementById("aiStateVal").innerText = Number(decision.value_estimate || 0).toFixed(2);

    // RENDER LEGAL ALTERNATIVES
    const altList = document.getElementById("aiAlternativesList");
    altList.innerHTML = "";
    if (decision.top_legal_actions && decision.top_legal_actions.length > 0) {
        decision.top_legal_actions.forEach(item => {
            const li = document.createElement("li");
            li.className = "alt-item";
            li.innerHTML = `<span>${item.action}</span><strong>${(item.probability * 100).toFixed(1)}%</strong>`;
            altList.appendChild(li);
        });
    }
}

// FETCH OLLAMA LLM EXPLANATION
async function fetchLLMExplanation() {
    if (!CURRENT_SESSION_ID) return;

    const box = document.getElementById("explanationResultBox");
    const textEl = document.getElementById("expText");
    const bulletsEl = document.getElementById("expEvidenceBullets");
    const valBadge = document.getElementById("expValBadge");
    const btn = document.getElementById("explainBtn");

    box.classList.remove("hidden");
    textEl.innerText = "Generating post-hoc Ollama explanation using qwen3.5:4b...";
    bulletsEl.innerHTML = "";
    btn.disabled = true;

    try {
        const res = await fetch("/api/explain", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ session_id: CURRENT_SESSION_ID })
        });

        const data = await res.json();
        btn.disabled = false;

        if (!res.ok) {
            textEl.innerText = `Explanation Error: ${data.error || 'Ollama connection failed.'}`;
            return;
        }

        textEl.innerText = data.explanation || "No explanation text generated.";
        
        if (data.validation && data.validation.passed) {
            valBadge.className = "val-badge valid";
            valBadge.innerText = "✓ GROUNDING PASSED";
        } else {
            valBadge.className = "val-badge invalid";
            valBadge.innerText = "✕ GROUNDING MISMATCH";
        }

        if (data.bullet_evidence && data.bullet_evidence.length > 0) {
            data.bullet_evidence.forEach(b => {
                const li = document.createElement("li");
                li.innerText = b;
                bulletsEl.appendChild(li);
            });
        }
    } catch (err) {
        btn.disabled = false;
        textEl.innerText = "Failed to connect to local Ollama daemon.";
    }
}

// GAME OVER RESULT MODAL
function showGameOverModal(state) {
    const modal = document.getElementById("resultModal");
    const icon = document.getElementById("resultIcon");
    const title = document.getElementById("resultTitle");
    const sub = document.getElementById("resultSubtitle");

    if (state.mode === 'watch') {
        const winnerName = (state.winner === 0) ? (state.p0_name || "RL Agent 0 (MaskablePPO)") : (state.p1_name || "RL Agent 1 (MaskedDQN)");
        icon.innerText = "🏆";
        title.innerText = `${winnerName.toUpperCase()} WINS!`;
        title.style.color = (state.winner === 0 ? "#0284c7" : "#7c3aed");
        sub.innerText = `Autonomous match concluded in ${state.turn_index} turns.`;
    } else {
        const isWin = (state.winner === 0);
        if (isWin) {
            icon.innerText = "🏆";
            title.innerText = "YOU WIN!";
            title.style.color = "#10b981";
            sub.innerText = "You successfully outplayed the RL Agent!";
        } else {
            icon.innerText = "💀";
            title.innerText = "DEFEAT";
            title.style.color = "#f43f5e";
            sub.innerText = "The RL Agent eliminated you!";
        }
    }

    document.getElementById("resTurns").innerText = state.turn_index;
    document.getElementById("resDeck").innerText = state.deck_size;
    const dda = state.dda_status;
    const ddaText = dda ? (dda.is_fixed ? `L${dda.current_difficulty} (${dda.checkpoint || dda.checkpoint_name}) [Fixed]` : `L${dda.current_difficulty} (${dda.checkpoint || dda.checkpoint_name}) [Auto]`) : "L2";
    document.getElementById("resDDA").innerText = ddaText;

    modal.classList.remove("hidden");
}

// MODAL & DRAWER TOGGLES
function toggleDrawer(drawerId) {
    const drw = document.getElementById(drawerId);
    drw.classList.toggle("open");
}

function openModal(modalId) {
    const m = document.getElementById(modalId);
    m.classList.remove("hidden");

    if (modalId === 'actionLogModal' && CURRENT_GAME_STATE) {
        renderFullActionLog(CURRENT_GAME_STATE.action_log);
    }
}

function closeModal(modalId) {
    document.getElementById(modalId).classList.add("hidden");
}

function playAgainFromModal() {
    closeModal('resultModal');
    if (CURRENT_GAME_STATE && CURRENT_GAME_STATE.mode === 'watch') {
        startNewGame('watch', 2);
    } else {
        startNewGame('1v1', SELECTED_PLAYER_COUNT || 2);
    }
}

function explainFromResultModal() {
    closeModal('resultModal');
    const drw = document.getElementById("aiInsightsDrawer");
    if (drw && !drw.classList.contains("open")) {
        drw.classList.add("open");
    }
    fetchLLMExplanation();
}

function openFullLogFromResultModal() {
    closeModal('resultModal');
    openModal('actionLogModal');
}

function renderFullActionLog(logArray) {
    const list = document.getElementById("fullActionLogList");
    list.innerHTML = "";
    (logArray || []).forEach(msg => {
        const div = document.createElement("div");
        div.className = `log-item ${msg.includes("RL Agent") ? 'ai-move' : ''}`;
        div.innerText = msg;
        list.appendChild(div);
    });
}

// RESEARCH LAB TAB CONTROLLER
function switchResearchTab(paneId, btnEl) {
    document.querySelectorAll(".research-pane").forEach(p => p.classList.remove("active"));
    document.querySelectorAll(".res-tab-btn").forEach(b => b.classList.remove("active"));

    document.getElementById(paneId).classList.add("active");
    if (btnEl) btnEl.classList.add("active");
}

// LOAD EXPERIMENT METRICS
async function loadResearchMetrics() {
    try {
        const res = await fetch("/api/experiments");
        const data = await res.json();
        const box = document.getElementById("jsonMetricsContainer");
        if (box) {
            box.innerText = JSON.stringify(data, null, 2);
        }
    } catch (err) {
        console.error("Failed to load metrics:", err);
    }
}
