import React from "react";
import {
  AbsoluteFill,
  interpolate,
  spring,
  useCurrentFrame,
  useVideoConfig,
  Sequence,
  Easing,
} from "remotion";

// ─── shared helpers ───────────────────────────────────────────────────────────

const fade = (frame: number, start: number, duration = 15) =>
  interpolate(frame, [start, start + duration], [0, 1], {
    extrapolateRight: "clamp",
    extrapolateLeft: "clamp",
  });

const slideUp = (frame: number, start: number, fps: number) =>
  spring({ frame: frame - start, fps, config: { damping: 14, stiffness: 120 }, from: 40, to: 0 });

const COUNT_EASE = Easing.out(Easing.cubic);

function useCountUp(target: number, startFrame: number, durationFrames: number) {
  const frame = useCurrentFrame();
  const t = interpolate(frame, [startFrame, startFrame + durationFrames], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: COUNT_EASE,
  });
  return Math.round(t * target);
}

// ─── palette ──────────────────────────────────────────────────────────────────

const BG = "#07090f";
const CARD = "#0d1117";
const BORDER = "#1e293b";
const INDIGO = "#6366f1";
const INDIGO_LIGHT = "#a5b4fc";
const INDIGO_DIM = "#1e1b4b";
const RED = "#ef4444";
const GREEN = "#22c55e";
const MUTED = "#475569";
const TEXT = "#e2e8f0";
const SUBTEXT = "#64748b";
const YELLOW = "#eab308";

const mono = "'SF Mono', 'Fira Code', 'Cascadia Code', monospace";

// ─── Scene 1: Title  (frames 0–119, 4s) ──────────────────────────────────────

const SceneTitle: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const logoOpacity = fade(frame, 0, 20);
  const tagOpacity = fade(frame, 20, 15);
  const subOpacity = fade(frame, 35, 15);
  const logoY = slideUp(frame, 0, fps);

  return (
    <AbsoluteFill style={{ background: BG, justifyContent: "center", alignItems: "center", flexDirection: "column", gap: 16 }}>
      <div style={{ opacity: logoOpacity, transform: `translateY(${logoY}px)`, textAlign: "center" }}>
        <div style={{ fontFamily: mono, fontSize: 96, fontWeight: 900, color: INDIGO, letterSpacing: -6, lineHeight: 1 }}>
          fittok
        </div>
        <div style={{ opacity: subOpacity, fontFamily: mono, fontSize: 22, color: SUBTEXT, marginTop: 16, letterSpacing: 1 }}>
          stop paying for code you don't need
        </div>
      </div>
      <div style={{ opacity: tagOpacity, display: "flex", gap: 12, marginTop: 8 }}>
        {["MCP server", "CLI", "Python library"].map((t) => (
          <div key={t} style={{ background: INDIGO_DIM, color: INDIGO_LIGHT, fontFamily: mono, fontSize: 14, padding: "6px 16px", borderRadius: 6, border: `1px solid ${INDIGO}` }}>
            {t}
          </div>
        ))}
      </div>
    </AbsoluteFill>
  );
};

// ─── Scene 2: Without fittok  (frames 120–269, 5s) ───────────────────────────

const FILES = [
  { name: "src/store/board-store.ts", tokens: 4210 },
  { name: "src/lib/supabase.ts", tokens: 2891 },
  { name: "src/hooks/useAuth.ts", tokens: 1840 },
  { name: "src/app/api/auth/route.ts", tokens: 2440 },
  { name: "src/middleware.ts", tokens: 2138 },
];
const TOTAL_WITHOUT = 13519;

const SceneWithout: React.FC = () => {
  const frame = useCurrentFrame();

  return (
    <AbsoluteFill style={{ background: BG, padding: "50px 120px", flexDirection: "column", justifyContent: "center" }}>
      <div style={{ opacity: fade(frame, 0, 12), fontFamily: mono, fontSize: 13, color: SUBTEXT, marginBottom: 24, textTransform: "uppercase", letterSpacing: 2 }}>
        Without fittok — Claude reads every file
      </div>

      <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
        {FILES.map((f, i) => {
          const rowOpacity = fade(frame, i * 10, 8);
          const barW = interpolate(frame, [i * 10, i * 10 + 18], [0, (f.tokens / TOTAL_WITHOUT) * 100], {
            extrapolateLeft: "clamp", extrapolateRight: "clamp",
          });
          return (
            <div key={f.name} style={{ opacity: rowOpacity }}>
              <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 5 }}>
                <span style={{ fontFamily: mono, fontSize: 14, color: TEXT }}>{f.name}</span>
                <span style={{ fontFamily: mono, fontSize: 14, color: RED, fontWeight: 700 }}>+{f.tokens.toLocaleString()} tokens</span>
              </div>
              <div style={{ height: 5, background: "#1e293b", borderRadius: 3, overflow: "hidden" }}>
                <div style={{ width: `${barW}%`, height: "100%", background: RED, borderRadius: 3 }} />
              </div>
            </div>
          );
        })}
      </div>

      <div style={{ opacity: fade(frame, 70, 15), marginTop: 28, padding: "16px 24px", background: "#1a0808", border: `1px solid #7f1d1d`, borderRadius: 10, display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <span style={{ fontFamily: mono, fontSize: 16, color: SUBTEXT }}>Total · 5 files read</span>
        <span style={{ fontFamily: mono, fontSize: 40, fontWeight: 900, color: RED }}>{TOTAL_WITHOUT.toLocaleString()} tokens</span>
      </div>
    </AbsoluteFill>
  );
};

// ─── Scene 3: Pipeline  (frames 270–569, 10s) ─────────────────────────────────
// Stage by stage, each revealed slowly with extra context

const STEPS = [
  {
    icon: "🌳",
    name: "graphify",
    sub: "tree-sitter",
    detail: "Parses every function, class & method into a knowledge graph. Nodes = definitions. Edges = calls, imports, inherits.",
    color: INDIGO,
  },
  {
    icon: "🎯",
    name: "slurp",
    sub: "semantic scoring",
    detail: "Embeds query + every node. Scores via cosine similarity, TF-IDF and PageRank. Applies relevance cliff to drop noise.",
    color: YELLOW,
  },
  {
    icon: "📄",
    name: "readable slice",
    sub: "trim to budget",
    detail: "Top 6 nodes → full source. Supporting nodes → signature only. Real, readable code — no compression word-salad.",
    color: "#06b6d4",
  },
  {
    icon: "🤖",
    name: "LLM answers",
    sub: "one call",
    detail: "Model reads the focused slice and answers directly. No file reads. No subagent. Token budget: 600–3,500.",
    color: GREEN,
  },
];

const ScenePipeline: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  return (
    <AbsoluteFill style={{ background: BG, padding: "40px 80px", flexDirection: "column", justifyContent: "center", gap: 28 }}>
      <div style={{ opacity: fade(frame, 0, 12), fontFamily: mono, fontSize: 13, color: SUBTEXT, textTransform: "uppercase", letterSpacing: 2 }}>
        The fittok pipeline — 3 stages
      </div>

      <div style={{ display: "flex", alignItems: "flex-start", gap: 0 }}>
        {STEPS.map((s, i) => {
          const mountFrame = i * 60 + 10;
          const sp = spring({ frame: frame - mountFrame, fps, config: { damping: 18, stiffness: 90 }, from: 0, to: 1 });
          const cardOpacity = interpolate(sp, [0, 1], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
          const cardScale = interpolate(sp, [0, 1], [0.82, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
          const arrowOpacity = frame >= mountFrame ? fade(frame, mountFrame + 10, 12) : 0;

          return (
            <React.Fragment key={s.name}>
              <div style={{ opacity: cardOpacity, transform: `scale(${cardScale})`, textAlign: "center", width: 230 }}>
                <div style={{ background: `${s.color}18`, border: `1px solid ${s.color}`, borderRadius: 14, padding: "20px 14px", minHeight: 200 }}>
                  <div style={{ fontSize: 36 }}>{s.icon}</div>
                  <div style={{ fontFamily: mono, fontSize: 18, fontWeight: 700, color: s.color, marginTop: 10 }}>{s.name}</div>
                  <div style={{ fontFamily: mono, fontSize: 11, color: SUBTEXT, marginTop: 4, marginBottom: 10 }}>{s.sub}</div>
                  <div style={{ fontFamily: mono, fontSize: 11, color: TEXT, lineHeight: 1.7, textAlign: "left" }}>{s.detail}</div>
                </div>
              </div>
              {i < STEPS.length - 1 && (
                <div style={{ opacity: arrowOpacity, fontSize: 24, color: MUTED, padding: "0 4px", marginTop: 80 }}>→</div>
              )}
            </React.Fragment>
          );
        })}
      </div>

      <div style={{ opacity: fade(frame, 250, 15), fontFamily: mono, fontSize: 13, color: SUBTEXT, textAlign: "center" }}>
        codebase → only what's relevant → answer in one call
      </div>
    </AbsoluteFill>
  );
};

// ─── Scene 4: Relevance cliff  (frames 570–869, 10s) ─────────────────────────

const SEEDED = Array.from({ length: 120 }, (_, i) => ({
  id: i,
  score: ((i * 137 + 31) % 100) / 100,
  relevant: i % 9 === 0 || i % 13 === 0 || i < 3,
}));

const CLIFF_EXPLAIN = [
  { text: "Every node is scored by cosine similarity against your query.", color: TEXT, delay: 100 },
  { text: "Top 60% of the score range survive. Below 0.22 cosine → dropped.", color: INDIGO_LIGHT, delay: 140 },
  { text: "Result: tight selection, zero noise, no budget-padding.", color: GREEN, delay: 180 },
];

const SceneCliff: React.FC = () => {
  const frame = useCurrentFrame();

  const cliffX = interpolate(frame, [30, 90], [100, 20], {
    extrapolateLeft: "clamp", extrapolateRight: "clamp",
    easing: Easing.inOut(Easing.cubic),
  });

  const countShown = Math.round(interpolate(frame, [30, 90], [120, 14], {
    extrapolateLeft: "clamp", extrapolateRight: "clamp",
  }));

  return (
    <AbsoluteFill style={{ background: BG, padding: "48px 100px", flexDirection: "column", justifyContent: "center", gap: 28 }}>
      <div style={{ opacity: fade(frame, 0, 15), fontFamily: mono, fontSize: 14, color: SUBTEXT, textTransform: "uppercase", letterSpacing: 2 }}>
        Relevance cliff in action — 4,039 nodes → 14
      </div>

      <div style={{ display: "flex", gap: 48, alignItems: "center" }}>
        {/* dot grid */}
        <div style={{ display: "grid", gridTemplateColumns: "repeat(12, 1fr)", gap: 10, flex: 1 }}>
          {SEEDED.slice(0, 120).map((n) => {
            const isLit = n.relevant && cliffX < 50;
            const isDimmed = !isLit && cliffX < 80;
            return (
              <div key={n.id} style={{
                width: 18, height: 18, borderRadius: "50%",
                background: isLit ? INDIGO : "#1e293b",
                border: `1px solid ${isLit ? INDIGO : "#334155"}`,
                opacity: isDimmed && !isLit ? 0.2 : 1,
              }} />
            );
          })}
        </div>

        {/* stats + what is the cliff */}
        <div style={{ width: 260, fontFamily: mono }}>
          <div style={{ marginBottom: 16 }}>
            <div style={{ fontSize: 13, color: SUBTEXT }}>Total nodes in graph</div>
            <div style={{ fontSize: 36, fontWeight: 800, color: TEXT }}>4,039</div>
          </div>
          <div style={{ height: 1, background: BORDER, marginBottom: 16 }} />
          <div style={{ marginBottom: 16 }}>
            <div style={{ fontSize: 13, color: SUBTEXT }}>After relevance cliff</div>
            <div style={{ fontSize: 36, fontWeight: 800, color: GREEN }}>{countShown}</div>
          </div>
          <div style={{ background: "#052e16", border: `1px solid #14532d`, borderRadius: 8, padding: "10px 14px", fontSize: 12, color: GREEN, lineHeight: 1.6 }}>
            ✂ cliff @ cosine 0.22<br />
            top 60% of score range<br />
            no budget-padding with noise
          </div>
        </div>
      </div>

      {/* inline explanation — fades in after the cliff animation */}
      <div style={{ display: "flex", flexDirection: "column", gap: 10, borderTop: `1px solid ${BORDER}`, paddingTop: 20 }}>
        <div style={{ opacity: fade(frame, 95, 10), fontFamily: mono, fontSize: 12, color: SUBTEXT, textTransform: "uppercase", letterSpacing: 1, marginBottom: 4 }}>
          What is the relevance cliff?
        </div>
        {CLIFF_EXPLAIN.map((l, i) => (
          <div key={i} style={{ opacity: fade(frame, l.delay, 12), fontFamily: mono, fontSize: 14, color: l.color, display: "flex", gap: 10, alignItems: "flex-start" }}>
            <span style={{ color: INDIGO }}>›</span>
            <span>{l.text}</span>
          </div>
        ))}
      </div>
    </AbsoluteFill>
  );
};

// ─── Scene 5: Savings bars  (frames 990–1289, 10s) ───────────────────────────

const SceneSavings: React.FC = () => {
  const frame = useCurrentFrame();
  const withoutW = interpolate(frame, [10, 45], [0, 100], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
  const withW = interpolate(frame, [50, 85], [0, 20.3], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
  const pct = useCountUp(797, 60, 50);

  return (
    <AbsoluteFill style={{ background: BG, padding: "60px 160px", flexDirection: "column", justifyContent: "center", gap: 40 }}>
      <div style={{ opacity: fade(frame, 0, 15), fontFamily: mono, fontSize: 14, color: SUBTEXT, textTransform: "uppercase", letterSpacing: 2 }}>
        Same question · same answer · fewer tokens
      </div>

      <div style={{ display: "flex", flexDirection: "column", gap: 24 }}>
        <div>
          <div style={{ display: "flex", justifyContent: "space-between", fontFamily: mono, fontSize: 16, marginBottom: 10 }}>
            <span style={{ color: SUBTEXT }}>Without fittok</span>
            <span style={{ color: RED, fontWeight: 700 }}>13,519 tokens</span>
          </div>
          <div style={{ height: 40, background: "#1e293b", borderRadius: 6, overflow: "hidden" }}>
            <div style={{ width: `${withoutW}%`, height: "100%", background: RED, display: "flex", alignItems: "center", paddingLeft: 14 }}>
              <span style={{ fontFamily: mono, fontSize: 13, fontWeight: 700, color: "white", opacity: withoutW > 30 ? 1 : 0 }}>13,519</span>
            </div>
          </div>
        </div>

        <div style={{ opacity: frame > 45 ? 1 : 0 }}>
          <div style={{ display: "flex", justifyContent: "space-between", fontFamily: mono, fontSize: 16, marginBottom: 10 }}>
            <span style={{ color: SUBTEXT }}>With fittok</span>
            <span style={{ color: GREEN, fontWeight: 700 }}>2,750 tokens</span>
          </div>
          <div style={{ height: 40, background: "#1e293b", borderRadius: 6, overflow: "hidden" }}>
            <div style={{ width: `${withW}%`, height: "100%", background: GREEN, display: "flex", alignItems: "center", paddingLeft: 14 }}>
              <span style={{ fontFamily: mono, fontSize: 13, fontWeight: 700, color: "white", opacity: withW > 10 ? 1 : 0 }}>2,750</span>
            </div>
          </div>
        </div>
      </div>

      <div style={{ opacity: fade(frame, 80, 20), textAlign: "center" }}>
        <div style={{ fontFamily: mono, fontSize: 72, fontWeight: 900, color: GREEN }}>{(pct / 10).toFixed(1)}%</div>
        <div style={{ fontFamily: mono, fontSize: 16, color: SUBTEXT, marginTop: 8 }}>fewer tokens · deterministic · every run</div>
        <div style={{ fontFamily: mono, fontSize: 14, color: INDIGO_LIGHT, marginTop: 16, background: INDIGO_DIM, display: "inline-block", padding: "8px 20px", borderRadius: 8 }}>
          🪙 saved 79.7% — 2,750 vs 13,519 tokens
        </div>
      </div>
    </AbsoluteFill>
  );
};

// ─── Scene 6: Side-by-side terminal  (frames 1290–1649, 12s) ─────────────────

// ─── Claude Code TUI constants ────────────────────────────────────────────────

const CC_ORANGE = "#d4822a";
const CC_BG     = "#0a0a0a";

type CCLine =
  | { kind: "prompt"; text: string }
  | { kind: "tool";   icon: string; text: string; color: string }
  | { kind: "output"; text: string; color: string }
  | { kind: "answer"; text: string };

const WITHOUT_CC: CCLine[] = [
  { kind: "prompt", text: "how does auth work?" },
  { kind: "tool",   icon: "⚙", text: "Grep(\"auth\") → 12 files",        color: MUTED   },
  { kind: "tool",   icon: "⚙", text: "Launching Explore subagent...",          color: YELLOW  },
  { kind: "tool",   icon: "↳", text: "Read board-store.ts   +4,210 tok", color: RED     },
  { kind: "tool",   icon: "↳", text: "Read supabase.ts      +2,891 tok", color: RED     },
  { kind: "tool",   icon: "↳", text: "Read useAuth.ts       +1,840 tok", color: RED     },
  { kind: "tool",   icon: "↳", text: "Read auth/route.ts    +2,440 tok", color: RED     },
  { kind: "tool",   icon: "↳", text: "Read middleware.ts    +2,138 tok", color: RED     },
  { kind: "output", text: "Subagent: 58,400 tokens consumed",                  color: YELLOW  },
  { kind: "answer", text: "Auth uses Supabase JWT sessions — middleware validates the token on every request..." },
];

const WITH_CC: CCLine[] = [
  { kind: "prompt", text: "how does auth work?" },
  { kind: "tool",   icon: "⚙", text: "fittok › optimize_context",         color: GREEN        },
  { kind: "output", text: "14 nodes · 2,750 tokens · saved 79.7%",   color: INDIGO_LIGHT },
  { kind: "answer", text: "Auth uses Supabase JWT sessions — middleware validates the token on every request..." },
];

const ClaudeCodePanel: React.FC<{
  label: string;
  labelColor: string;
  borderColor: string;
  lines: CCLine[];
  startFrame: number;
  framesPerLine: number;
}> = ({ label, labelColor, borderColor, lines, startFrame, framesPerLine }) => {
  const frame = useCurrentFrame();
  const linesShown = Math.floor(
    interpolate(frame, [startFrame, startFrame + lines.length * framesPerLine], [0, lines.length], {
      extrapolateLeft: "clamp", extrapolateRight: "clamp",
    })
  );
  const allDone = linesShown >= lines.length;
  const cursor  = Math.round(frame * 0.15) % 2 === 0;

  return (
    <div style={{ flex: 1, display: "flex", flexDirection: "column", background: CC_BG, border: `1px solid ${borderColor}`, borderRadius: 6, fontFamily: mono, fontSize: 12, overflow: "hidden" }}>

      {/* title bar */}
      <div style={{ background: CC_BG, borderBottom: `1px solid ${borderColor}`, padding: "7px 14px", display: "flex", alignItems: "center", gap: 10 }}>
        <span style={{ color: CC_ORANGE, fontSize: 11 }}>{"──"} Claude Code v2.1.177</span>
        <span style={{ flex: 1 }} />
        <span style={{ color: labelColor, fontSize: 11, fontWeight: 700 }}>{label}</span>
      </div>

      {/* workspace row */}
      <div style={{ borderBottom: "1px solid #1a1a1a", padding: "5px 14px", color: "#555", fontSize: 10 }}>
        Opus 4.8 (1M context) · ~/MyStuff/Projects/packright
      </div>

      {/* conversation body */}
      <div style={{ flex: 1, padding: "12px 14px", display: "flex", flexDirection: "column", gap: 7, overflow: "hidden" }}>
        {lines.slice(0, linesShown).map((l, i) => {
          if (l.kind === "prompt") return (
            <div key={i} style={{ display: "flex", gap: 8, alignItems: "flex-start" }}>
              <span style={{ color: CC_ORANGE, flexShrink: 0 }}>&gt;</span>
              <span style={{ color: "#e0e0e0" }}>{l.text}</span>
            </div>
          );
          if (l.kind === "tool") return (
            <div key={i} style={{ display: "flex", gap: 8, alignItems: "flex-start", paddingLeft: 16 }}>
              <span style={{ color: l.color, flexShrink: 0 }}>{l.icon}</span>
              <span style={{ color: l.color }}>{l.text}</span>
            </div>
          );
          if (l.kind === "output") return (
            <div key={i} style={{ paddingLeft: 32, color: l.color, fontSize: 11 }}>
              {"⎿"} {l.text}
            </div>
          );
          if (l.kind === "answer") return (
            <div key={i} style={{ marginTop: 6, color: "#c0c0c0", lineHeight: 1.6, borderLeft: `2px solid ${CC_ORANGE}44`, paddingLeft: 10 }}>
              {l.text}
            </div>
          );
          return null;
        })}
        {!allDone && <span style={{ opacity: cursor ? 1 : 0, color: CC_ORANGE }}>█</span>}
      </div>

      {/* status bar */}
      <div style={{ borderTop: "1px solid #1a1a1a", padding: "5px 14px", display: "flex", justifyContent: "space-between" }}>
        <span style={{ color: CC_ORANGE, fontSize: 10 }}>{"►► auto mode on · ← for agents"}</span>
        <span style={{ color: "#444", fontSize: 10 }}>{"● high · /effort"}</span>
      </div>
    </div>
  );
};

const SceneTerminalAB: React.FC = () => {
  const frame = useCurrentFrame();
  const doneFr = 10 + WITHOUT_CC.length * 22 + 10;

  return (
    <AbsoluteFill style={{ background: "#050505", padding: "44px 48px", flexDirection: "column", justifyContent: "center", gap: 16 }}>
      <div style={{ opacity: fade(frame, 0, 12), fontFamily: mono, fontSize: 12, color: SUBTEXT, textTransform: "uppercase", letterSpacing: 2 }}>
        Live comparison — same question, same codebase
      </div>

      <div style={{ display: "flex", gap: 20, flex: 1, alignItems: "stretch" }}>
        <ClaudeCodePanel label="Without fittok" labelColor={RED}   borderColor={RED}   lines={WITHOUT_CC} startFrame={10} framesPerLine={22} />
        <ClaudeCodePanel label="With fittok"    labelColor={GREEN} borderColor={GREEN} lines={WITH_CC}    startFrame={10} framesPerLine={22} />
      </div>

      <div style={{ opacity: fade(frame, doneFr, 15), display: "flex", justifyContent: "space-between", fontFamily: mono, fontSize: 12 }}>
        <span style={{ color: RED }}>13,519 tokens consumed</span>
        <span style={{ color: INDIGO_LIGHT }}>fittok replaced 5 file reads with 1 tool call</span>
        <span style={{ color: GREEN }}>2,750 tokens — 79.7% saved</span>
      </div>
    </AbsoluteFill>
  );
};

// ─── Scene 7: Real Opus 4.8 A/B  (frames 1650–1949, 10s) ────────────────────

const SceneAB: React.FC = () => {
  const frame = useCurrentFrame();
  const without = useCountUp(84000, 10, 60);
  const withVal = useCountUp(27000, 40, 60);

  return (
    <AbsoluteFill style={{ background: BG, padding: "60px 100px", flexDirection: "column", justifyContent: "center", gap: 32 }}>
      <div style={{ opacity: fade(frame, 0, 15), fontFamily: mono, fontSize: 14, color: SUBTEXT, textTransform: "uppercase", letterSpacing: 2 }}>
        Real Opus 4.8 A/B · same question · same codebase
      </div>

      <div style={{ display: "flex", gap: 32 }}>
        <div style={{ flex: 1, background: "#1a0808", border: `1px solid #7f1d1d`, borderRadius: 14, padding: 32 }}>
          <div style={{ fontFamily: mono, fontSize: 13, color: SUBTEXT, marginBottom: 16, textTransform: "uppercase", letterSpacing: 1 }}>Without fittok</div>
          <div style={{ fontFamily: mono, fontSize: 56, fontWeight: 900, color: RED }}>~{Math.round(without / 1000)}k</div>
          <div style={{ fontFamily: mono, fontSize: 13, color: MUTED, marginTop: 12, lineHeight: 1.8 }}>
            26k messages<br />
            <span style={{ color: RED }}>+ 58k Explore subagent</span><br />
            (file crawl — 5+ reads)
          </div>
        </div>

        <div style={{ flex: 1, background: "#052e16", border: `1px solid #14532d`, borderRadius: 14, padding: 32 }}>
          <div style={{ fontFamily: mono, fontSize: 13, color: SUBTEXT, marginBottom: 16, textTransform: "uppercase", letterSpacing: 1 }}>With fittok</div>
          <div style={{ fontFamily: mono, fontSize: 56, fontWeight: 900, color: GREEN }}>~{Math.round(withVal / 1000)}k</div>
          <div style={{ fontFamily: mono, fontSize: 13, color: MUTED, marginTop: 12, lineHeight: 1.8 }}>
            1 tool call<br />
            <span style={{ color: GREEN }}>0 file reads</span><br />
            answered directly from slice
          </div>
        </div>
      </div>

      <div style={{ opacity: fade(frame, 80, 20), textAlign: "center", fontFamily: mono }}>
        <div style={{ fontSize: 22, fontWeight: 700, color: GREEN }}>↓ 68% total token reduction</div>
        <div style={{ fontSize: 13, color: SUBTEXT, marginTop: 8 }}>fittok eliminated the entire 58k-token Explore crawl</div>
      </div>
    </AbsoluteFill>
  );
};

// ─── Scene 8: CTA  (frames 1830–2369, 18s) ───────────────────────────────────
//
// Phase 1 (0–230):  two terminal sessions animate side by side
// Phase 2 (250+):   crossfade to grouped CLI / MCP end-state

// CLI session lines (animation panel)
const CLI_LINES = [
  { prompt: true,  text: "uvx fittok index",                        comment: "" },
  { prompt: false, text: "Indexed 4039 nodes / 9302 edges. Cached.", comment: "" },
  { prompt: true,  text: 'uvx fittok query "how does auth work"',   comment: "" },
  { prompt: false, text: "## Relevant code",                        comment: "" },
  { prompt: false, text: "  claimItem()  markAsPacked()  ...",      comment: "" },
  { prompt: false, text: "saved 79.7% — 2,750 vs 13,519 tokens",   comment: "" },
];

// MCP session lines (animation panel)
const MCP_LINES = [
  { prompt: true,  text: "claude mcp add fittok -s user -- uvx fittok", comment: "" },
  { prompt: false, text: "MCP server 'fittok' added (user scope)",       comment: "" },
  { prompt: false, text: "",                                              comment: "" },
  { prompt: false, text: "# Restart → /mcp → connected",                comment: "" },
  { prompt: false, text: "# Ask codebase questions normally.",            comment: "" },
  { prompt: false, text: "# fittok fires automatically.",                 comment: "" },
];

// CLI: 15 + 6*18 = 123. MCP: 120 + 6*16 = 216. Give a few extra frames buffer.
const ANIM_DONE      = 225;
const CROSSFADE_START = 240;
// End-state command typing starts after crossfade
const END_START = CROSSFADE_START + 25;

type CtaLine = { prompt: boolean; text: string; comment: string };

const TerminalSession: React.FC<{
  heading: string; headingColor: string; borderColor: string;
  lines: CtaLine[]; startFrame: number; framePerLine: number;
}> = ({ heading, headingColor, borderColor, lines, startFrame, framePerLine }) => {
  const frame = useCurrentFrame();
  const linesShown = Math.floor(
    interpolate(frame, [startFrame, startFrame + lines.length * framePerLine], [0, lines.length], {
      extrapolateLeft: "clamp", extrapolateRight: "clamp",
    })
  );
  return (
    <div style={{ flex: 1, display: "flex", flexDirection: "column", gap: 10, overflow: "hidden" }}>
      <div style={{ opacity: fade(frame, startFrame - 5, 10), fontFamily: mono, fontSize: 12, color: headingColor, textTransform: "uppercase", letterSpacing: 2, fontWeight: 700 }}>
        {heading}
      </div>
      <div style={{ background: CARD, border: `1px solid ${borderColor}`, borderRadius: 10, padding: "14px 18px", fontFamily: mono, fontSize: 12, flex: 1, overflow: "hidden" }}>
        {lines.slice(0, linesShown).map((l, i) => (
          <div key={i} style={{ display: "flex", gap: 6, marginBottom: 5, alignItems: "baseline" }}>
            {l.prompt ? <span style={{ color: INDIGO, flexShrink: 0 }}>$</span> : <span style={{ color: "transparent", flexShrink: 0 }}>$</span>}
            <span style={{ color: l.prompt ? "#7ee787" : l.text.startsWith("#") ? SUBTEXT : l.text.startsWith("saved") ? INDIGO_LIGHT : MUTED, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
              {l.text}
            </span>
          </div>
        ))}
        {linesShown < lines.length && <span style={{ opacity: Math.round(frame * 0.15) % 2 === 0 ? 1 : 0, color: GREEN }}>█</span>}
      </div>
    </div>
  );
};

// ─── End-state grouped layout ─────────────────────────────────────────────────

const MCP_JSON = `{
  "mcpServers": {
    "fittok": {
      "command": "uvx",
      "args": ["fittok"]
    }
  }
}`;

const SceneCTA: React.FC = () => {
  const frame = useCurrentFrame();
  const terminalOpacity = interpolate(frame, [CROSSFADE_START, CROSSFADE_START + 18], [1, 0], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
  const endOpacity      = interpolate(frame, [CROSSFADE_START, CROSSFADE_START + 18], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });

  const type = (text: string, start: number, speed = 1.1) => {
    const n = Math.round(interpolate(frame, [start, start + text.length * speed], [0, text.length], { extrapolateLeft: "clamp", extrapolateRight: "clamp" }));
    return text.slice(0, n);
  };
  const cur = (start: number, len: number, speed = 1.1) =>
    frame > start && frame < start + len * speed
      ? <span style={{ opacity: Math.round(frame * 0.1) % 2 === 0 ? 1 : 0, color: GREEN }}>█</span>
      : null;

  // staggered start frames for end-state commands
  const cmd0Start = END_START;
  const cmd1Start = cmd0Start + 20 + "uvx fittok index".length * 1.1;
  const cmd2Start = cmd1Start + 20 + 'uvx fittok query "how does auth work"'.length * 1.1;
  const jsonStart  = cmd2Start + 30 + "claude mcp add fittok -s user -- uvx fittok".length * 1.1;
  const footerStart = jsonStart + 60;

  return (
    <AbsoluteFill style={{ background: BG, padding: "44px 80px", flexDirection: "column", justifyContent: "center", gap: 20 }}>
      <div style={{ opacity: fade(frame, 0, 15), fontFamily: mono, fontSize: 14, color: SUBTEXT, textTransform: "uppercase", letterSpacing: 2 }}>
        Get started in 30 seconds
      </div>

      {/* ── Phase 1: terminal animation panels ── */}
      <div style={{ opacity: terminalOpacity, display: "flex", gap: 24, flex: 1, alignItems: "stretch" }}>
        <TerminalSession heading="CLI — no install" headingColor={INDIGO_LIGHT} borderColor={INDIGO}    lines={CLI_LINES} startFrame={15}  framePerLine={18} />
        <TerminalSession heading="MCP — Claude Code" headingColor={GREEN}       borderColor="#14532d" lines={MCP_LINES} startFrame={120} framePerLine={16} />
      </div>

      {/* ── Phase 2: grouped end-state ── */}
      <div style={{ opacity: endOpacity, position: "absolute", left: 80, right: 80, top: 90, bottom: 60, display: "flex", flexDirection: "column", gap: 28 }}>

        {/* CLI group */}
        <div>
          <div style={{ fontFamily: mono, fontSize: 12, color: INDIGO_LIGHT, textTransform: "uppercase", letterSpacing: 2, marginBottom: 12 }}>CLI</div>
          <div style={{ background: CARD, border: `1px solid ${BORDER}`, borderRadius: 10, padding: "16px 22px", fontFamily: mono, fontSize: 15, display: "flex", flexDirection: "column", gap: 12 }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <span>
                <span style={{ color: MUTED }}>$ </span>
                <span style={{ color: "#7ee787" }}>{type("uvx fittok index", cmd0Start)}</span>
                {cur(cmd0Start, "uvx fittok index".length)}
              </span>
              <span style={{ opacity: fade(frame, cmd1Start, 10), color: SUBTEXT, fontSize: 12 }}># pre-warm — parse repo into graph + cache embeddings (~15s)</span>
            </div>
            <div style={{ opacity: frame >= cmd1Start ? 1 : 0, display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <span>
                <span style={{ color: MUTED }}>$ </span>
                <span style={{ color: "#7ee787" }}>{type('uvx fittok query "how does auth work"', cmd1Start)}</span>
                {cur(cmd1Start, 'uvx fittok query "how does auth work"'.length)}
              </span>
              <span style={{ opacity: fade(frame, cmd2Start, 10), color: SUBTEXT, fontSize: 12 }}># retrieve relevant slice + savings footer</span>
            </div>
          </div>
        </div>

        {/* MCP group */}
        <div>
          <div style={{ fontFamily: mono, fontSize: 12, color: GREEN, textTransform: "uppercase", letterSpacing: 2, marginBottom: 12 }}>MCP</div>
          <div style={{ background: CARD, border: `1px solid ${BORDER}`, borderRadius: 10, padding: "16px 22px", fontFamily: mono, fontSize: 15, display: "flex", flexDirection: "column", gap: 12 }}>
            {/* Claude Code command */}
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <span>
                <span style={{ color: MUTED }}>$ </span>
                <span style={{ color: "#7ee787" }}>{type("claude mcp add fittok -s user -- uvx fittok", cmd2Start)}</span>
                {cur(cmd2Start, "claude mcp add fittok -s user -- uvx fittok".length)}
              </span>
              <span style={{ opacity: fade(frame, jsonStart, 10), color: SUBTEXT, fontSize: 12 }}># Claude Code (all projects)</span>
            </div>
            {/* JSON snippet for Cursor/Windsurf */}
            <div style={{ opacity: fade(frame, jsonStart, 15), borderTop: `1px solid ${BORDER}`, paddingTop: 10 }}>
              <div style={{ color: SUBTEXT, fontSize: 11, marginBottom: 6 }}>Cursor / Windsurf / any MCP client — add to config:</div>
              <pre style={{ margin: 0, color: INDIGO_LIGHT, fontSize: 12, lineHeight: 1.6 }}>{type(MCP_JSON, jsonStart, 0.6)}</pre>
            </div>
          </div>
        </div>

        {/* footer */}
        <div style={{ opacity: fade(frame, footerStart, 20), display: "flex", gap: 24, alignItems: "center", justifyContent: "center" }}>
          <span style={{ fontFamily: mono, fontSize: 13, color: SUBTEXT }}>github.com/likhithreddy/fittok</span>
          <div style={{ width: 1, height: 14, background: BORDER }} />
          <span style={{ fontFamily: mono, fontSize: 13, color: SUBTEXT }}>pypi.org/project/fittok</span>
          <div style={{ width: 1, height: 14, background: BORDER }} />
          <span style={{ fontFamily: mono, fontSize: 22, fontWeight: 900, color: INDIGO, letterSpacing: -1 }}>fittok</span>
        </div>
      </div>
    </AbsoluteFill>
  );
};

// ─── Root composition  (total: 2190 frames = 73s) ────────────────────────────

export const FittokDemo: React.FC = () => (
  <AbsoluteFill style={{ background: BG }}>
    <Sequence from={0}    durationInFrames={120}><SceneTitle /></Sequence>
    <Sequence from={120}  durationInFrames={150}><SceneWithout /></Sequence>
    <Sequence from={270}  durationInFrames={300}><ScenePipeline /></Sequence>
    <Sequence from={570}  durationInFrames={300}><SceneCliff /></Sequence>
    <Sequence from={870}  durationInFrames={300}><SceneSavings /></Sequence>
    <Sequence from={1170} durationInFrames={360}><SceneTerminalAB /></Sequence>
    <Sequence from={1530} durationInFrames={300}><SceneAB /></Sequence>
    <Sequence from={1830} durationInFrames={540}><SceneCTA /></Sequence>
  </AbsoluteFill>
);
