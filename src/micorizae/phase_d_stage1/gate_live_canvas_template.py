"""Plantilla gate-am-live-metrics.canvas.tsx v16.

Layout completo original: Pasos 1-3, stats, graficos, tablas.
Sin logica de scroll. Stats vivos en bloque LIVE_STATS (parche ligero).
"""

GATE_LIVE_CANVAS_TSX = r'''import {
  BarChart,
  Card,
  CardBody,
  CardHeader,
  Divider,
  Grid,
  H1,
  H3,
  LineChart,
  Pill,
  Row,
  Stack,
  Stat,
  Table,
  Text,
  useHostTheme,
} from "cursor/canvas";

type Tone = "neutral" | "success" | "warning" | "info";
type StatVM = { label: string; value: string };
type PhaseVM = {
  id: string;
  label: string;
  status_label: string;
  tone: Tone;
  pct_label: string;
  pct_bar: number | null;
  eta_label: string;
  tiles_label: string;
  speed_label: string;
  detail: string;
};
type SeriesVM = { name: string; data: number[] };
type ChartVM = { categories: string[]; series: SeriesVM[]; caption: string };
type TableVM = { columns: string[]; rows: string[][] };

type ViewModel = {
  canvas_version?: number;
  header: {
    run_id: string;
    status: string;
    fase_label: string;
    fase_tone: Tone;
    fase_pill: string;
    is_cache: boolean;
    synced_at: string;
    checkpoint_metric: string;
  };
  phases: PhaseVM[];
  show_train: boolean;
  train_stats: StatVM[];
  eta_stats: StatVM[];
  target_stats: StatVM[];
  epoch_curves: ChartVM | null;
  recall_bars: ChartVM | null;
  targets_table: TableVM | null;
  history_table: TableVM | null;
  eta_note: string | null;
};

type LiveStats = { train: StatVM[]; eta: StatVM[]; synced_at: string };

// CANVAS_TEMPLATE_VERSION = 16
// marker: pipeline_phases
// marker: canvas_layout_v16
// LIVE_STATS_BEGIN
const LIVE_STATS: LiveStats | null = null;
// LIVE_STATS_END
// LIVE_METRICS_BEGIN
const LIVE_METRICS_SNAPSHOT: ViewModel | null = null;
const LIVE_METRICS_SYNCED_AT = "";
// LIVE_METRICS_END

const COMPACT = { fontSize: "0.85em", lineHeight: 1.35 };

function StatGrid({ items, columns }: { items: StatVM[]; columns: number }) {
  return (
    <Grid columns={columns} gap={8}>
      {items.map((s, i) => (
        <Stat key={i} label={s.label} value={s.value} />
      ))}
    </Grid>
  );
}

function PhaseCard({ step }: { step: PhaseVM }) {
  const theme = useHostTheme();
  return (
    <div
      style={{
        border: `1px solid ${theme.stroke.secondary}`,
        borderRadius: 6,
        padding: 8,
        background: theme.bg.elevated,
        height: "100%",
        display: "flex",
        flexDirection: "column",
        gap: 6,
      }}
    >
      <Row align="center" justify="space-between" style={{ gap: 6, wrap: true }}>
        <Text style={{ fontWeight: 600, fontSize: "0.95em" }}>{step.label}</Text>
        <Pill tone={step.tone}>{step.status_label}</Pill>
      </Row>
      <Grid columns={2} gap={6}>
        <Stat label="Avance" value={step.pct_label} />
        <Stat label="ETA" value={step.eta_label} />
        <Stat label="Tiles" value={step.tiles_label} />
        <Stat label="Velocidad" value={step.speed_label} />
      </Grid>
      {step.pct_bar != null && (
        <div
          style={{
            height: 5,
            width: "100%",
            borderRadius: 999,
            background: theme.fill.secondary,
            overflow: "hidden",
          }}
        >
          <div
            style={{
              width: `${step.pct_bar}%`,
              height: "100%",
              background: theme.accent.primary,
            }}
          />
        </div>
      )}
      {step.detail ? (
        <Text muted style={{ fontSize: "0.82em", wordBreak: "break-word" }}>
          {step.detail}
        </Text>
      ) : null}
    </div>
  );
}

export default function GateAmLiveMetrics() {
  const m = LIVE_METRICS_SNAPSHOT;
  const syncAt = LIVE_STATS?.synced_at || LIVE_METRICS_SYNCED_AT;

  if (!m) {
    return (
      <Stack gap={12} style={COMPACT}>
        <H1 style={{ fontSize: "1.35em", margin: 0 }}>Gate AM — Live</H1>
        <Text muted>Esperando metricas. Inicia: python run.py</Text>
      </Stack>
    );
  }

  const trainStats =
    LIVE_STATS?.train && LIVE_STATS.train.length > 0 ? LIVE_STATS.train : m.train_stats;
  const etaStats =
    LIVE_STATS?.eta && LIVE_STATS.eta.length > 0 ? LIVE_STATS.eta : m.eta_stats;

  return (
    <Stack gap={12} style={COMPACT}>
      <Grid columns={3} gap={10}>
        <div style={{ gridColumn: "1 / -1" }}>
          <Row align="center" justify="space-between" style={{ wrap: true, gap: 8 }}>
            <H1 style={{ fontSize: "1.35em", margin: 0 }}>Gate AM — Live</H1>
            <Pill tone={m.header.status === "running" ? "info" : "neutral"}>
              {m.header.status}
            </Pill>
          </Row>
          <Text muted style={{ marginTop: 4 }}>
            run {m.header.run_id} · sync {syncAt || m.header.synced_at}
          </Text>
        </div>
        <div style={{ gridColumn: "1 / -1" }}>
          <Row align="center" gap={8} style={{ wrap: true }}>
            <Text>Fase: {m.header.fase_label}</Text>
            <Pill tone={m.header.fase_tone}>{m.header.fase_pill}</Pill>
          </Row>
        </div>
        {m.phases.map((step) => (
          <div key={step.id} style={{ minWidth: 0 }}>
            <PhaseCard step={step} />
          </div>
        ))}
      </Grid>

      {m.show_train && (
        <>
          <Divider />
          <StatGrid items={trainStats} columns={4} />
          {etaStats.length > 0 && <StatGrid items={etaStats} columns={3} />}
          <Divider />
          <StatGrid items={m.target_stats} columns={4} />
        </>
      )}

      {(m.epoch_curves || m.recall_bars) && (
        <Grid columns={2} gap={10}>
          {m.epoch_curves && (
            <Card>
              <CardHeader>
                <H3 style={{ fontSize: "1em", margin: 0 }}>Curvas por epoca</H3>
              </CardHeader>
              <CardBody>
                <LineChart
                  categories={m.epoch_curves.categories}
                  series={m.epoch_curves.series}
                  height={220}
                />
                <Text muted style={{ marginTop: 6, fontSize: "0.82em" }}>
                  {m.epoch_curves.caption}
                </Text>
              </CardBody>
            </Card>
          )}
          {m.recall_bars && (
            <Card>
              <CardHeader>
                <H3 style={{ fontSize: "1em", margin: 0 }}>Recall vs umbral G1 (por clase)</H3>
              </CardHeader>
              <CardBody>
                <BarChart
                  categories={m.recall_bars.categories}
                  series={m.recall_bars.series}
                  height={220}
                />
                <Text muted style={{ marginTop: 6, fontSize: "0.82em" }}>
                  {m.recall_bars.caption}
                </Text>
              </CardBody>
            </Card>
          )}
        </Grid>
      )}

      {m.targets_table && (
        <>
          <H3 style={{ fontSize: "1em", margin: 0 }}>Objetivos G1 (config)</H3>
          <Table headers={m.targets_table.columns} rows={m.targets_table.rows} />
        </>
      )}

      {m.history_table && (
        <>
          <H3 style={{ fontSize: "1em", margin: 0 }}>Historial epocas</H3>
          <Table
            headers={m.history_table.columns}
            rows={m.history_table.rows}
            framed
            striped
            stickyHeader
          />
        </>
      )}

      {m.eta_note && (
        <Text muted style={{ fontSize: "0.78em" }}>
          {m.eta_note}
        </Text>
      )}
    </Stack>
  );
}
'''
