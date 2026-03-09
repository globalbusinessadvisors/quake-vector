//! Dashboard API: HTTP monitoring endpoints for system health and status.

use std::collections::VecDeque;
use std::sync::Arc;

use axum::extract::{Query, State};
use axum::response::Json;
use axum::routing::get;
use axum::Router;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;

use quake_vector_types::{Alert, SensorId, SensorStatus, SeismicPrediction, StationId};

use crate::peer::{PeerHealth, PeerState};

/// Shared dashboard state updated periodically by the runtime.
pub struct DashboardState {
    pub station_id: StationId,
    pub boot_type: String,
    pub boot_duration_ms: u32,
    pub latest_predictions: VecDeque<SeismicPrediction>,
    pub latest_alerts: VecDeque<Alert>,
    pub ram_utilization: f32,
    pub degradation_level: String,
    pub uptime_secs: u64,
    pub graph_node_count: u64,
    pub sensor_statuses: Vec<(SensorId, SensorStatus)>,
    pub peer_statuses: Vec<PeerSnapshot>,
}

/// Serializable snapshot of a peer's state.
#[derive(Debug, Clone, Serialize)]
pub struct PeerSnapshot {
    pub id: u64,
    pub health: String,
    pub last_seen_us: u64,
}

impl PeerSnapshot {
    pub fn from_peer(peer: &PeerState) -> Self {
        let health = match peer.health {
            PeerHealth::Healthy => "healthy",
            PeerHealth::Stale => "stale",
            PeerHealth::Lost => "lost",
        };
        Self {
            id: peer.station_id.0,
            health: health.to_string(),
            last_seen_us: peer.last_seen_us,
        }
    }
}

impl DashboardState {
    pub fn new(station_id: StationId, boot_type: &str, boot_duration_ms: u32) -> Self {
        Self {
            station_id,
            boot_type: boot_type.to_string(),
            boot_duration_ms,
            latest_predictions: VecDeque::with_capacity(100),
            latest_alerts: VecDeque::with_capacity(100),
            ram_utilization: 0.0,
            degradation_level: "normal".to_string(),
            uptime_secs: 0,
            graph_node_count: 0,
            sensor_statuses: Vec::new(),
            peer_statuses: Vec::new(),
        }
    }

    /// Push a prediction, keeping at most 100.
    pub fn push_prediction(&mut self, pred: SeismicPrediction) {
        if self.latest_predictions.len() >= 100 {
            self.latest_predictions.pop_front();
        }
        self.latest_predictions.push_back(pred);
    }

    /// Push an alert, keeping at most 100.
    pub fn push_alert(&mut self, alert: Alert) {
        if self.latest_alerts.len() >= 100 {
            self.latest_alerts.pop_front();
        }
        self.latest_alerts.push_back(alert);
    }
}

// ---------------------------------------------------------------------------
// JSON response types
// ---------------------------------------------------------------------------

#[derive(Serialize)]
struct HealthResponse {
    station_id: u64,
    status: String,
    uptime_secs: u64,
    graph_node_count: u64,
    ram_utilization: f32,
    degradation_level: String,
    sensors: Vec<SensorInfo>,
    peers: Vec<PeerInfo>,
    boot_type: String,
    boot_duration_ms: u32,
}

#[derive(Serialize)]
struct SensorInfo {
    id: u64,
    status: String,
}

#[derive(Serialize)]
struct PeerInfo {
    id: u64,
    health: String,
}

#[derive(Deserialize)]
pub struct LimitQuery {
    pub limit: Option<usize>,
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

type SharedState = Arc<RwLock<DashboardState>>;

async fn health_handler(State(state): State<SharedState>) -> Json<serde_json::Value> {
    let s = state.read().await;
    let sensors: Vec<SensorInfo> = s
        .sensor_statuses
        .iter()
        .map(|(id, status)| SensorInfo {
            id: id.0,
            status: format!("{status:?}").to_lowercase(),
        })
        .collect();
    let peers: Vec<PeerInfo> = s
        .peer_statuses
        .iter()
        .map(|p| PeerInfo {
            id: p.id,
            health: p.health.clone(),
        })
        .collect();

    let resp = HealthResponse {
        station_id: s.station_id.0,
        status: "operational".to_string(),
        uptime_secs: s.uptime_secs,
        graph_node_count: s.graph_node_count,
        ram_utilization: s.ram_utilization,
        degradation_level: s.degradation_level.clone(),
        sensors,
        peers,
        boot_type: s.boot_type.clone(),
        boot_duration_ms: s.boot_duration_ms,
    };

    Json(serde_json::to_value(resp).unwrap_or_default())
}

async fn predictions_handler(
    State(state): State<SharedState>,
    Query(q): Query<LimitQuery>,
) -> Json<serde_json::Value> {
    let s = state.read().await;
    let limit = q.limit.unwrap_or(10).min(100);
    let preds: Vec<&SeismicPrediction> = s
        .latest_predictions
        .iter()
        .rev()
        .take(limit)
        .collect();
    Json(serde_json::to_value(preds).unwrap_or_default())
}

async fn alerts_handler(
    State(state): State<SharedState>,
    Query(q): Query<LimitQuery>,
) -> Json<serde_json::Value> {
    let s = state.read().await;
    let limit = q.limit.unwrap_or(10).min(100);
    let alerts: Vec<&Alert> = s
        .latest_alerts
        .iter()
        .rev()
        .take(limit)
        .collect();
    Json(serde_json::to_value(alerts).unwrap_or_default())
}

// ---------------------------------------------------------------------------
// Router & Service
// ---------------------------------------------------------------------------

/// Build the axum router with all dashboard endpoints.
pub fn build_router(state: SharedState) -> Router {
    Router::new()
        .route("/api/health", get(health_handler))
        .route("/api/predictions", get(predictions_handler))
        .route("/api/alerts", get(alerts_handler))
        .with_state(state)
}

/// Dashboard HTTP service.
pub struct DashboardService {
    bind_addr: String,
    state: SharedState,
}

impl DashboardService {
    pub fn new(bind_addr: &str, state: Arc<RwLock<DashboardState>>) -> Self {
        Self {
            bind_addr: bind_addr.to_string(),
            state,
        }
    }

    /// Spawn the axum server. Returns a JoinHandle.
    pub fn start(self) -> tokio::task::JoinHandle<()> {
        let router = build_router(self.state);
        let bind_addr = self.bind_addr.clone();
        tokio::spawn(async move {
            let listener = match tokio::net::TcpListener::bind(&bind_addr).await {
                Ok(l) => l,
                Err(e) => {
                    tracing::error!(error = %e, addr = %bind_addr, "failed to bind dashboard");
                    return;
                }
            };
            tracing::info!(addr = %bind_addr, "dashboard API started");
            if let Err(e) = axum::serve(listener, router).await {
                tracing::error!(error = %e, "dashboard server error");
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use tower::ServiceExt;

    fn make_state() -> SharedState {
        Arc::new(RwLock::new(DashboardState::new(
            StationId(1),
            "cold",
            150,
        )))
    }

    #[tokio::test]
    async fn dashboard_state_update_and_read() {
        let state = make_state();
        {
            let mut s = state.write().await;
            s.uptime_secs = 3600;
            s.graph_node_count = 50000;
            s.ram_utilization = 0.65;
            s.degradation_level = "normal".to_string();
            s.sensor_statuses.push((SensorId(1), SensorStatus::Active));
            s.push_prediction(SeismicPrediction {
                event_probability: 0.3,
                estimated_magnitude: 2.0,
                estimated_time_to_peak_s: 5.0,
                confidence: 0.8,
                contributing_wave_ids: vec![],
                model_version: 1,
                inference_latency_us: 50,
            });
        }

        let s = state.read().await;
        assert_eq!(s.uptime_secs, 3600);
        assert_eq!(s.graph_node_count, 50000);
        assert_eq!(s.latest_predictions.len(), 1);
    }

    #[tokio::test]
    async fn health_endpoint_returns_valid_json() {
        let state = make_state();
        {
            let mut s = state.write().await;
            s.uptime_secs = 100;
            s.graph_node_count = 42;
            s.ram_utilization = 0.55;
            s.sensor_statuses.push((SensorId(0), SensorStatus::Active));
        }

        let app = build_router(state);
        let req = Request::builder()
            .uri("/api/health")
            .body(Body::empty())
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body = axum::body::to_bytes(resp.into_body(), 1_000_000)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();

        assert_eq!(json["station_id"], 1);
        assert_eq!(json["status"], "operational");
        assert_eq!(json["uptime_secs"], 100);
        assert_eq!(json["graph_node_count"], 42);
        assert_eq!(json["boot_type"], "cold");
        assert_eq!(json["boot_duration_ms"], 150);
        assert!(json["sensors"].is_array());
    }

    #[tokio::test]
    async fn predictions_endpoint_respects_limit() {
        let state = make_state();
        {
            let mut s = state.write().await;
            for i in 0..20 {
                s.push_prediction(SeismicPrediction {
                    event_probability: i as f32 / 20.0,
                    estimated_magnitude: 3.0,
                    estimated_time_to_peak_s: 5.0,
                    confidence: 0.8,
                    contributing_wave_ids: vec![],
                    model_version: 1,
                    inference_latency_us: 50,
                });
            }
        }

        let app = build_router(state);
        let req = Request::builder()
            .uri("/api/predictions?limit=5")
            .body(Body::empty())
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body = axum::body::to_bytes(resp.into_body(), 1_000_000)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json.as_array().unwrap().len(), 5);
    }

    #[tokio::test]
    async fn alerts_endpoint_returns_empty_array() {
        let state = make_state();
        let app = build_router(state);
        let req = Request::builder()
            .uri("/api/alerts")
            .body(Body::empty())
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body = axum::body::to_bytes(resp.into_body(), 1_000_000)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert!(json.as_array().unwrap().is_empty());
    }

    #[tokio::test]
    async fn push_prediction_caps_at_100() {
        let state = make_state();
        {
            let mut s = state.write().await;
            for _ in 0..150 {
                s.push_prediction(SeismicPrediction {
                    event_probability: 0.5,
                    estimated_magnitude: 3.0,
                    estimated_time_to_peak_s: 5.0,
                    confidence: 0.8,
                    contributing_wave_ids: vec![],
                    model_version: 1,
                    inference_latency_us: 50,
                });
            }
        }
        let s = state.read().await;
        assert_eq!(s.latest_predictions.len(), 100);
    }
}
