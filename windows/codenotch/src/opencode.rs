//! OpenCode usage adapter, ported from the Mac app's `OpenCodeProvider` / `OpenCodeUsage` /
//! `OpenCodeCredentials` (the ring upstream added in #11).
//!
//! Data path — the Go plan's official usage endpoint, with the key OpenCode itself stores on
//! sign-in, so nothing here asks anyone to sign in twice:
//!
//!   1. Credential, first hit wins. `%USERPROFILE%\.local\share\opencode\auth.json` holds the
//!      `opencode-go` entry OpenCode writes on `opencode auth login` — both shapes have shipped:
//!      the key as a bare string, or an object carrying it (key / apiKey / api_key / token /
//!      accessToken); any other entry (`openai`, `google`, …) is that vendor's key and is never
//!      claimed here. Failing that, `OPENCODE_GO_API_KEY` in the process environment, then the
//!      same variable in a Hermes env file (`%LOCALAPPDATA%\hermes\.env` and its
//!      `profiles\*\.env`), where the Hermes agent keeps the key when it drives OpenCode itself.
//!      Read into memory only — the value never reaches a log, an event or the UI.
//!
//!   2. Endpoint: `GET https://opencode.ai/zen/go/v1/usage` (Bearer, Accept: application/json,
//!      15 s). An explicit User-Agent goes with it: the host is behind Cloudflare, which answers
//!      some software user-agents with a 403 bot page (Error 1010) while the same key from an
//!      ordinary one is served — verified against this endpoint 2026-09-19. Reply:
//!      ```text
//!      {"usage":{"rolling":{"status":"ok","percent":0,"resetsAt":"2026-09-06T12:31:06.611Z"},
//!                "weekly":{…},"monthly":{…}}}
//!      ```
//!      `percent` is *used*, matching the dashboard's "X% used"; `resetsAt` carries milliseconds,
//!      which chrono reads as RFC 3339 either way. A window missing `percent` is skipped; no
//!      windows at all reads as a bad reply rather than an invented ring.
//!
//!   3. Upstream quirks, both from the Mac's comments: a valid key with no Go plan answers 401
//!      the same as a bad key, so 401 is "nothing readable here" (needsAuth); a plain 403 means
//!      the key is readable but not entitled to Go — that is `none`, not an error. A 403 whose
//!      body is Cloudflare's bot page is neither: it is the CDN refusing this client, reported as
//!      the transport failure it is. 429 backs off 60 s × 2^n capped at 15 minutes, Retry-After
//!      only raising it, and the deadline is persisted so a restart does not poll into the limit.

use crate::usage::{LimitWindow, UsageSnapshot};
use crate::AppState;
use std::path::PathBuf;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tauri::{AppHandle, Emitter, Manager};

const ENDPOINT: &str = "https://opencode.ai/zen/go/v1/usage";
const POLL_SECS: u64 = 300;
const KEY_ENV: &str = "OPENCODE_GO_API_KEY";
/// Cloudflare answers the bare library user-agents with a bot page (see the module doc).
const USER_AGENT: &str = concat!("Codenotch/", env!("CARGO_PKG_VERSION"), " (Windows)");
const BACKOFF_BASE_SECS: u64 = 60;
const BACKOFF_CAP_SECS: u64 = 900;

static REFRESH: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

pub fn request_refresh() {
    REFRESH.store(true, std::sync::atomic::Ordering::Relaxed);
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

fn store_path() -> PathBuf {
    crate::config::config_path().with_file_name("opencode.json")
}

pub fn load_persisted() -> UsageSnapshot {
    std::fs::read_to_string(store_path())
        .ok()
        .and_then(|t| serde_json::from_str::<UsageSnapshot>(&t).ok())
        .map(|mut s| {
            if !s.windows.is_empty() {
                s.status = "stale".into();
            }
            s
        })
        .unwrap_or_default()
}

fn persist(s: &UsageSnapshot) {
    if let Ok(t) = serde_json::to_string_pretty(s) {
        let _ = std::fs::write(store_path(), t);
    }
}

pub fn present() -> bool {
    load_credentials().is_some()
}

// ---------------- Credentials ----------------

/// Windows: %USERPROFILE%\.local\share\opencode\auth.json (macOS: ~/.local/share/opencode/auth.json)
pub fn auth_path() -> Option<PathBuf> {
    dirs::home_dir().map(|h| {
        h.join(".local")
            .join("share")
            .join("opencode")
            .join("auth.json")
    })
}

/// Where the key was found, for the note and doctor (never the key itself).
#[derive(Clone, Copy, PartialEq)]
pub enum Source {
    AuthJson,
    Env,
    HermesFile,
}

impl Source {
    fn note(self) -> &'static str {
        match self {
            Source::AuthJson => "Go plan · via OpenCode",
            Source::Env => "Go plan · via OPENCODE_GO_API_KEY",
            Source::HermesFile => "Go plan · via the Hermes env file",
        }
    }
    fn probe(self) -> &'static str {
        match self {
            Source::AuthJson => "OpenCode's own sign-in",
            Source::Env => "the OPENCODE_GO_API_KEY environment variable",
            Source::HermesFile => "a Hermes env file",
        }
    }
}

pub struct Creds {
    pub token: String,
    pub source: Source,
}

/// The `opencode-go` entry, in either shape that has shipped; every other entry is ignored.
fn key_from_auth_json(root: &serde_json::Value) -> Option<String> {
    let entry = root.get("opencode-go")?;
    let non_empty = |s: Option<&str>| s.filter(|v| !v.is_empty()).map(|v| v.to_string());
    if let Some(token) = non_empty(entry.as_str()) {
        return Some(token);
    }
    let object = entry.as_object()?;
    ["key", "apiKey", "api_key", "token", "accessToken"]
        .iter()
        .find_map(|k| non_empty(object.get(*k).and_then(|x| x.as_str())))
}

/// One `KEY=VALUE` line out of an env file. Only our own variable is read; comments and blank
/// lines are skipped, surrounding quotes dropped, `=` inside the value kept.
fn key_from_env_text(text: &str) -> Option<String> {
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let Some((name, value)) = line.split_once('=') else {
            continue;
        };
        if name.trim() != KEY_ENV {
            continue;
        }
        let value = value.trim().trim_matches(['"', '\'']);
        if !value.is_empty() {
            return Some(value.to_string());
        }
    }
    None
}

/// `%LOCALAPPDATA%\hermes\.env` first, then each `profiles\<name>\.env` alphabetically, then the
/// pre-AppData `~/.hermes/.env` — the layouts Hermes has used on Windows.
fn hermes_env_files() -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = Vec::new();
    if let Some(local) = dirs::data_local_dir() {
        let root = local.join("hermes");
        out.push(root.join(".env"));
        let mut profiles: Vec<PathBuf> = std::fs::read_dir(root.join("profiles"))
            .map(|rd| {
                rd.flatten()
                    .filter(|e| e.path().is_dir())
                    .map(|e| e.path().join(".env"))
                    .collect()
            })
            .unwrap_or_default();
        profiles.sort();
        out.extend(profiles);
    }
    if let Some(home) = dirs::home_dir() {
        out.push(home.join(".hermes").join(".env"));
    }
    out
}

/// Re-read every time: OpenCode rotates the key on sign-in, and holding an old value signs us out
fn load_credentials() -> Option<Creds> {
    if let Some(p) = auth_path() {
        if let Ok(text) = std::fs::read_to_string(&p) {
            if let Ok(root) = serde_json::from_str::<serde_json::Value>(&text) {
                if let Some(token) = key_from_auth_json(&root) {
                    return Some(Creds {
                        token,
                        source: Source::AuthJson,
                    });
                }
            }
        }
    }
    if let Ok(token) = std::env::var(KEY_ENV) {
        let token = token.trim().to_string();
        if !token.is_empty() {
            return Some(Creds {
                token,
                source: Source::Env,
            });
        }
    }
    for p in hermes_env_files() {
        if let Ok(text) = std::fs::read_to_string(&p) {
            if let Some(token) = key_from_env_text(&text) {
                return Some(Creds {
                    token,
                    source: Source::HermesFile,
                });
            }
        }
    }
    None
}

/// For doctor: contains no secret values
pub fn probe() -> String {
    match load_credentials() {
        Some(c) => format!(
            "OpenCode: key found in {} ({} chars)",
            c.source.probe(),
            c.token.len()
        ),
        None => format!(
            "OpenCode: no key — sign in with `opencode auth login` ({}) or set {KEY_ENV}",
            auth_path()
                .map(|p| p.display().to_string())
                .unwrap_or_else(|| "auth.json".into())
        ),
    }
}

// ---------------- Parsing ----------------

fn iso_ms(v: Option<&serde_json::Value>) -> Option<u64> {
    v.and_then(|x| x.as_str())
        .and_then(|s| chrono::DateTime::parse_from_rfc3339(s).ok())
        .map(|d| d.timestamp_millis().max(0) as u64)
}

/// The account's Go plan windows in headline order. The ids are the wire ids; the labels are the
/// Mac app's own wording (`OpenCodeUsage.windows`), so both platforms name the same window.
fn window_specs() -> [(&'static str, &'static str); 3] {
    [
        ("rolling", "5h limit"),
        ("weekly", "Weekly limit"),
        ("monthly", "Monthly limit"),
    ]
}

pub fn parse_usage(v: &serde_json::Value) -> Option<Vec<LimitWindow>> {
    let usage = v.get("usage")?;
    let mut out: Vec<LimitWindow> = Vec::new();
    for (id, label) in window_specs() {
        let Some(entry) = usage.get(id) else { continue };
        let Some(pct) = entry.get("percent").and_then(|x| x.as_f64()) else {
            continue;
        };
        out.push(LimitWindow {
            id: id.into(),
            label: label.into(),
            used: (pct / 100.0).clamp(0.0, 1.0),
            resets_at: iso_ms(entry.get("resetsAt")),
            ..Default::default()
        });
    }
    (!out.is_empty()).then_some(out)
}

// ---------------- Fetch ----------------

#[derive(Debug)]
enum FetchErr {
    NeedsAuth,
    /// Readable key, but not entitled to the Go plan — nothing to meter, not an error.
    NotEntitled,
    /// Suggested wait in seconds from Retry-After (before the floor is applied).
    RateLimited(Option<u64>),
    Other(String),
}

/// Cloudflare's bot page rather than OpenCode's own answer (see the module doc).
fn is_bot_page(body: &str) -> bool {
    body.contains("cloudflare") || body.contains("Error 1010") || body.contains("error_code")
}

fn classify_status(code: u16, body: &str) -> FetchErr {
    match code {
        401 => FetchErr::NeedsAuth,
        403 if is_bot_page(body) => FetchErr::Other(
            "OpenCode refused the request (CDN bot check) — the next poll retries".into(),
        ),
        403 => FetchErr::NotEntitled,
        other => FetchErr::Other(format!("HTTP {other}")),
    }
}

/// `Retry-After` is either a number of seconds or an HTTP date.
fn retry_after(resp: &ureq::Response) -> Option<u64> {
    let header = resp.header("retry-after")?.trim();
    if let Ok(seconds) = header.parse::<u64>() {
        return Some(seconds);
    }
    chrono::DateTime::parse_from_rfc2822(header)
        .ok()
        .map(|d| (d.timestamp() - chrono::Utc::now().timestamp()).max(0) as u64)
}

fn fetch_once(token: &str) -> Result<serde_json::Value, FetchErr> {
    let agent = ureq::AgentBuilder::new()
        .timeout(Duration::from_secs(15))
        .build();
    match agent
        .get(ENDPOINT)
        .set("Authorization", &format!("Bearer {token}"))
        .set("Accept", "application/json")
        .set("User-Agent", USER_AGENT)
        .call()
    {
        Ok(r) => r
            .into_json::<serde_json::Value>()
            .map_err(|e| FetchErr::Other(format!("parse: {e}"))),
        Err(ureq::Error::Status(429, r)) => Err(FetchErr::RateLimited(retry_after(&r))),
        Err(ureq::Error::Status(code, r)) => {
            let body = r.into_string().unwrap_or_default();
            Err(classify_status(code, &body))
        }
        Err(e) => Err(FetchErr::Other(format!("{e}"))),
    }
}

/// 60 s doubling per consecutive limit, capped so it always recovers on its own. The server's own
/// hint is honoured only as a floor-raiser (the Mac's rule: Retry-After only raises it).
fn backoff_secs(consecutive: u32, retry_after_floor: u64) -> u64 {
    let doubled = BACKOFF_BASE_SECS.saturating_mul(1u64 << consecutive.min(4));
    doubled
        .clamp(BACKOFF_BASE_SECS, BACKOFF_CAP_SECS)
        .max(retry_after_floor)
}

fn read_once(prev: &UsageSnapshot, consecutive_429: &mut u32) -> UsageSnapshot {
    let mut snap = prev.clone();
    let Some(creds) = load_credentials() else {
        snap.status = "needsAuth".into();
        snap.note = format!("Sign in with `opencode auth login`, or set {KEY_ENV}, to see usage.");
        return snap;
    };
    match fetch_once(&creds.token) {
        Ok(v) => match parse_usage(&v) {
            Some(windows) => {
                *consecutive_429 = 0;
                snap.status = "ok".into();
                snap.windows = windows;
                snap.fetched_at = now_ms();
                snap.note = creds.source.note().into();
                snap.backoff_until = 0;
            }
            None => {
                // An answer without windows is not a reading: keep the old one, and say so.
                snap.status = if snap.windows.is_empty() {
                    "error"
                } else {
                    "stale"
                }
                .into();
                snap.note = "OpenCode answered without usage windows".into();
            }
        },
        Err(FetchErr::NeedsAuth) => {
            snap.status = "needsAuth".into();
            snap.note = "OpenCode rejected the key — sign in again (`opencode auth login`) or update the key".into();
        }
        Err(FetchErr::NotEntitled) => {
            snap.status = "none".into();
            snap.windows.clear();
            snap.note = "No Go plan on this key — nothing to meter".into();
        }
        Err(FetchErr::RateLimited(ra)) => {
            let wait = backoff_secs(*consecutive_429, ra.unwrap_or(0));
            *consecutive_429 += 1;
            // The status is left alone: a refused refresh says nothing about the reading we are
            // holding. Age decides, as it does everywhere in this app.
            snap.note = format!("Rate limited, retrying in {wait}s");
            snap.backoff_until = now_ms() + wait * 1000;
        }
        Err(FetchErr::Other(msg)) => {
            // Stale beats invented: keep the old reading, marked stale
            snap.status = if snap.windows.is_empty() {
                "error"
            } else {
                "stale"
            }
            .into();
            snap.note = msg;
        }
    }
    snap
}

fn broadcast(app: &AppHandle, snap: UsageSnapshot) {
    let st = app.state::<AppState>();
    *st.opencode.lock().unwrap() = snap.clone();
    persist(&snap);
    let _ = app.emit("opencode", &snap);
}

fn sleep_interruptible(secs: u64) {
    for _ in 0..secs {
        if REFRESH.swap(false, std::sync::atomic::Ordering::Relaxed) {
            return;
        }
        std::thread::sleep(Duration::from_secs(1));
    }
}

pub fn start(app: AppHandle) {
    std::thread::spawn(move || {
        {
            let st = app.state::<AppState>();
            let snap = st.opencode.lock().unwrap().clone();
            let _ = app.emit("opencode", &snap);
        }
        if !present() {
            broadcast(
                &app,
                UsageSnapshot {
                    status: "absent".into(),
                    ..Default::default()
                },
            );
            loop {
                sleep_interruptible(600); // no key anywhere: look again every 10 minutes
                if present() {
                    break;
                }
            }
        }
        let mut consecutive_429: u32 = 0;
        loop {
            // No requests inside the backoff window
            let bu = {
                let st = app.state::<AppState>();
                let u = st.opencode.lock().unwrap();
                u.backoff_until
            };
            let now = now_ms();
            if bu > now {
                sleep_interruptible(((bu - now) / 1000).clamp(1, 30));
                continue;
            }
            let prev = {
                let st = app.state::<AppState>();
                let s = st.opencode.lock().unwrap().clone();
                s
            };
            let snap = read_once(&prev, &mut consecutive_429);
            if snap.status == "error" || snap.status == "stale" {
                crate::applog(&format!("opencode: {}", snap.note));
            }
            broadcast(&app, snap);
            sleep_interruptible(POLL_SECS);
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = r#"{"usage":{
        "rolling":{"status":"ok","percent":0,"resetsAt":"2026-09-06T12:31:06.611Z"},
        "weekly":{"status":"ok","percent":17.5,"resetsAt":"2026-09-07T00:00:00.611Z"},
        "monthly":{"status":"ok","percent":100,"resetsAt":"2026-10-03T13:09:45.611Z"}}}"#;

    #[test]
    fn the_three_go_windows_are_read_in_headline_order() {
        let v: serde_json::Value = serde_json::from_str(SAMPLE).unwrap();
        let w = parse_usage(&v).expect("three windows");
        assert_eq!(w.len(), 3);
        assert_eq!(w[0].id, "rolling");
        assert_eq!(w[0].label, "5h limit");
        assert_eq!(w[1].id, "weekly");
        assert_eq!(w[2].id, "monthly");
        assert!((w[0].used - 0.0).abs() < 1e-9, "0 is a reading");
        assert!((w[1].used - 0.175).abs() < 1e-9);
        assert!((w[2].used - 1.0).abs() < 1e-9);
        assert!(w[0].resets_at.is_some(), "milliseconds are accepted");
    }

    #[test]
    fn a_window_without_a_percentage_is_skipped_and_nothing_parses_as_none() {
        let partial: serde_json::Value =
            serde_json::from_str(r#"{"usage":{"rolling":{"status":"ok","percent":1}}}"#).unwrap();
        let w = parse_usage(&partial).expect("one window");
        assert_eq!(w.len(), 1);
        assert_eq!(w[0].id, "rolling");
        assert!(
            w[0].resets_at.is_none(),
            "a window with no reset time still shows"
        );

        let none: serde_json::Value = serde_json::from_str(r#"{"usage":{}}"#).unwrap();
        assert!(parse_usage(&none).is_none());
        let what: serde_json::Value = serde_json::from_str(r#"{"nothing":true}"#).unwrap();
        assert!(parse_usage(&what).is_none());
    }

    #[test]
    fn the_go_key_is_taken_in_either_shape_and_only_from_its_own_entry() {
        let as_string: serde_json::Value =
            serde_json::from_str(r#"{"opencode-go":"sk-go-1","openai":"sk-openai"}"#).unwrap();
        assert_eq!(key_from_auth_json(&as_string).as_deref(), Some("sk-go-1"));

        let as_object: serde_json::Value = serde_json::from_str(
            r#"{"opencode-go":{"type":"api","key":"sk-go-2"},"google":{"key":"google-key"}}"#,
        )
        .unwrap();
        assert_eq!(key_from_auth_json(&as_object).as_deref(), Some("sk-go-2"));

        let alt: serde_json::Value =
            serde_json::from_str(r#"{"opencode-go":{"accessToken":"sk-go-3"}}"#).unwrap();
        assert_eq!(key_from_auth_json(&alt).as_deref(), Some("sk-go-3"));

        let empty: serde_json::Value = serde_json::from_str(r#"{"opencode-go":""}"#).unwrap();
        assert!(
            key_from_auth_json(&empty).is_none(),
            "an empty key is worse than a missing one"
        );

        let other: serde_json::Value = serde_json::from_str(r#"{"openai":"sk-openai"}"#).unwrap();
        assert!(key_from_auth_json(&other).is_none());
    }

    #[test]
    fn the_env_file_line_is_read_with_quotes_comments_and_equals_kept() {
        let text = "# Hermes env\nTERMINAL_TIMEOUT=120\nOPENCODE_GO_API_KEY=\"sk-go=abc\"\n";
        assert_eq!(key_from_env_text(text).as_deref(), Some("sk-go=abc"));
        assert!(
            key_from_env_text("OPENCODE_API_KEY=other\n").is_none(),
            "the Zen key is not the Go key"
        );
        assert!(key_from_env_text("OPENCODE_GO_API_KEY=\n").is_none());
        assert!(
            key_from_env_text("OPENCODE_GO_API_KEY='/quoted/'\n").as_deref() == Some("/quoted/")
        );
    }

    #[test]
    fn a_cloudflare_bot_page_is_not_read_as_no_go_plan() {
        let bot = r#"{"type":"https://developers.cloudflare.com/support/troubleshooting/http-status-codes/cloudflare-1xxx-errors/error-1010/","title":"Error 1010: Access denied","status":403,"error_code":1010}"#;
        assert!(is_bot_page(bot));
        assert!(matches!(classify_status(403, bot), FetchErr::Other(_)));

        let plain = r#"{"error":"no go plan on this key"}"#;
        assert!(!is_bot_page(plain));
        assert!(matches!(classify_status(403, plain), FetchErr::NotEntitled));
        assert!(matches!(classify_status(401, ""), FetchErr::NeedsAuth));
        assert!(matches!(classify_status(500, ""), FetchErr::Other(_)));
    }

    #[test]
    fn the_backoff_starts_at_a_minute_doubles_and_caps() {
        assert_eq!(backoff_secs(0, 0), 60);
        assert_eq!(backoff_secs(1, 0), 120);
        assert_eq!(backoff_secs(2, 0), 240);
        assert_eq!(backoff_secs(4, 0), BACKOFF_CAP_SECS);
        assert_eq!(backoff_secs(9, 0), BACKOFF_CAP_SECS, "never unbounded");
        assert_eq!(backoff_secs(0, 3600), 3600, "Retry-After raises the floor");
    }

    /// The real endpoint with whatever key this machine holds — the check that matters for the
    /// Cloudflare user-agent question, which no fixture can answer. Opt in:
    /// `cargo test live_go_endpoint -- --ignored --nocapture`
    #[test]
    #[ignore = "Hits the live OpenCode endpoint with this machine's key; opt in"]
    fn live_go_endpoint_answers_with_windows() {
        let creds = load_credentials().expect("an OpenCode Go key on this machine");
        let v = fetch_once(&creds.token).expect("the endpoint answered this client");
        let windows = parse_usage(&v).expect("Go windows in the reply");
        assert_eq!(windows[0].id, "rolling");
        assert!(windows[0].resets_at.is_some());
        eprintln!(
            "opencode live: {} window(s), headline {:.1}% used, via {}",
            windows.len(),
            windows[0].used * 100.0,
            creds.source.probe()
        );
    }
}
