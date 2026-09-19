//! Hermes usage adapter — the tokens the Hermes agent itself recorded, read out of the databases
//! it keeps on this machine.
//!
//! This cell exists because Hermes is the tool that drives the other subscriptions here: what it
//! spends has no vendor endpoint of its own, and Hermes's own record is the only place the number
//! exists. The Mac app reads the same table for its Hermes row (in `HermesGeminiUsage`); this port
//! reads it across every Hermes profile rather than one, because on Windows the agent keeps one
//! database per profile.
//!
//!   1. Sources, all read-only, summed together: `%LOCALAPPDATA%\hermes\state.db`,
//!      `%LOCALAPPDATA%\hermes\profiles\<name>\state.db` and `%USERPROFILE%\.hermes\state.db`
//!      (the pre-AppData location).
//!
//!   2. Table: `session_model_usage`, one row per session and model. Tokens are
//!      `input + cache_read + cache_write + output`. `reasoning_tokens` is left out on purpose,
//!      the Mac's rule: Hermes's own totals count reasoning *inside* `output_tokens`, so adding it
//!      back would bill every thinking model twice.
//!
//!   3. A row is an aggregate over a whole session, so it cannot be split across midnight or the
//!      month's edge: it lands, whole, in the bucket of its `last_seen` — the same granularity
//!      Hermes's own `/usage` report has (its `TAGS.md`-era comment, carried over).
//!
//!   4. Buckets are local month and local day, not UTC: the user's "today" is local, and the two
//!      windows have to agree on where the boundary is.
//!
//! Nothing here is a limit: Hermes publishes no allowance, so the cell is a counted reading
//! (`count` + `~`), not a percentage — like Antigravity's request count and the Mac's derived
//! Gemini figures. Read only, never written; no key, network or prompt is involved.

use crate::usage::{LimitWindow, UsageSnapshot};
use crate::AppState;
use chrono::{DateTime, Datelike, Local, NaiveDate, TimeZone};
use std::path::PathBuf;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tauri::{AppHandle, Emitter, Manager};

const POLL_SECS: u64 = 300;
/// How many model rows the card carries; the rest would push its title off the top.
const MODEL_ROWS: usize = 5;

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
    crate::config::config_path().with_file_name("hermes.json")
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

/// The profile databases under %LOCALAPPDATA%\hermes, plus the pre-AppData home location.
pub fn db_paths() -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = Vec::new();
    if let Some(local) = dirs::data_local_dir() {
        let root = local.join("hermes");
        out.push(root.join("state.db"));
        let mut profiles: Vec<PathBuf> = std::fs::read_dir(root.join("profiles"))
            .map(|rd| {
                rd.flatten()
                    .filter(|e| e.path().is_dir())
                    .map(|e| e.path().join("state.db"))
                    .collect()
            })
            .unwrap_or_default();
        profiles.sort();
        out.extend(profiles);
    }
    if let Some(home) = dirs::home_dir() {
        out.push(home.join(".hermes").join("state.db"));
    }
    out.into_iter().filter(|p| p.is_file()).collect()
}

pub fn present() -> bool {
    !db_paths().is_empty()
}

// ---------------- SQLite, read only ----------------

/// mode=ro first (it sees what a running Hermes writes through the WAL), then immutable=1 (once
/// the agent has exited and the -shm is gone, mode=ro can fail to open; by then the WAL has been
/// checkpointed, so ignoring it costs nothing) — the same bargain cursor.rs strikes.
fn open_ro(path: &std::path::Path) -> Option<rusqlite::Connection> {
    use rusqlite::OpenFlags;
    let probe = |c: &rusqlite::Connection| {
        c.prepare("SELECT 1 FROM session_model_usage LIMIT 1")
            .and_then(|mut s| s.query([]).map(|_| ()))
            .is_ok()
    };
    if let Ok(c) = rusqlite::Connection::open_with_flags(
        path,
        OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX,
    ) {
        if probe(&c) {
            return Some(c);
        }
    }
    // Only the URI form takes immutable=1; a Windows path becomes file:///C:/... with \ → /
    let mut uri = String::from("file:///");
    uri.push_str(
        &path
            .to_string_lossy()
            .replace('\\', "/")
            .trim_start_matches('/')
            .replace('#', "%23")
            .replace('?', "%3F"),
    );
    uri.push_str("?immutable=1");
    rusqlite::Connection::open_with_flags(
        &uri,
        OpenFlags::SQLITE_OPEN_READ_ONLY
            | OpenFlags::SQLITE_OPEN_URI
            | OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )
    .ok()
    .filter(probe)
}

#[derive(Debug, Clone)]
pub struct Row {
    /// epoch seconds
    pub last_seen: i64,
    pub model: String,
    /// input + cache_read + cache_write + output (reasoning is inside output — see the module doc)
    pub tokens: i64,
}

/// This month's rows only: the cut-off is pushed into SQL so a years-old database is not paged
/// through on every poll (the column is indexed).
fn query_rows(conn: &rusqlite::Connection, month_start: i64) -> Option<Vec<Row>> {
    let mut stmt = conn
        .prepare(
            "SELECT last_seen, model,
                    COALESCE(input_tokens, 0) + COALESCE(cache_read_tokens, 0)
                  + COALESCE(cache_write_tokens, 0) + COALESCE(output_tokens, 0)
             FROM session_model_usage
             WHERE last_seen >= ?1",
        )
        .ok()?;
    let rows = stmt
        .query_map([month_start], |r| {
            Ok(Row {
                last_seen: r.get::<_, f64>(0)? as i64,
                model: r.get::<_, Option<String>>(1)?.unwrap_or_default(),
                tokens: r.get::<_, Option<i64>>(2)?.unwrap_or(0),
            })
        })
        .ok()?
        .flatten()
        .collect::<Vec<Row>>();
    Some(rows)
}

// ---------------- Buckets ----------------

fn start_of_month(now: DateTime<Local>) -> i64 {
    NaiveDate::from_ymd_opt(now.year(), now.month(), 1)
        .and_then(|d| d.and_hms_opt(0, 0, 0))
        .and_then(|naive| Local.from_local_datetime(&naive).earliest())
        .map(|d| d.timestamp())
        .unwrap_or(0)
}

fn end_of_month_ms(now: DateTime<Local>) -> Option<u64> {
    let (year, month) = if now.month() == 12 {
        (now.year() + 1, 1)
    } else {
        (now.year(), now.month() + 1)
    };
    NaiveDate::from_ymd_opt(year, month, 1)
        .and_then(|d| d.and_hms_opt(0, 0, 0))
        .and_then(|naive| Local.from_local_datetime(&naive).earliest())
        .map(|d| (d.timestamp_millis().max(0)) as u64)
}

fn start_of_day(now: DateTime<Local>) -> i64 {
    now.date_naive()
        .and_hms_opt(0, 0, 0)
        .and_then(|naive| Local.from_local_datetime(&naive).earliest())
        .map(|d| d.timestamp())
        .unwrap_or(0)
}

#[derive(Debug, Clone, Copy, Default)]
pub struct Totals {
    pub month: i64,
    pub today: i64,
}

/// One pass over the rows: the month total, the day bucket, and the per-model month sums. A row
/// belongs to exactly one day and one month (its `last_seen` decides both), so nothing is counted
/// twice and nothing is split.
pub fn totals_and_models(rows: &[Row], day_start: i64) -> (Totals, Vec<(String, i64)>) {
    let mut totals = Totals::default();
    let mut models: std::collections::HashMap<String, i64> = std::collections::HashMap::new();
    for row in rows {
        let tokens = row.tokens.max(0);
        totals.month += tokens;
        if row.last_seen >= day_start {
            totals.today += tokens;
        }
        if !row.model.is_empty() {
            *models.entry(row.model.clone()).or_insert(0) += tokens;
        }
    }
    let mut model_rows: Vec<(String, i64)> = models.into_iter().collect();
    // Busiest first; ties by name so the order cannot flicker between polls
    model_rows.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    model_rows.truncate(MODEL_ROWS);
    (totals, model_rows)
}

pub fn windows_from(
    totals: Totals,
    models: &[(String, i64)],
    now: DateTime<Local>,
) -> Vec<LimitWindow> {
    let counted = |id: String, label: String, count: i64, resets_at: Option<u64>| LimitWindow {
        id,
        label,
        used: 0.0,
        resets_at,
        count: Some(count),
        derived: true,
        group: None,
        unit: Some("tokens".into()),
    };
    let mut out = vec![
        counted(
            "month".into(),
            "Tokens this month".into(),
            totals.month,
            end_of_month_ms(now),
        ),
        counted("today".into(), "Tokens today".into(), totals.today, None),
    ];
    out.extend(models.iter().map(|(model, tokens)| {
        counted(
            format!("model:{model}"),
            format!("{model} · this month"),
            *tokens,
            None,
        )
    }));
    out
}

// ---------------- Read ----------------

/// Every readable database, summed. `None` only when a database exists but none can be read
/// (a Hermes too old to carry the table, or a locked file) — a database that answers with zero
/// rows is a real zero and is shown.
pub struct Readout {
    pub rows: Vec<Row>,
    pub dbs: usize,
    pub files: usize,
}

pub fn read_all() -> Readout {
    let paths = db_paths();
    let month_start = start_of_month(Local::now());
    let mut rows: Vec<Row> = Vec::new();
    let mut dbs = 0usize;
    for p in &paths {
        if let Some(conn) = open_ro(p) {
            if let Some(mut r) = query_rows(&conn, month_start) {
                dbs += 1;
                rows.append(&mut r);
            }
        }
    }
    Readout {
        rows,
        dbs,
        files: paths.len(),
    }
}

/// For doctor: contains no secret values
pub fn probe() -> String {
    let paths = db_paths();
    if paths.is_empty() {
        return "Hermes: no state.db found under %LOCALAPPDATA%\\hermes (the agent has not run here)".into();
    }
    let readout = read_all();
    let now = Local::now();
    let (totals, _) = totals_and_models(&readout.rows, start_of_day(now));
    match readout.dbs {
        0 => format!(
            "Hermes: {} database(s) found but none could be read (a pre-usage-table Hermes, or a locked file)",
            readout.files
        ),
        n => format!(
            "Hermes: {n} of {} database(s) readable, {} row(s) this month, ~{} tokens",
            readout.files,
            readout.rows.len(),
            totals.month
        ),
    }
}

fn read_once(prev: &UsageSnapshot) -> UsageSnapshot {
    let mut snap = prev.clone();
    let readout = read_all();
    if readout.files == 0 {
        snap.status = "absent".into();
        return snap;
    }
    if readout.dbs == 0 {
        snap.status = "none".into();
        snap.windows.clear();
        snap.note =
            "Hermes's usage table could not be read (older version, or the file is locked)".into();
        return snap;
    }
    let now = Local::now();
    let (totals, models) = totals_and_models(&readout.rows, start_of_day(now));
    snap.status = "ok".into();
    snap.windows = windows_from(totals, &models, now);
    snap.fetched_at = now_ms();
    snap.note = format!(
        "Hermes's own records · {} database{}",
        readout.dbs,
        if readout.dbs == 1 { "" } else { "s" }
    );
    snap.backoff_until = 0;
    snap
}

fn broadcast(app: &AppHandle, snap: UsageSnapshot) {
    let st = app.state::<AppState>();
    *st.hermes.lock().unwrap() = snap.clone();
    persist(&snap);
    let _ = app.emit("hermes", &snap);
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
            let snap = st.hermes.lock().unwrap().clone();
            let _ = app.emit("hermes", &snap);
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
                sleep_interruptible(600); // Hermes is not installed: look again every 10 minutes
                if present() {
                    break;
                }
            }
        }
        loop {
            let prev = {
                let st = app.state::<AppState>();
                let s = st.hermes.lock().unwrap().clone();
                s
            };
            let snap = read_once(&prev);
            if snap.status == "error" {
                crate::applog(&format!("hermes: {}", snap.note));
            }
            broadcast(&app, snap);
            sleep_interruptible(POLL_SECS);
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(last_seen: i64, model: &str, tokens: i64) -> Row {
        Row {
            last_seen,
            model: model.into(),
            tokens,
        }
    }

    /// The table as Hermes writes it, in memory: the query must add the four components and not
    /// the reasoning column (Hermes counts reasoning inside output_tokens).
    #[test]
    fn the_query_sums_the_four_components_and_never_reasoning() {
        let conn = rusqlite::Connection::open_in_memory().unwrap();
        conn.execute_batch(
            "CREATE TABLE session_model_usage(
                session_id TEXT, model TEXT, billing_provider TEXT, billing_base_url TEXT,
                billing_mode TEXT, task TEXT, api_call_count INTEGER,
                input_tokens INTEGER, output_tokens INTEGER, cache_read_tokens INTEGER,
                cache_write_tokens INTEGER, reasoning_tokens INTEGER,
                estimated_cost_usd REAL, actual_cost_usd REAL, cost_status TEXT,
                cost_source TEXT, first_seen REAL, last_seen REAL);
             INSERT INTO session_model_usage(session_id, model, last_seen,
                input_tokens, output_tokens, cache_read_tokens, cache_write_tokens, reasoning_tokens)
             VALUES ('s1', 'omen-alpha', 2000, 100, 50, 10, 5, 9999),
                    ('s2', 'omen-alpha', 500, 1, 2, 3, 4, 777),
                    ('s3', 'mimo-v2.5', 1500, 8, 8, 0, 0, 0);",
        )
        .unwrap();
        let rows = query_rows(&conn, 1000).expect("the table reads");
        assert_eq!(
            rows.len(),
            2,
            "the row before the cut-off is dropped in SQL"
        );
        assert!(rows.iter().all(|r| r.last_seen >= 1000));
        let fresh = rows
            .iter()
            .find(|r| r.last_seen == 2000)
            .expect("the fresh row");
        assert_eq!(
            fresh.tokens, 165,
            "100 + 50 + 10 + 5 (reasoning stays out: it is inside output_tokens)"
        );
        assert_eq!(fresh.model, "omen-alpha");
    }

    #[test]
    fn a_row_lands_whole_in_the_bucket_of_its_last_seen() {
        let day_start = 1_000i64;
        let rows = vec![
            row(999, "a", 100),  // yesterday: month yes, today no
            row(1_500, "a", 10), // today
            row(2_000, "b", 5),  // today
        ];
        let (totals, models) = totals_and_models(&rows, day_start);
        assert_eq!(totals.month, 115);
        assert_eq!(totals.today, 15);
        assert_eq!(models, vec![("a".to_string(), 110), ("b".to_string(), 5)]);
    }

    #[test]
    fn models_sort_busiest_first_ties_by_name_and_the_card_keeps_five() {
        // Six models; "a" and "e" tie at 100 and the tie breaks by name, so the order cannot
        // flicker between polls. "f" is the one the card's five rows leave out.
        let mut rows = vec![
            row(10, "e", 100),
            row(10, "d", 99),
            row(10, "c", 98),
            row(10, "b", 97),
            row(10, "a", 93),
            row(10, "f", 1),
        ];
        rows.push(row(10, "a", 7));
        let (_, models) = totals_and_models(&rows, 0);
        assert_eq!(models.len(), MODEL_ROWS);
        assert_eq!(models[0].0, "a");
        assert_eq!(models[0].1, 100);
        assert_eq!(models[1].0, "e");
        assert_eq!(models[1].1, 100);
        assert_eq!(models[4].0, "b");
    }

    #[test]
    fn empty_model_names_count_towards_totals_but_get_no_row() {
        let rows = vec![row(10, "", 42), row(10, "m", 8)];
        let (totals, models) = totals_and_models(&rows, 0);
        assert_eq!(totals.month, 50);
        assert_eq!(models, vec![("m".to_string(), 8)]);
    }

    #[test]
    fn the_month_window_is_a_count_with_tokens_as_its_unit_and_a_reset_at_month_end() {
        let now = Local.with_ymd_and_hms(2026, 9, 19, 12, 0, 0).unwrap();
        let totals = Totals {
            month: 31_408_225,
            today: 1_203_004,
        };
        let models = vec![("omen-alpha".to_string(), 15_000_000)];
        let w = windows_from(totals, &models, now);
        assert_eq!(w[0].id, "month");
        assert_eq!(w[0].count, Some(31_408_225));
        assert_eq!(w[0].unit.as_deref(), Some("tokens"));
        assert!(w[0].derived, "the number is assembled here, not a vendor's");
        let resets = w[0].resets_at.expect("the month ends");
        let next_month = Local
            .with_ymd_and_hms(2026, 10, 1, 0, 0, 0)
            .earliest()
            .expect("a next month");
        assert_eq!(resets / 1000, next_month.timestamp() as u64);
        assert_eq!(w[1].id, "today");
        assert_eq!(w[1].count, Some(1_203_004));
        assert_eq!(w[2].id, "model:omen-alpha");
        assert_eq!(w[2].label, "omen-alpha · this month");
    }

    #[test]
    fn local_day_and_month_starts_bracket_now() {
        let now = Local.with_ymd_and_hms(2026, 9, 19, 12, 0, 0).unwrap();
        let day = start_of_day(now);
        let month = start_of_month(now);
        assert!(day <= now.timestamp() && now.timestamp() < day + 86_400);
        assert!(month <= now.timestamp());
        let diff = now.timestamp() - month;
        assert!(
            (18 * 86_400..19 * 86_400).contains(&diff),
            "19 Sep is the 19th day of the month"
        );
    }
}
