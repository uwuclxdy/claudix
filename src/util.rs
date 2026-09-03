use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use crate::error::{ClaudixError, RecoveryHint, Result};
use crate::prompts::hints;

const SECONDS_PER_DAY: i64 = 86_400;

/// Root of every claudix path outside the project: the global config, the
/// bundled model cache, the install log.
///
/// A test build resolves it from `CLAUDIX_SANDBOX_HOME` and nothing else, so
/// no test can reach the developer's own `~/.claude`, and a spawned `claudix`
/// binary inherits the same sandbox. `.cargo/config.toml` sets that variable
/// for every process cargo starts. An unset variable aborts instead of falling
/// back, because falling back is how a suite ends up asserting against
/// whatever the machine running it happens to have configured.
///
/// The condition carries `test` as well as the feature: `cargo test --lib`
/// enables neither `--all-features` nor `test-stub`, and a feature-only guard
/// leaves those unit tests reading the real home. It matches the spelling every
/// other test-only path in this crate already uses.
#[cfg_attr(any(test, feature = "test-stub"), allow(clippy::panic))]
pub(crate) fn home_root() -> Option<PathBuf> {
    #[cfg(any(test, feature = "test-stub"))]
    {
        match std::env::var_os("CLAUDIX_SANDBOX_HOME") {
            Some(dir) => Some(PathBuf::from(dir)),
            None => panic!(
                "CLAUDIX_SANDBOX_HOME is unset in a test build: a test must never read real user paths. `.cargo/config.toml` sets it for every process cargo starts."
            ),
        }
    }
    #[cfg(not(any(test, feature = "test-stub")))]
    {
        dirs::home_dir()
    }
}

/// The global config file, resolved in one place so config loading and
/// `claudix install` can never disagree about where it lives.
pub(crate) fn global_config_path() -> Option<PathBuf> {
    home_root().map(|home| home.join(".claude").join("claudix.toml"))
}

pub fn now_rfc3339() -> String {
    format_rfc3339(SystemTime::now())
}

pub fn format_rfc3339(time: SystemTime) -> String {
    let duration = time
        .duration_since(UNIX_EPOCH)
        .unwrap_or_else(|_| Duration::from_secs(0));
    let total_seconds = i64::try_from(duration.as_secs()).unwrap_or(i64::MAX);
    let days = total_seconds.div_euclid(SECONDS_PER_DAY);
    let seconds_of_day = total_seconds.rem_euclid(SECONDS_PER_DAY);
    let (year, month, day) = civil_from_days(days);
    let hour = seconds_of_day / 3_600;
    let minute = (seconds_of_day % 3_600) / 60;
    let second = seconds_of_day % 60;

    format!("{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}Z")
}

pub fn parse_rfc3339(timestamp: &str) -> Result<SystemTime> {
    let bytes = timestamp.as_bytes();
    if bytes.len() != 20
        || bytes[4] != b'-'
        || bytes[7] != b'-'
        || bytes[10] != b'T'
        || bytes[13] != b':'
        || bytes[16] != b':'
        || bytes[19] != b'Z'
    {
        return Err(invalid_timestamp(timestamp));
    }

    let year = parse_number(&bytes[0..4], timestamp)?;
    let month = parse_number(&bytes[5..7], timestamp)?;
    let day = parse_number(&bytes[8..10], timestamp)?;
    let hour = parse_number(&bytes[11..13], timestamp)?;
    let minute = parse_number(&bytes[14..16], timestamp)?;
    let second = parse_number(&bytes[17..19], timestamp)?;

    if !(1..=12).contains(&month)
        || day < 1
        || day > days_in_month(year, month)
        || hour > 23
        || minute > 59
        || second > 59
    {
        return Err(invalid_timestamp(timestamp));
    }

    let days = days_from_civil(year, month, day).ok_or_else(|| invalid_timestamp(timestamp))?;
    let seconds = days
        .checked_mul(SECONDS_PER_DAY)
        .and_then(|value| value.checked_add(hour * 3_600 + minute * 60 + second))
        .ok_or_else(|| invalid_timestamp(timestamp))?;
    let seconds = u64::try_from(seconds).map_err(|_| invalid_timestamp(timestamp))?;

    Ok(UNIX_EPOCH + Duration::from_secs(seconds))
}

fn days_in_month(year: i64, month: i64) -> i64 {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if is_leap_year(year) => 29,
        2 => 28,
        _ => 0,
    }
}

fn is_leap_year(year: i64) -> bool {
    year % 4 == 0 && (year % 100 != 0 || year % 400 == 0)
}

fn parse_number(bytes: &[u8], original: &str) -> Result<i64> {
    let mut value = 0_i64;
    for byte in bytes {
        if !byte.is_ascii_digit() {
            return Err(invalid_timestamp(original));
        }
        value = value * 10 + i64::from(byte - b'0');
    }
    Ok(value)
}

fn invalid_timestamp(timestamp: &str) -> ClaudixError {
    ClaudixError::ConfigInvalid {
        message: format!("invalid RFC3339 timestamp: {timestamp}"),
        recovery: RecoveryHint(hints::UTC_TIMESTAMP_FORMAT),
    }
}

fn civil_from_days(days_since_unix_epoch: i64) -> (i64, i64, i64) {
    let z = days_since_unix_epoch + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let year = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = mp + if mp < 10 { 3 } else { -9 };
    let year = year + if month <= 2 { 1 } else { 0 };

    (year, month, day)
}

fn days_from_civil(year: i64, month: i64, day: i64) -> Option<i64> {
    let adjusted_year = year - if month <= 2 { 1 } else { 0 };
    let era = if adjusted_year >= 0 {
        adjusted_year
    } else {
        adjusted_year - 399
    } / 400;
    let yoe = adjusted_year - era * 400;
    let month_offset = month + if month > 2 { -3 } else { 9 };
    let doy = (153 * month_offset + 2) / 5 + day - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;

    era.checked_mul(146_097)
        .and_then(|value| value.checked_add(doe))
        .and_then(|value| value.checked_sub(719_468))
}

/// Last line starting with `error:` in a log file, if any. Scans from the end of a
/// typically-small log (tens of KB). Surfaces a recorded failure (background index,
/// binary download) without re-running the failed work. None when the log is missing
/// or records no error.
pub fn last_error_line(log_path: &Path) -> Option<String> {
    let text = fs::read_to_string(log_path).ok()?;
    text.lines()
        .rev()
        .find(|line| line.starts_with("error:"))
        .map(str::to_owned)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn last_error_line_returns_last_error_in_log() {
        let temp = tempfile::tempdir();
        assert!(temp.is_ok());
        let temp = temp.ok().unwrap_or_else(|| unreachable!());
        let log = temp.path().join("install.log");
        assert!(
            fs::write(
                &log,
                "downloading claudix-linux-x86_64 v0.1.6\nok progress\nerror: checksum mismatch\n",
            )
            .is_ok()
        );

        assert_eq!(
            last_error_line(&log),
            Some("error: checksum mismatch".to_owned())
        );
    }

    #[test]
    fn last_error_line_none_when_no_error_or_missing() {
        let temp = tempfile::tempdir();
        assert!(temp.is_ok());
        let temp = temp.ok().unwrap_or_else(|| unreachable!());
        let log = temp.path().join("install.log");
        assert!(fs::write(&log, "downloading...\ninstalled\n").is_ok());
        assert_eq!(last_error_line(&log), None);
        assert_eq!(last_error_line(&temp.path().join("missing.log")), None);
    }

    #[test]
    fn format_and_parse_round_trip() {
        let timestamp = UNIX_EPOCH + Duration::from_secs(1_746_057_600);
        let formatted = format_rfc3339(timestamp);
        assert_eq!(formatted, "2025-05-01T00:00:00Z");

        let parsed = parse_rfc3339(&formatted);
        assert!(parsed.is_ok());
        assert_eq!(parsed.ok().unwrap_or_else(|| unreachable!()), timestamp);
    }

    #[test]
    fn parse_rejects_invalid_shape() {
        let parsed = parse_rfc3339("2026-04-27 12:00:00");
        assert!(matches!(parsed, Err(ClaudixError::ConfigInvalid { .. })));
    }

    #[test]
    fn parse_rejects_invalid_month_days() {
        for timestamp in ["2026-02-29T00:00:00Z", "2026-04-31T00:00:00Z"] {
            let parsed = parse_rfc3339(timestamp);
            assert!(matches!(parsed, Err(ClaudixError::ConfigInvalid { .. })));
        }

        assert!(parse_rfc3339("2028-02-29T00:00:00Z").is_ok());
    }
}
