pub(super) fn extract_search_command(command: Option<&str>) -> Option<String> {
    let command = command?.trim();
    let tokens = shell_words::split(command).ok()?;
    let mut idx = strip_command_wrappers(&tokens);
    let tool = tokens.get(idx)?.as_str();
    idx += 1;
    match tool {
        "rg" | "ripgrep" | "grep" | "ag" | "ack" => {}
        "git" => {
            if tokens.get(idx).map(String::as_str) != Some("grep") {
                return None;
            }
            idx += 1;
        }
        _ => return None,
    }
    extract_pattern_from_args(&tokens[idx..])
}

/// Skip `time`/`nice`/`stdbuf -oL`/`env KEY=VAL` wrappers in token form so the
/// search-tool detector can still see the real command. Returns the index of
/// the first non-wrapper token (or `tokens.len()` if the whole stream is
/// wrappers — caller will treat that as "no command").
fn strip_command_wrappers(tokens: &[String]) -> usize {
    let mut idx = 0;
    while idx < tokens.len() {
        match tokens[idx].as_str() {
            "time" | "nice" => idx += 1,
            "stdbuf" => {
                idx += 1;
                // stdbuf's flags are -iL/-oL/-eL or --input=L style — all start
                // with `-`. Consume any number of them so `stdbuf -oL -eL rg foo`
                // still resolves to `rg`, but bail on a non-flag token so a
                // malformed `stdbuf rg foo` doesn't swallow the actual tool.
                while idx < tokens.len() && tokens[idx].starts_with('-') {
                    idx += 1;
                }
            }
            "env" => {
                idx += 1;
                while let Some(token) = tokens.get(idx) {
                    if token.contains('=') && !token.starts_with('=') {
                        idx += 1;
                    } else {
                        break;
                    }
                }
            }
            _ => break,
        }
    }
    idx
}

/// Walk the rg/grep arg list and decide whether the call is a candidate for
/// semantic-search interception. Returns the pattern only when the command is
/// unambiguous: exactly one pattern, no output-mode flags, no positional path
/// scope. Anything else returns `None`, which means passthrough.
///
/// `Grep` already gates output-mode flags via `grep_input_has_scoping_flag`;
/// this is the equivalent for Bash invocations.
fn extract_pattern_from_args(args: &[String]) -> Option<String> {
    // Long flags whose mere presence breaks semantic-search substitution
    // (counts, file lists, JSON, invert, context lines, output transforms).
    const LONG_DISRUPTIVE: &[&str] = &[
        "files-with-matches",
        "files-without-match",
        "files",
        "count",
        "count-matches",
        "json",
        "only-matching",
        "invert-match",
        "passthru",
        "passthrough",
        "quiet",
        "after-context",
        "before-context",
        "context",
        "vimgrep",
        "no-filename",
        "with-filename",
        "stats",
        "replace",
        "file",
    ];
    // Long flags that take a value but don't change semantics — skip the value.
    const LONG_VALUE_NEUTRAL: &[&str] = &[
        "type",
        "type-not",
        "type-add",
        "type-clear",
        "glob",
        "iglob",
        "include",
        "exclude",
        "include-dir",
        "exclude-dir",
        "max-count",
        "max-depth",
        "max-filesize",
        "color",
        "colors",
        "engine",
        "encoding",
        "threads",
        "sort",
        "sortr",
        "pre",
        "pre-glob",
        "context-separator",
        "field-context-separator",
        "field-match-separator",
        "dfa-size-limit",
        "regex-size-limit",
    ];
    // Long flags whose value is itself a pattern (`-e/--regex PATTERN`).
    const LONG_PATTERN: &[&str] = &["regex", "regexp"];
    // Short single-character flags that are disruptive on their own or in a bundle.
    const SHORT_DISRUPTIVE: &[char] = &['l', 'L', 'c', 'v', 'o', 'q', 'A', 'B', 'C', 'f'];
    // Short value-taking neutral flags (consume the next token unless glued).
    const SHORT_VALUE_NEUTRAL: &[char] = &['t', 'T', 'g', 'm'];

    let mut explicit_patterns: Vec<String> = Vec::new();
    let mut positionals: Vec<String> = Vec::new();
    let mut idx = 0;

    while idx < args.len() {
        let arg = args[idx].as_str();

        if arg == "--" {
            for tail in &args[idx + 1..] {
                positionals.push(tail.clone());
            }
            break;
        }

        if let Some(rest) = arg.strip_prefix("--") {
            if rest.is_empty() {
                idx += 1;
                continue;
            }
            let (name, embedded) = match rest.split_once('=') {
                Some((n, v)) => (n, Some(v.to_owned())),
                None => (rest, None),
            };
            if LONG_DISRUPTIVE.contains(&name) {
                return None;
            }
            if LONG_PATTERN.contains(&name) {
                let value = match embedded {
                    Some(v) => v,
                    None => {
                        idx += 1;
                        args.get(idx).cloned()?
                    }
                };
                explicit_patterns.push(value);
                idx += 1;
                continue;
            }
            if LONG_VALUE_NEUTRAL.contains(&name) {
                idx += if embedded.is_some() { 1 } else { 2 };
                continue;
            }
            // Unknown long flag — assume valueless and move on.
            idx += 1;
            continue;
        }

        if let Some(rest) = arg.strip_prefix('-') {
            if rest.is_empty() {
                // Bare `-` is a positional (stdin in grep).
                positionals.push(arg.to_owned());
                idx += 1;
                continue;
            }
            if rest.chars().any(|c| SHORT_DISRUPTIVE.contains(&c)) {
                return None;
            }
            if let Some(stripped) = rest.strip_prefix('e') {
                if stripped.is_empty() {
                    idx += 1;
                    explicit_patterns.push(args.get(idx).cloned()?);
                    idx += 1;
                } else {
                    explicit_patterns.push(stripped.to_owned());
                    idx += 1;
                }
                continue;
            }
            if rest.len() == 1
                && let Some(ch) = rest.chars().next()
                && SHORT_VALUE_NEUTRAL.contains(&ch)
            {
                idx += 2;
                continue;
            }
            idx += 1;
            continue;
        }

        positionals.push(arg.to_owned());
        idx += 1;
    }

    match (explicit_patterns.len(), positionals.len()) {
        (1, 0) => explicit_patterns.into_iter().next(),
        (0, 1) => positionals.into_iter().next(),
        _ => None,
    }
}

pub(super) fn should_passthrough(query: &str) -> bool {
    let trimmed = query.trim();
    trimmed.is_empty()
        || looks_like_regex(trimmed)
        || looks_like_file_target(trimmed)
        || token_count(trimmed) < 3
}

fn looks_like_regex(query: &str) -> bool {
    query.contains('^')
        || query.contains('$')
        || query.contains('\\')
        || query.contains('[')
        || query.contains(']')
        || query.contains('(')
        || query.contains(')')
        || query.contains('+')
        || query.contains('?')
        || query.contains('{')
        || query.contains('}')
        || query.contains(".*")
}

fn looks_like_file_target(query: &str) -> bool {
    const FILE_EXTENSIONS: &[&str] = &[
        ".rs", ".py", ".js", ".mjs", ".cjs", ".ts", ".tsx", ".go", ".java", ".c", ".h", ".cpp",
        ".cc", ".cxx", ".hpp", ".hxx", ".cs", ".sql",
    ];

    query.contains("--glob")
        || query.contains("--include")
        || query.contains("*.")
        || query.contains("src/")
        || FILE_EXTENSIONS.iter().any(|ext| query.contains(ext))
}

fn token_count(query: &str) -> usize {
    query
        .split(|character: char| !character.is_alphanumeric())
        .filter(|token| !token.is_empty())
        .count()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn looks_like_regex_detects_metacharacters() {
        for pattern in [
            "^pub fn",
            "fn\\s+\\w+",
            "[a-z]+",
            "fn(x)",
            "x+",
            "x?",
            "{3}",
            "x$",
            ".*",
        ] {
            assert!(
                looks_like_regex(pattern),
                "expected regex detection for: {pattern}"
            );
        }
        assert!(!looks_like_regex("error handling"));
        assert!(!looks_like_regex("handle_session_start"));
    }

    #[test]
    fn extract_search_command_handles_extra_tools_and_wrappers() {
        // `git grep "x" -- src/` scopes to src/ — passthrough so the user keeps grep semantics.
        assert_eq!(
            extract_search_command(Some("git grep \"error handling\" -- src/")),
            None
        );
        assert_eq!(
            extract_search_command(Some("git grep \"error handling some words\"")),
            Some("error handling some words".to_owned())
        );
        assert_eq!(
            extract_search_command(Some("time rg foo_bar")),
            Some("foo_bar".to_owned())
        );
        assert_eq!(
            extract_search_command(Some("nice rg \"some thing\"")),
            Some("some thing".to_owned())
        );
        assert_eq!(
            extract_search_command(Some("env FOO=1 BAR=2 rg target_pattern")),
            Some("target_pattern".to_owned())
        );
        assert_eq!(
            extract_search_command(Some("stdbuf -oL rg target_pattern")),
            Some("target_pattern".to_owned())
        );
        assert_eq!(
            extract_search_command(Some("stdbuf -oL -eL rg target_pattern")),
            Some("target_pattern".to_owned()),
            "stdbuf with chained flags must still resolve the real tool"
        );
        assert_eq!(
            extract_search_command(Some("ack pattern")),
            Some("pattern".to_owned())
        );
        // Non-search command must still return None.
        assert_eq!(extract_search_command(Some("ls -la src/")), None);
    }

    #[test]
    fn extract_search_command_passes_through_when_replace_consumes_first_quote() {
        // `--replace="X"` no longer hijacks the pattern slot.
        assert_eq!(
            extract_search_command(Some("rg --replace=\"X\" \"actual pattern phrase\"")),
            None,
            "replace mode is disruptive — must passthrough so grep semantics survive"
        );
    }

    #[test]
    fn extract_search_command_passes_through_on_output_flags() {
        for command in [
            "rg -l \"phrase to find here\"",
            "rg -c \"phrase to find here\"",
            "rg --count \"phrase to find here\"",
            "rg --count-matches \"phrase to find here\"",
            "rg --files-with-matches \"phrase to find here\"",
            "rg --json \"phrase to find here\"",
            "rg -A 3 \"phrase to find here\"",
            "rg -B 3 \"phrase to find here\"",
            "rg -C 3 \"phrase to find here\"",
            "rg --after-context=3 \"phrase to find here\"",
            "rg -o \"phrase to find here\"",
            "rg -v \"phrase to find here\"",
            "rg -q \"phrase to find here\"",
        ] {
            assert_eq!(
                extract_search_command(Some(command)),
                None,
                "output flag must trigger passthrough: {command}"
            );
        }
    }

    #[test]
    fn extract_search_command_passes_through_on_positional_path_scope() {
        assert_eq!(
            extract_search_command(Some("rg \"phrase to find here\" tests/")),
            None,
            "positional path arg means user scoped the grep — passthrough"
        );
        assert_eq!(
            extract_search_command(Some("grep -rn \"phrase to find here\" src/lib.rs")),
            None
        );
    }

    #[test]
    fn extract_search_command_handles_flag_values_via_equals() {
        // `--type=rust` and `--glob=*.rs` value tokens must not be mistaken for the pattern.
        assert_eq!(
            extract_search_command(Some("rg --type=rust \"phrase to find here\"")),
            Some("phrase to find here".to_owned())
        );
        assert_eq!(
            extract_search_command(Some("rg --type rust \"phrase to find here\"")),
            Some("phrase to find here".to_owned())
        );
    }

    #[test]
    fn extract_search_command_picks_explicit_e_pattern() {
        assert_eq!(
            extract_search_command(Some("rg -e \"phrase to find here\"")),
            Some("phrase to find here".to_owned())
        );
        // Multiple `-e` flags signal OR-match semantics; passthrough.
        assert_eq!(
            extract_search_command(Some(
                "rg -e \"first phrase here\" -e \"second phrase here\""
            )),
            None
        );
    }

    #[test]
    fn looks_like_file_target_covers_all_supported_extensions() {
        for ext in &[
            ".rs", ".py", ".js", ".mjs", ".cjs", ".ts", ".tsx", ".go", ".java", ".c", ".h", ".cpp",
            ".cc", ".cxx", ".hpp", ".hxx", ".cs", ".sql",
        ] {
            let query = format!("search routes{ext}");
            assert!(
                looks_like_file_target(&query),
                "expected passthrough for query containing {ext}"
            );
        }
        assert!(!looks_like_file_target("error handling retry logic"));
        assert!(looks_like_file_target("find *.rs files"));
        assert!(looks_like_file_target("search in src/"));
    }
}
