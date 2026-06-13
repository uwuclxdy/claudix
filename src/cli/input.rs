use std::path::Path;

use crate::config::validate_project_relative_path;
use crate::error::{ClaudixError, RecoveryHint, Result};
use crate::prompts::hints;
use crate::types::{Language, RelativePath};

pub(super) fn validate_search_query(query: &str) -> Result<()> {
    if query.trim().is_empty() {
        return Err(ClaudixError::ConfigInvalid {
            message: "search query cannot be empty".into(),
            recovery: RecoveryHint(hints::SEARCH_QUERY_NON_EMPTY),
        });
    }
    Ok(())
}

pub(super) fn validate_search_top_k(top_k: usize) -> Result<()> {
    if top_k == 0 {
        return Err(ClaudixError::ConfigInvalid {
            message: "top_k must be > 0".into(),
            recovery: RecoveryHint(hints::POSITIVE_TOP_K),
        });
    }

    Ok(())
}

pub(super) fn parse_language_filter(
    language_filter: Option<Vec<String>>,
) -> Result<Option<Vec<Language>>> {
    let Some(language_filter) = language_filter else {
        return Ok(None);
    };
    if language_filter.is_empty() {
        return Ok(None);
    }

    let mut parsed = Vec::with_capacity(language_filter.len());
    for language in language_filter {
        parsed.push(parse_language(&language)?);
    }

    Ok(Some(parsed))
}

pub(super) fn parse_path_prefix(path_prefix: Option<String>) -> Result<Option<RelativePath>> {
    let Some(prefix) = path_prefix else {
        return Ok(None);
    };
    let trimmed = prefix.trim();
    if trimmed.is_empty() {
        return Ok(None);
    }

    validate_project_relative_path(Path::new(trimmed), "search.path_prefix")?;
    let prefix = RelativePath::new(trimmed.to_owned());
    // Defense-in-depth: `RelativePath::new` only normalizes separators, so
    // re-check the constructed value against the trust boundary even though the
    // string already passed `validate_project_relative_path` above.
    prefix.reject_escape(hints::PROJECT_RELATIVE_PATH)?;
    Ok(Some(prefix))
}

fn parse_language(value: &str) -> Result<Language> {
    Language::from_filter_input(value).ok_or_else(|| ClaudixError::ConfigInvalid {
        message: format!("unsupported language filter: {}", value.trim()),
        recovery: RecoveryHint(hints::VALID_LANGUAGE_FILTERS),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_search_query_rejects_empty_and_whitespace() {
        for query in ["", "   ", "\t", "\n"] {
            let result = validate_search_query(query);
            assert!(matches!(result, Err(ClaudixError::ConfigInvalid { .. })));
        }
    }

    #[test]
    fn validate_search_top_k_rejects_zero() {
        let result = validate_search_top_k(0);

        assert!(matches!(result, Err(ClaudixError::ConfigInvalid { .. })));
    }

    #[test]
    fn parse_language_filter_accepts_supported_aliases() {
        let parsed = parse_language_filter(Some(vec![
            " rust ".to_owned(),
            "js".to_owned(),
            "ts".to_owned(),
            "c++".to_owned(),
        ]));

        assert!(matches!(
            parsed,
            Ok(Some(ref languages))
                if languages == &vec![
                    Language::Rust,
                    Language::JavaScript,
                    Language::TypeScript,
                    Language::Cpp,
                ]
        ));
    }

    #[test]
    fn parse_language_filter_rejects_extensions() {
        let parsed = parse_language_filter(Some(vec!["rs".to_owned()]));

        assert!(matches!(parsed, Err(ClaudixError::ConfigInvalid { .. })));
    }

    #[test]
    fn parse_language_filter_treats_empty_list_as_no_filter() {
        let parsed = parse_language_filter(Some(Vec::new()));

        assert!(matches!(parsed, Ok(None)));
    }

    #[test]
    fn parse_path_prefix_treats_blank_values_as_no_filter() {
        assert!(matches!(parse_path_prefix(None), Ok(None)));
        assert!(matches!(
            parse_path_prefix(Some("   ".to_owned())),
            Ok(None)
        ));
        assert_eq!(
            parse_path_prefix(Some(" src/math ".to_owned()))
                .ok()
                .flatten()
                .as_ref()
                .map(RelativePath::as_str),
            Some("src/math")
        );
    }

    #[test]
    fn parse_path_prefix_rejects_paths_outside_project() {
        for path_prefix in ["../src", "/tmp/src"] {
            let parsed = parse_path_prefix(Some(path_prefix.to_owned()));
            assert!(matches!(parsed, Err(ClaudixError::ConfigInvalid { .. })));
        }
    }
}
