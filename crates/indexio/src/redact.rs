//! Secret redaction for the two sources that record what an agent saw
//! (SPEC-P10 §39): rendered session transcripts and stored command output.
//! A session that prints its environment or a connection string puts the
//! values into the recall index otherwise. The rules are deliberately
//! shape-based and conservative: an assignment to a secret-looking key
//! (`KEY`, `TOKEN`, `SECRET`, `PASSWORD`, `DATABASE_URL` …), a password
//! inside a URL's user-info, and the well-known token prefixes (`sk-`,
//! `AKIA`, `ghp_`, `xoxb-`, JWTs, PEM private-key blocks). Code and prose
//! are left alone; only the value is replaced.

use std::borrow::Cow;
use std::sync::OnceLock;

use regex::Regex;

const MASK: &str = "<redacted>";

/// `KEY=value`, `"token": "value"`, `SECRET: value`: a secret-looking name
/// (ending in the keyword, so `token_count` and `AUTHOR` are not names)
/// and the value after `=` or `:`. Whether the value is worth masking is
/// decided in [`looks_secret`], not here.
fn assignment() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r#"(?i)\b([A-Z0-9_.-]*(?:KEY|TOKEN|SECRET|PASSWORD|PASSWD|PASS|PWD|CREDENTIALS?|AUTH|DATABASE_URL|REDIS_URL|DSN)(?:S|_ID)?\b["']?\s*[:=]\s*["']?)([A-Za-z0-9_\-+/=.:@%~]{8,})"#).unwrap()
    })
}

/// A value an assignment rule may mask: long enough, and either carrying a
/// digit or long enough to be a token — so `std::env::var`, `Bearer` and an
/// identifier on the right of `=` in source code are left alone.
fn looks_secret(value: &str) -> bool {
    value.len() >= 8 && !value.contains("::") && (value.bytes().any(|b| b.is_ascii_digit()) || value.len() >= 20)
}

fn rules() -> &'static [(Regex, &'static str)] {
    static RULES: OnceLock<Vec<(Regex, &'static str)>> = OnceLock::new();
    RULES.get_or_init(|| {
        vec![
            // scheme://user:password@host — keep the user, mask the password
            (Regex::new(r"(?i)\b([a-z][a-z0-9+.-]*://[^/\s:@]+:)([^@\s/]{3,})@").unwrap(), "${1}<redacted>@"),
            // Authorization headers and bearer tokens
            (Regex::new(r"(?i)\b(bearer\s+|basic\s+)([A-Za-z0-9._~+/=-]{16,})").unwrap(), "${1}<redacted>"),
            // well-known token shapes
            (Regex::new(r"\b(sk|rk|pk)-(?:live|test|proj|ant)?-?[A-Za-z0-9_-]{16,}\b").unwrap(), MASK),
            (Regex::new(r"\bAKIA[0-9A-Z]{16}\b").unwrap(), MASK),
            (Regex::new(r"\b(?:ghp|gho|ghu|ghs|ghr|github_pat)_[A-Za-z0-9_]{20,}\b").unwrap(), MASK),
            (Regex::new(r"\bxox[abprs]-[A-Za-z0-9-]{10,}\b").unwrap(), MASK),
            (Regex::new(r"\bAIza[0-9A-Za-z_-]{30,}\b").unwrap(), MASK),
            (Regex::new(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b").unwrap(), MASK),
            // PEM private key bodies
            (
                Regex::new(r"(?s)(-----BEGIN [A-Z ]*PRIVATE KEY-----)(.*?)(-----END [A-Z ]*PRIVATE KEY-----)").unwrap(),
                "${1}\n<redacted>\n${3}",
            ),
        ]
    })
}

/// `text` with secret values replaced by `<redacted>`; borrowed when
/// nothing matched.
pub fn redact(text: &str) -> Cow<'_, str> {
    let mut out: Cow<str> = Cow::Borrowed(text);
    if assignment().is_match(&out) {
        let replaced = assignment().replace_all(&out, |c: &regex::Captures| {
            if looks_secret(&c[2]) {
                format!("{}{MASK}", &c[1])
            } else {
                c[0].to_string()
            }
        });
        if replaced != out {
            out = Cow::Owned(replaced.into_owned());
        }
    }
    for (re, rep) in rules() {
        if re.is_match(&out) {
            out = Cow::Owned(re.replace_all(&out, *rep).into_owned());
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn values_of_secret_looking_keys_are_masked_and_names_kept() {
        let s = "DATABASE_URL=postgres://app:s3cr3tpw@db.internal:5432/app\nSMTP_PORT=587\nTWILIO_AUTH_TOKEN=abcdef0123456789abcdef0123456789\n";
        let r = redact(s);
        assert!(r.contains("DATABASE_URL=<redacted>"), "{r}");
        assert!(r.contains("SMTP_PORT=587"), "a port is not a secret: {r}");
        assert!(r.contains("TWILIO_AUTH_TOKEN=<redacted>"), "{r}");
        assert!(!r.contains("s3cr3tpw") && !r.contains("abcdef0123"), "{r}");
    }

    #[test]
    fn json_headers_urls_and_known_token_shapes() {
        let s = r#"{"api_key": "sk-live-abcdefghijklmnopqrstuvwxyz012345", "user": "bob"}
Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U
redis://default:hunter22@cache:6379/0
ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdef0123
"#;
        let r = redact(s);
        assert!(r.contains(r#""api_key": "<redacted>""#), "{r}");
        assert!(r.contains(r#""user": "bob""#), "{r}");
        assert!(r.contains("Bearer <redacted>"), "{r}");
        assert!(r.contains("redis://default:<redacted>@cache:6379/0"), "{r}");
        assert!(!r.contains("ghp_ABC"), "{r}");
    }

    #[test]
    fn code_and_prose_pass_through_untouched() {
        let s = "fn load_config(path: &str) -> Result<Config> {\n    let key = std::env::var(\"API_KEY\")?;\n    let token_count = tokens.len();\n}\nThe password field is required.\n";
        assert!(matches!(redact(s), Cow::Borrowed(_)), "{}", redact(s));
        let pem = "-----BEGIN RSA PRIVATE KEY-----\nMIIEowIBAAKCAQEA\nMIIEowIBAAKCAQEB\n-----END RSA PRIVATE KEY-----\n";
        let r = redact(pem);
        assert!(r.contains("-----BEGIN RSA PRIVATE KEY-----\n<redacted>\n-----END RSA PRIVATE KEY-----"), "{r}");
    }
}
