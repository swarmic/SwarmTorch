//! Offline-first observability primitives (artifact-compatible).
//!
//! This module defines SwarmTorch's canonical ID types and record schemas for
//! spans/events/metrics. The goal is to be compatible with W3C Trace Context and
//! OpenTelemetry ID sizing, while remaining OTel-independent.

use core::fmt;

/// Error parsing a hex-encoded ID.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParseIdError {
    /// The input had the wrong length for the target ID type.
    InvalidLength,
    /// The input contained non-hex characters.
    InvalidHex,
    /// All-zero IDs are invalid by contract.
    AllZeroInvalid,
}

impl fmt::Display for ParseIdError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ParseIdError::InvalidLength => write!(f, "invalid id length"),
            ParseIdError::InvalidHex => write!(f, "invalid hex in id"),
            ParseIdError::AllZeroInvalid => write!(f, "all-zero id is invalid"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for ParseIdError {}

const HEX_LOWER: &[u8; 16] = b"0123456789abcdef";

fn write_hex_lower(bytes: &[u8], out: &mut [u8]) {
    debug_assert_eq!(out.len(), bytes.len() * 2);
    for (i, b) in bytes.iter().enumerate() {
        out[i * 2] = HEX_LOWER[(b >> 4) as usize];
        out[i * 2 + 1] = HEX_LOWER[(b & 0x0f) as usize];
    }
}

fn decode_hex_nibble(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        b'A'..=b'F' => Some(b - b'A' + 10),
        _ => None,
    }
}

fn parse_hex_exact<const N: usize>(s: &str) -> core::result::Result<[u8; N], ParseIdError> {
    let expected = N * 2;
    if s.len() != expected {
        return Err(ParseIdError::InvalidLength);
    }
    let bytes = s.as_bytes();
    let mut out = [0u8; N];
    for i in 0..N {
        let hi = decode_hex_nibble(bytes[i * 2]).ok_or(ParseIdError::InvalidHex)?;
        let lo = decode_hex_nibble(bytes[i * 2 + 1]).ok_or(ParseIdError::InvalidHex)?;
        out[i] = (hi << 4) | lo;
    }
    Ok(out)
}

fn is_all_zero(bytes: &[u8]) -> bool {
    bytes.iter().all(|b| *b == 0)
}

fn serialize_id_hex_or_bytes<S, const N: usize>(
    bytes: &[u8; N],
    serializer: S,
) -> core::result::Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    if serializer.is_human_readable() {
        // Rust 1.75 doesn't support `N * 2` in array lengths; use a fixed max buffer.
        let mut buf = [0u8; 32];
        let hex_len = N * 2;
        write_hex_lower(bytes, &mut buf[..hex_len]);
        let s = core::str::from_utf8(&buf[..hex_len]).map_err(serde::ser::Error::custom)?;
        serializer.serialize_str(s)
    } else {
        serializer.serialize_bytes(bytes)
    }
}

fn deserialize_fixed_bytes<'de, D, const N: usize>(
    deserializer: D,
) -> core::result::Result<[u8; N], D::Error>
where
    D: serde::Deserializer<'de>,
{
    struct FixedBytesVisitor<const N: usize>;

    impl<'de, const N: usize> serde::de::Visitor<'de> for FixedBytesVisitor<N> {
        type Value = [u8; N];

        fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "a byte array of length {N}")
        }

        fn visit_bytes<E>(self, v: &[u8]) -> core::result::Result<Self::Value, E>
        where
            E: serde::de::Error,
        {
            if v.len() != N {
                return Err(E::invalid_length(v.len(), &self));
            }
            let mut out = [0u8; N];
            out.copy_from_slice(v);
            Ok(out)
        }

        #[cfg(feature = "alloc")]
        fn visit_byte_buf<E>(self, v: alloc::vec::Vec<u8>) -> core::result::Result<Self::Value, E>
        where
            E: serde::de::Error,
        {
            self.visit_bytes(&v)
        }
    }

    // `visit_byte_buf` needs `alloc`; when `alloc` is not available, serde will use `visit_bytes`.
    deserializer.deserialize_bytes(FixedBytesVisitor::<N>)
}

fn deserialize_id_hex_or_bytes<'de, D, const N: usize>(
    deserializer: D,
) -> core::result::Result<[u8; N], D::Error>
where
    D: serde::Deserializer<'de>,
{
    if deserializer.is_human_readable() {
        struct HexVisitor<const N: usize>;

        impl<'de, const N: usize> serde::de::Visitor<'de> for HexVisitor<N> {
            type Value = [u8; N];

            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "a lowercase hex string of length {}", N * 2)
            }

            fn visit_str<E>(self, v: &str) -> core::result::Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                parse_hex_exact::<N>(v).map_err(E::custom)
            }

            #[cfg(feature = "alloc")]
            fn visit_string<E>(
                self,
                v: alloc::string::String,
            ) -> core::result::Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                self.visit_str(&v)
            }
        }

        deserializer.deserialize_str(HexVisitor::<N>)
    } else {
        deserialize_fixed_bytes::<D, N>(deserializer)
    }
}

/// A 16-byte trace identifier (32 lowercase hex chars in human-readable encodings).
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct TraceId(pub [u8; 16]);

impl TraceId {
    pub const fn from_bytes(bytes: [u8; 16]) -> Self {
        Self(bytes)
    }

    pub const fn as_bytes(&self) -> &[u8; 16] {
        &self.0
    }

    pub fn is_valid(&self) -> bool {
        !is_all_zero(&self.0)
    }

    pub fn parse_hex(s: &str) -> core::result::Result<Self, ParseIdError> {
        let bytes = parse_hex_exact::<16>(s)?;
        if is_all_zero(&bytes) {
            return Err(ParseIdError::AllZeroInvalid);
        }
        Ok(Self(bytes))
    }

    pub fn write_lower_hex(&self, out: &mut [u8; 32]) {
        write_hex_lower(&self.0, out);
    }
}

impl fmt::Debug for TraceId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl fmt::Display for TraceId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut buf = [0u8; 32];
        self.write_lower_hex(&mut buf);
        let s = core::str::from_utf8(&buf).map_err(|_| fmt::Error)?;
        f.write_str(s)
    }
}

impl serde::Serialize for TraceId {
    fn serialize<S>(&self, serializer: S) -> core::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serialize_id_hex_or_bytes::<S, 16>(&self.0, serializer)
    }
}

impl<'de> serde::Deserialize<'de> for TraceId {
    fn deserialize<D>(deserializer: D) -> core::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let bytes = deserialize_id_hex_or_bytes::<D, 16>(deserializer)?;
        if is_all_zero(&bytes) {
            return Err(serde::de::Error::custom(ParseIdError::AllZeroInvalid));
        }
        Ok(Self(bytes))
    }
}

/// An 8-byte span identifier (16 lowercase hex chars in human-readable encodings).
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct SpanId(pub [u8; 8]);

impl SpanId {
    pub const fn from_bytes(bytes: [u8; 8]) -> Self {
        Self(bytes)
    }

    pub const fn as_bytes(&self) -> &[u8; 8] {
        &self.0
    }

    pub fn is_valid(&self) -> bool {
        !is_all_zero(&self.0)
    }

    pub fn parse_hex(s: &str) -> core::result::Result<Self, ParseIdError> {
        let bytes = parse_hex_exact::<8>(s)?;
        if is_all_zero(&bytes) {
            return Err(ParseIdError::AllZeroInvalid);
        }
        Ok(Self(bytes))
    }

    pub fn write_lower_hex(&self, out: &mut [u8; 16]) {
        write_hex_lower(&self.0, out);
    }
}

impl fmt::Debug for SpanId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl fmt::Display for SpanId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut buf = [0u8; 16];
        self.write_lower_hex(&mut buf);
        let s = core::str::from_utf8(&buf).map_err(|_| fmt::Error)?;
        f.write_str(s)
    }
}

impl serde::Serialize for SpanId {
    fn serialize<S>(&self, serializer: S) -> core::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serialize_id_hex_or_bytes::<S, 8>(&self.0, serializer)
    }
}

impl<'de> serde::Deserialize<'de> for SpanId {
    fn deserialize<D>(deserializer: D) -> core::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let bytes = deserialize_id_hex_or_bytes::<D, 8>(deserializer)?;
        if is_all_zero(&bytes) {
            return Err(serde::de::Error::custom(ParseIdError::AllZeroInvalid));
        }
        Ok(Self(bytes))
    }
}

/// A 16-byte run identifier (by default, equal to the run root `trace_id`).
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct RunId(pub [u8; 16]);

impl RunId {
    pub const fn from_bytes(bytes: [u8; 16]) -> Self {
        Self(bytes)
    }

    pub const fn as_bytes(&self) -> &[u8; 16] {
        &self.0
    }

    pub fn is_valid(&self) -> bool {
        !is_all_zero(&self.0)
    }

    pub fn parse_hex(s: &str) -> core::result::Result<Self, ParseIdError> {
        let bytes = parse_hex_exact::<16>(s)?;
        if is_all_zero(&bytes) {
            return Err(ParseIdError::AllZeroInvalid);
        }
        Ok(Self(bytes))
    }

    pub fn write_lower_hex(&self, out: &mut [u8; 32]) {
        write_hex_lower(&self.0, out);
    }
}

impl fmt::Debug for RunId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl fmt::Display for RunId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut buf = [0u8; 32];
        self.write_lower_hex(&mut buf);
        let s = core::str::from_utf8(&buf).map_err(|_| fmt::Error)?;
        f.write_str(s)
    }
}

impl serde::Serialize for RunId {
    fn serialize<S>(&self, serializer: S) -> core::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serialize_id_hex_or_bytes::<S, 16>(&self.0, serializer)
    }
}

impl<'de> serde::Deserialize<'de> for RunId {
    fn deserialize<D>(deserializer: D) -> core::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let bytes = deserialize_id_hex_or_bytes::<D, 16>(deserializer)?;
        if is_all_zero(&bytes) {
            return Err(serde::de::Error::custom(ParseIdError::AllZeroInvalid));
        }
        Ok(Self(bytes))
    }
}

#[cfg(feature = "alloc")]
use alloc::{collections::BTreeMap, string::String};

/// A sink for SwarmTorch span/event/metric records.
///
/// This is intentionally OTel-independent: it emits SwarmTorch's canonical record
/// types (artifact-compatible). Implementations may export to OTLP or persist
/// NDJSON/Parquet as an implementation detail.
#[cfg(feature = "alloc")]
pub trait RunEventEmitter: Send + Sync {
    type Error;

    fn emit_span(&self, span: &SpanRecord) -> core::result::Result<(), Self::Error>;
    fn emit_event(&self, event: &EventRecord) -> core::result::Result<(), Self::Error>;
    fn emit_metric(&self, metric: &MetricRecord) -> core::result::Result<(), Self::Error>;
}

/// Attribute values for spans/events/metrics.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(untagged)]
pub enum AttrValue {
    Str(String),
    Bool(bool),
    I64(i64),
    U64(u64),
    F64(f64),
}

/// Canonical attribute map type (deterministic ordering via BTreeMap).
#[cfg(feature = "alloc")]
pub type AttrMap = BTreeMap<String, AttrValue>;

/// A span record (NDJSON line schema v1).
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SpanRecord {
    pub schema_version: u32,
    pub trace_id: TraceId,
    pub span_id: SpanId,
    pub parent_span_id: Option<SpanId>,
    pub name: String,
    pub start_unix_nanos: u64,
    pub end_unix_nanos: Option<u64>,
    pub attrs: AttrMap,
}

/// An event record (NDJSON line schema v1).
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct EventRecord {
    pub schema_version: u32,
    pub ts_unix_nanos: u64,
    pub trace_id: TraceId,
    pub span_id: Option<SpanId>,
    pub name: String,
    pub attrs: AttrMap,
}

/// A metric record (NDJSON line schema v1).
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct MetricRecord {
    pub schema_version: u32,
    pub ts_unix_nanos: u64,
    pub trace_id: TraceId,
    pub span_id: Option<SpanId>,
    pub name: String,
    pub value: f64,
    pub unit: Option<String>,
    pub attrs: AttrMap,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trace_id_hex_roundtrip() {
        let id = TraceId::from_bytes([0x11u8; 16]);
        let s = id.to_string();
        let parsed = TraceId::parse_hex(&s).unwrap();
        assert_eq!(parsed.as_bytes(), id.as_bytes());
    }

    #[test]
    fn span_id_hex_roundtrip() {
        let id = SpanId::from_bytes([0x22u8; 8]);
        let s = id.to_string();
        let parsed = SpanId::parse_hex(&s).unwrap();
        assert_eq!(parsed.as_bytes(), id.as_bytes());
    }

    #[test]
    fn all_zero_invalid() {
        assert_eq!(
            TraceId::parse_hex("00000000000000000000000000000000").unwrap_err(),
            ParseIdError::AllZeroInvalid
        );
        assert_eq!(
            SpanId::parse_hex("0000000000000000").unwrap_err(),
            ParseIdError::AllZeroInvalid
        );
        assert_eq!(
            RunId::parse_hex("00000000000000000000000000000000").unwrap_err(),
            ParseIdError::AllZeroInvalid
        );
    }

    #[test]
    fn invalid_length_rejected() {
        assert_eq!(
            TraceId::parse_hex("abcd").unwrap_err(),
            ParseIdError::InvalidLength
        );
        assert_eq!(
            SpanId::parse_hex("abcd").unwrap_err(),
            ParseIdError::InvalidLength
        );
    }
}
