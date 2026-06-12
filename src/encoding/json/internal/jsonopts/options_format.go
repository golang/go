// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.jsonv2

package jsonopts

// NOTE: While [ExperimentalSupportFormatTag] is exported,
// it is in an internal package and thus inaccessible for public use.
//
// The [github.com/go-json-experiment/json] module is kept in sync
// with the Go standard standard library and will expose this option
// in a way that public code can now directly reference.

type supportFormatTag struct {
	Options
	v bool
}

var (
	enableFormatTag  = supportFormatTag{v: true}
	disableFormatTag = supportFormatTag{v: false}
)

// ExperimentalSupportFormatTag is a marker method that is specially
// recognized by the "encoding/json/v2" runtime.
func (o *supportFormatTag) ExperimentalSupportFormatTag() bool {
	return o.v
}

// ExperimentalSupportFormatTag enables support for the `format` tag.
//
// WARNING: This is an experimental feature and will be removed in the future
// as either a failed experiment or be formally included in "encoding/json/v2"
// in some semantically similar form, in which case, users of this option
// must migrate to the officially supported feature.
//
// The `format` tag was originally part of the "encoding/json/v2" experiment
// but support for it was removed (see https://go.dev/issue/79071) for the
// initial release of "encoding/json/v2" in light of the anticipation that
// the Go language might support typed struct tags (https://go.dev/issue/74472).
//
// Typed struct tags are a more expressive and type-safe way to express format
// attributes than the bespoke `format` tag option that implements a miniature
// domain-specific language (DSL) within the "json" package itself.
//
// While "encoding/json/v2" was in the experimental phase,
// some users were already depending on the `format` tag option.
// This experimental option exists to provide a temporary workaround
// until the (hopeful) inclusion of typed struct tags and support for
// formatting directives in "encoding/json/v2" using that language mechanism.
//
// This option enables support for the `format` tag option, which specifies
// a format flag used to specialize the formatting of the field value.
// The option is a key-value pair specified as "format:value" where
// the value must be either a literal consisting of letters and numbers
// (e.g., `format:RFC3339`) or a single-quoted string literal
// (e.g., `format:'2006-01-02'`). The interpretation of the format flag
// is determined by the struct field type.
//
// Go types with alternative representations are as follows:
//
//   - A Go []byte or [N]byte is usually represented as a JSON string
//     containing the binary value encoded using RFC 4648.
//     If the format is "base64" or unspecified, then this uses RFC 4648, section 4.
//     If the format is "base64url", then this uses RFC 4648, section 5.
//     If the format is "base32", then this uses RFC 4648, section 6.
//     If the format is "base32hex", then this uses RFC 4648, section 7.
//     If the format is "base16" or "hex", then this uses RFC 4648, section 8.
//     If the format is "array", then the bytes value is represented as a JSON array
//     where each element recursively uses the JSON representation of each byte.
//
//   - A Go float is usually represented as a JSON number.
//     If the format is "nonfinite", then NaN, +Inf, and -Inf are represented as
//     the JSON strings "NaN", "Infinity", and "-Infinity", respectively.
//     Without the use of this format, such string values result in a [SemanticError].
//
//   - A nil Go map is usually encoded using an empty JSON object.
//     If the format is "emitnull", then a nil map is encoded as a JSON null.
//     If the format is "emitempty", then a nil map is encoded as an empty JSON object,
//     regardless of whether [FormatNilMapAsNull] is specified.
//
//   - A nil Go slice is usually encoded using an empty JSON array.
//     If the format is "emitnull", then a nil slice is encoded as a JSON null.
//     If the format is "emitempty", then a nil slice is encoded as an empty JSON array,
//     regardless of whether [FormatNilSliceAsNull] is specified.
//
//   - A Go pointer usually uses the JSON representation of the underlying value.
//     The format is forwarded to the marshaling and unmarshaling of the underlying type.
//
//   - A Go [time.Time] is usually represented as a JSON string containing
//     the timestamp formatted in RFC 3339 with nanosecond precision.
//     If the format matches one of the format constants declared
//     in the time package (e.g., RFC1123), then that format is used.
//     If the format is "unix", "unixmilli", "unixmicro", or "unixnano",
//     then the timestamp is represented as a possibly fractional JSON number
//     of the number of seconds (or milliseconds, microseconds, or nanoseconds)
//     since the Unix epoch, which is January 1st, 1970 at 00:00:00 UTC.
//     To avoid a fractional component when encoding,
//     round the timestamp to the relevant unit.
//     Otherwise if non-empty, the format is used as-is and
//     encoded using [time.Time.Format] and
//     decoded using [time.Time.Parse].
//
//   - A Go [time.Duration] usually has no default representation.
//     If the format is "sec", "milli", "micro", or "nano",
//     then the duration is represented as a possibly fractional JSON number
//     of the number of seconds (or milliseconds, microseconds, or nanoseconds).
//     To avoid a fractional component when encoding,
//     round the duration to the relevant unit.
//     If the format is "units", it is represented as a JSON string
//     encoded using [time.Duration.String] and decoded using [time.ParseDuration]
//     (e.g., "1h30m" for 1 hour 30 minutes).
//     If the format is "iso8601", it is represented as a JSON string using the
//     ISO 8601 standard for durations (e.g., "PT1H30M" for 1 hour 30 minutes)
//     using only accurate units of hours, minutes, and seconds.
func ExperimentalSupportFormatTag(v bool) Options {
	if v {
		return &enableFormatTag
	} else {
		return &disableFormatTag
	}
}
