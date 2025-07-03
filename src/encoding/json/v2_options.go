// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.jsonv2

// Migrating to v2
//
// This package (i.e., [encoding/json]) is now formally known as the v1 package
// since a v2 package now exists at [encoding/json/v2].
// All the behavior of the v1 package is implemented in terms of
// the v2 package with the appropriate set of options specified that
// preserve the historical behavior of v1.
//
// The [jsonv2.Marshal] function is the newer equivalent of v1 [Marshal].
// The [jsonv2.Unmarshal] function is the newer equivalent of v1 [Unmarshal].
// The v2 functions have the same calling signature as the v1 equivalent
// except that they take in variadic [Options] arguments that can be specified
// to alter the behavior of marshal or unmarshal. Both v1 and v2 generally
// behave in similar ways, but there are some notable differences.
//
// The following is a list of differences between v1 and v2:
//
//   - In v1, JSON object members are unmarshaled into a Go struct using a
//     case-insensitive name match with the JSON name of the fields.
//     In contrast, v2 matches fields using an exact, case-sensitive match.
//     The [jsonv2.MatchCaseInsensitiveNames] and [MatchCaseSensitiveDelimiter]
//     options control this behavior difference. To explicitly specify a Go struct
//     field to use a particular name matching scheme, either the `case:ignore`
//     or the `case:strict` field option can be specified.
//     Field-specified options take precedence over caller-specified options.
//
//   - In v1, when marshaling a Go struct, a field marked as `omitempty`
//     is omitted if the field value is an "empty" Go value, which is defined as
//     false, 0, a nil pointer, a nil interface value, and
//     any empty array, slice, map, or string. In contrast, v2 redefines
//     `omitempty` to omit a field if it encodes as an "empty" JSON value,
//     which is defined as a JSON null, or an empty JSON string, object, or array.
//     The [OmitEmptyWithLegacyDefinition] option controls this behavior difference.
//     Note that `omitempty` behaves identically in both v1 and v2 for a
//     Go array, slice, map, or string (assuming no user-defined MarshalJSON method
//     overrides the default representation). Existing usages of `omitempty` on a
//     Go bool, number, pointer, or interface value should migrate to specifying
//     `omitzero` instead (which is identically supported in both v1 and v2).
//
//   - In v1, a Go struct field marked as `string` can be used to quote a
//     Go string, bool, or number as a JSON string. It does not recursively
//     take effect on composite Go types. In contrast, v2 restricts
//     the `string` option to only quote a Go number as a JSON string.
//     It does recursively take effect on Go numbers within a composite Go type.
//     The [StringifyWithLegacySemantics] option controls this behavior difference.
//
//   - In v1, a nil Go slice or Go map is marshaled as a JSON null.
//     In contrast, v2 marshals a nil Go slice or Go map as
//     an empty JSON array or JSON object, respectively.
//     The [jsonv2.FormatNilSliceAsNull] and [jsonv2.FormatNilMapAsNull] options
//     control this behavior difference. To explicitly specify a Go struct field
//     to use a particular representation for nil, either the `format:emitempty`
//     or `format:emitnull` field option can be specified.
//     Field-specified options take precedence over caller-specified options.
//
//   - In v1, a Go array may be unmarshaled from a JSON array of any length.
//     In contrast, in v2 a Go array must be unmarshaled from a JSON array
//     of the same length, otherwise it results in an error.
//     The [UnmarshalArrayFromAnyLength] option controls this behavior difference.
//
//   - In v1, a Go byte array is represented as a JSON array of JSON numbers.
//     In contrast, in v2 a Go byte array is represented as a Base64-encoded JSON string.
//     The [FormatBytesWithLegacySemantics] option controls this behavior difference.
//     To explicitly specify a Go struct field to use a particular representation,
//     either the `format:array` or `format:base64` field option can be specified.
//     Field-specified options take precedence over caller-specified options.
//
//   - In v1, MarshalJSON methods declared on a pointer receiver are only called
//     if the Go value is addressable. In contrast, in v2 a MarshalJSON method
//     is always callable regardless of addressability.
//     The [CallMethodsWithLegacySemantics] option controls this behavior difference.
//
//   - In v1, MarshalJSON and UnmarshalJSON methods are never called for Go map keys.
//     In contrast, in v2 a MarshalJSON or UnmarshalJSON method is eligible for
//     being called for Go map keys.
//     The [CallMethodsWithLegacySemantics] option controls this behavior difference.
//
//   - In v1, a Go map is marshaled in a deterministic order.
//     In contrast, in v2 a Go map is marshaled in a non-deterministic order.
//     The [jsonv2.Deterministic] option controls this behavior difference.
//
//   - In v1, JSON strings are encoded with HTML-specific or JavaScript-specific
//     characters being escaped. In contrast, in v2 JSON strings use the minimal
//     encoding and only escape if required by the JSON grammar.
//     The [jsontext.EscapeForHTML] and [jsontext.EscapeForJS] options
//     control this behavior difference.
//
//   - In v1, bytes of invalid UTF-8 within a string are silently replaced with
//     the Unicode replacement character. In contrast, in v2 the presence of
//     invalid UTF-8 results in an error. The [jsontext.AllowInvalidUTF8] option
//     controls this behavior difference.
//
//   - In v1, a JSON object with duplicate names is permitted.
//     In contrast, in v2 a JSON object with duplicate names results in an error.
//     The [jsontext.AllowDuplicateNames] option controls this behavior difference.
//
//   - In v1, when unmarshaling a JSON null into a non-empty Go value it will
//     inconsistently either zero out the value or do nothing.
//     In contrast, in v2 unmarshaling a JSON null will consistently and always
//     zero out the underlying Go value. The [MergeWithLegacySemantics] option
//     controls this behavior difference.
//
//   - In v1, when unmarshaling a JSON value into a non-zero Go value,
//     it merges into the original Go value for array elements, slice elements,
//     struct fields (but not map values),
//     pointer values, and interface values (only if a non-nil pointer).
//     In contrast, in v2 unmarshal merges into the Go value
//     for struct fields, map values, pointer values, and interface values.
//     In general, the v2 semantic merges when unmarshaling a JSON object,
//     otherwise it replaces the value. The [MergeWithLegacySemantics] option
//     controls this behavior difference.
//
//   - In v1, a [time.Duration] is represented as a JSON number containing
//     the decimal number of nanoseconds. In contrast, in v2 a [time.Duration]
//     is represented as a JSON string containing the formatted duration
//     (e.g., "1h2m3.456s") according to [time.Duration.String].
//     The [FormatTimeWithLegacySemantics] option controls this behavior difference.
//     To explicitly specify a Go struct field to use a particular representation,
//     either the `format:nano` or `format:units` field option can be specified.
//     Field-specified options take precedence over caller-specified options.
//
//   - In v1, errors are never reported at runtime for Go struct types
//     that have some form of structural error (e.g., a malformed tag option).
//     In contrast, v2 reports a runtime error for Go types that are invalid
//     as they relate to JSON serialization. For example, a Go struct
//     with only unexported fields cannot be serialized.
//     The [ReportErrorsWithLegacySemantics] option controls this behavior difference.
//
// As mentioned, the entirety of v1 is implemented in terms of v2,
// where options are implicitly specified to opt into legacy behavior.
// For example, [Marshal] directly calls [jsonv2.Marshal] with [DefaultOptionsV1].
// Similarly, [Unmarshal] directly calls [jsonv2.Unmarshal] with [DefaultOptionsV1].
// The [DefaultOptionsV1] option represents the set of all options that specify
// default v1 behavior.
//
// For many of the behavior differences, there are Go struct field options
// that the author of a Go type can specify to control the behavior such that
// the type is represented identically in JSON under either v1 or v2 semantics.
//
// The availability of [DefaultOptionsV1] and [jsonv2.DefaultOptionsV2],
// where later options take precedence over former options allows for
// a gradual migration from v1 to v2. For example:
//
//   - jsonv1.Marshal(v)
//     uses default v1 semantics.
//
//   - jsonv2.Marshal(v, jsonv1.DefaultOptionsV1())
//     is semantically equivalent to jsonv1.Marshal
//     and thus uses default v1 semantics.
//
//   - jsonv2.Marshal(v, jsonv1.DefaultOptionsV1(), jsontext.AllowDuplicateNames(false))
//     uses mostly v1 semantics, but opts into one particular v2-specific behavior.
//
//   - jsonv2.Marshal(v, jsonv1.CallMethodsWithLegacySemantics(true))
//     uses mostly v2 semantics, but opts into one particular v1-specific behavior.
//
//   - jsonv2.Marshal(v, ..., jsonv2.DefaultOptionsV2())
//     is semantically equivalent to jsonv2.Marshal since
//     jsonv2.DefaultOptionsV2 overrides any options specified earlier
//     and thus uses default v2 semantics.
//
//   - jsonv2.Marshal(v)
//     uses default v2 semantics.
//
// All new usages of "json" in Go should use the v2 package,
// but the v1 package will forever remain supported.
package json

import (
	"encoding"

	"encoding/json/internal/jsonflags"
	"encoding/json/internal/jsonopts"
	"encoding/json/jsontext"
	jsonv2 "encoding/json/v2"
)

// Reference encoding, jsonv2, and jsontext packages to assist pkgsite
// in being able to hotlink references to those packages.
var (
	_ encoding.TextMarshaler
	_ encoding.TextUnmarshaler
	_ jsonv2.Options
	_ jsontext.Options
)

// Options are a set of options to configure the v2 "json" package
// to operate with v1 semantics for particular features.
// Values of this type can be passed to v2 functions like
// [jsonv2.Marshal] or [jsonv2.Unmarshal].
// Instead of referencing this type, use [jsonv2.Options].
//
// See the "Migrating to v2" section for guidance on how to migrate usage
// of "json" from using v1 to using v2 instead.
type Options = jsonopts.Options

// DefaultOptionsV1 is the full set of all options that define v1 semantics.
// It is equivalent to the following boolean options being set to true:
//
//   - [CallMethodsWithLegacySemantics]
//   - [EscapeInvalidUTF8]
//   - [FormatBytesWithLegacySemantics]
//   - [FormatTimeWithLegacySemantics]
//   - [MatchCaseSensitiveDelimiter]
//   - [MergeWithLegacySemantics]
//   - [OmitEmptyWithLegacyDefinition]
//   - [ReportErrorsWithLegacySemantics]
//   - [StringifyWithLegacySemantics]
//   - [UnmarshalArrayFromAnyLength]
//   - [jsonv2.Deterministic]
//   - [jsonv2.FormatNilMapAsNull]
//   - [jsonv2.FormatNilSliceAsNull]
//   - [jsonv2.MatchCaseInsensitiveNames]
//   - [jsontext.AllowDuplicateNames]
//   - [jsontext.AllowInvalidUTF8]
//   - [jsontext.EscapeForHTML]
//   - [jsontext.EscapeForJS]
//   - [jsontext.PreserveRawStrings]
//
// All other boolean options are set to false.
// All non-boolean options are set to the zero value,
// except for [jsontext.WithIndent], which defaults to "\t".
//
// The [Marshal] and [Unmarshal] functions in this package are
// semantically identical to calling the v2 equivalents with this option:
//
//	jsonv2.Marshal(v, jsonv1.DefaultOptionsV1())
//	jsonv2.Unmarshal(b, v, jsonv1.DefaultOptionsV1())
func DefaultOptionsV1() Options {
	return &jsonopts.DefaultOptionsV1
}

// CallMethodsWithLegacySemantics specifies that calling of type-provided
// marshal and unmarshal methods follow legacy semantics:
//
//   - When marshaling, a marshal method declared on a pointer receiver
//     is only called if the Go value is addressable.
//     Values obtained from an interface or map element are not addressable.
//     Values obtained from a pointer or slice element are addressable.
//     Values obtained from an array element or struct field inherit
//     the addressability of the parent. In contrast, the v2 semantic
//     is to always call marshal methods regardless of addressability.
//
//   - When marshaling or unmarshaling, the [Marshaler] or [Unmarshaler]
//     methods are ignored for map keys. However, [encoding.TextMarshaler]
//     or [encoding.TextUnmarshaler] are still callable.
//     In contrast, the v2 semantic is to serialize map keys
//     like any other value (with regard to calling methods),
//     which may include calling [Marshaler] or [Unmarshaler] methods,
//     where it is the implementation's responsibility to represent the
//     Go value as a JSON string (as required for JSON object names).
//
//   - When marshaling, if a map key value implements a marshal method
//     and is a nil pointer, then it is serialized as an empty JSON string.
//     In contrast, the v2 semantic is to report an error.
//
//   - When marshaling, if an interface type implements a marshal method
//     and the interface value is a nil pointer to a concrete type,
//     then the marshal method is always called.
//     In contrast, the v2 semantic is to never directly call methods
//     on interface values and to instead defer evaluation based upon
//     the underlying concrete value. Similar to non-interface values,
//     marshal methods are not called on nil pointers and
//     are instead serialized as a JSON null.
//
// This affects either marshaling or unmarshaling.
// The v1 default is true.
func CallMethodsWithLegacySemantics(v bool) Options {
	if v {
		return jsonflags.CallMethodsWithLegacySemantics | 1
	} else {
		return jsonflags.CallMethodsWithLegacySemantics | 0
	}
}

// EscapeInvalidUTF8 specifies that when encoding a [jsontext.String]
// with bytes of invalid UTF-8, such bytes are escaped as
// a hexadecimal Unicode codepoint (i.e., \ufffd).
// In contrast, the v2 default is to use the minimal representation,
// which is to encode invalid UTF-8 as the Unicode replacement rune itself
// (without any form of escaping).
//
// This only affects encoding and is ignored when decoding.
// The v1 default is true.
func EscapeInvalidUTF8(v bool) Options {
	if v {
		return jsonflags.EscapeInvalidUTF8 | 1
	} else {
		return jsonflags.EscapeInvalidUTF8 | 0
	}
}

// FormatBytesWithLegacySemantics specifies that handling of
// []~byte and [N]~byte types follow legacy semantics:
//
//   - A Go [N]~byte is always treated as as a normal Go array
//     in contrast to the v2 default of treating [N]byte as
//     using some form of binary data encoding (RFC 4648).
//
//   - A Go []~byte is to be treated as using some form of
//     binary data encoding (RFC 4648) in contrast to the v2 default
//     of only treating []byte as such. In particular, v2 does not
//     treat slices of named byte types as representing binary data.
//
//   - When marshaling, if a named byte implements a marshal method,
//     then the slice is serialized as a JSON array of elements,
//     each of which call the marshal method.
//
//   - When unmarshaling, if the input is a JSON array,
//     then unmarshal into the []~byte as if it were a normal Go slice.
//     In contrast, the v2 default is to report an error unmarshaling
//     a JSON array when expecting some form of binary data encoding.
//
//   - When unmarshaling, '\r' and '\n' characters are ignored
//     within the encoded "base32" and "base64" data.
//     In contrast, the v2 default is to report an error in order to be
//     strictly compliant with RFC 4648, section 3.3,
//     which specifies that non-alphabet characters must be rejected.
//
// This affects either marshaling or unmarshaling.
// The v1 default is true.
func FormatBytesWithLegacySemantics(v bool) Options {
	if v {
		return jsonflags.FormatBytesWithLegacySemantics | 1
	} else {
		return jsonflags.FormatBytesWithLegacySemantics | 0
	}
}

// FormatTimeWithLegacySemantics specifies that [time] types are formatted
// with legacy semantics:
//
//   - When marshaling or unmarshaling, a [time.Duration] is formatted as
//     a JSON number representing the number of nanoseconds.
//     In contrast, the default v2 behavior uses a JSON string
//     with the duration formatted with [time.Duration.String].
//     If a duration field has a `format` tag option,
//     then the specified formatting takes precedence.
//
//   - When unmarshaling, a [time.Time] follows loose adherence to RFC 3339.
//     In particular, it permits historically incorrect representations,
//     allowing for deviations in hour format, sub-second separator,
//     and timezone representation. In contrast, the default v2 behavior
//     is to strictly comply with the grammar specified in RFC 3339.
//
// This affects either marshaling or unmarshaling.
// The v1 default is true.
func FormatTimeWithLegacySemantics(v bool) Options {
	if v {
		return jsonflags.FormatTimeWithLegacySemantics | 1
	} else {
		return jsonflags.FormatTimeWithLegacySemantics | 0
	}
}

// MatchCaseSensitiveDelimiter specifies that underscores and dashes are
// not to be ignored when performing case-insensitive name matching which
// occurs under [jsonv2.MatchCaseInsensitiveNames] or the `case:ignore` tag option.
// Thus, case-insensitive name matching is identical to [strings.EqualFold].
// Use of this option diminishes the ability of case-insensitive matching
// to be able to match common case variants (e.g, "foo_bar" with "fooBar").
//
// This affects either marshaling or unmarshaling.
// The v1 default is true.
func MatchCaseSensitiveDelimiter(v bool) Options {
	if v {
		return jsonflags.MatchCaseSensitiveDelimiter | 1
	} else {
		return jsonflags.MatchCaseSensitiveDelimiter | 0
	}
}

// MergeWithLegacySemantics specifies that unmarshaling into a non-zero
// Go value follows legacy semantics:
//
//   - When unmarshaling a JSON null, this preserves the original Go value
//     if the kind is a bool, int, uint, float, string, array, or struct.
//     Otherwise, it zeros the Go value.
//     In contrast, the default v2 behavior is to consistently and always
//     zero the Go value when unmarshaling a JSON null into it.
//
//   - When unmarshaling a JSON value other than null, this merges into
//     the original Go value for array elements, slice elements,
//     struct fields (but not map values),
//     pointer values, and interface values (only if a non-nil pointer).
//     In contrast, the default v2 behavior is to merge into the Go value
//     for struct fields, map values, pointer values, and interface values.
//     In general, the v2 semantic merges when unmarshaling a JSON object,
//     otherwise it replaces the original value.
//
// This only affects unmarshaling and is ignored when marshaling.
// The v1 default is true.
func MergeWithLegacySemantics(v bool) Options {
	if v {
		return jsonflags.MergeWithLegacySemantics | 1
	} else {
		return jsonflags.MergeWithLegacySemantics | 0
	}
}

// OmitEmptyWithLegacyDefinition specifies that the `omitempty` tag option
// follows a definition of empty where a field is omitted if the Go value is
// false, 0, a nil pointer, a nil interface value,
// or any empty array, slice, map, or string.
// This overrides the v2 semantic where a field is empty if the value
// marshals as a JSON null or an empty JSON string, object, or array.
//
// The v1 and v2 definitions of `omitempty` are practically the same for
// Go strings, slices, arrays, and maps. Usages of `omitempty` on
// Go bools, ints, uints floats, pointers, and interfaces should migrate to use
// the `omitzero` tag option, which omits a field if it is the zero Go value.
//
// This only affects marshaling and is ignored when unmarshaling.
// The v1 default is true.
func OmitEmptyWithLegacyDefinition(v bool) Options {
	if v {
		return jsonflags.OmitEmptyWithLegacyDefinition | 1
	} else {
		return jsonflags.OmitEmptyWithLegacyDefinition | 0
	}
}

// ReportErrorsWithLegacySemantics specifies that Marshal and Unmarshal
// should report errors with legacy semantics:
//
//   - When marshaling or unmarshaling, the returned error values are
//     usually of types such as [SyntaxError], [MarshalerError],
//     [UnsupportedTypeError], [UnsupportedValueError],
//     [InvalidUnmarshalError], or [UnmarshalTypeError].
//     In contrast, the v2 semantic is to always return errors as either
//     [jsonv2.SemanticError] or [jsontext.SyntacticError].
//
//   - When marshaling, if a user-defined marshal method reports an error,
//     it is always wrapped in a [MarshalerError], even if the error itself
//     is already a [MarshalerError], which may lead to multiple redundant
//     layers of wrapping. In contrast, the v2 semantic is to
//     always wrap an error within [jsonv2.SemanticError]
//     unless it is already a semantic error.
//
//   - When unmarshaling, if a user-defined unmarshal method reports an error,
//     it is never wrapped and reported verbatim. In contrast, the v2 semantic
//     is to always wrap an error within [jsonv2.SemanticError]
//     unless it is already a semantic error.
//
//   - When marshaling or unmarshaling, if a Go struct contains type errors
//     (e.g., conflicting names or malformed field tags), then such errors
//     are ignored and the Go struct uses a best-effort representation.
//     In contrast, the v2 semantic is to report a runtime error.
//
//   - When unmarshaling, the syntactic structure of the JSON input
//     is fully validated before performing the semantic unmarshaling
//     of the JSON data into the Go value. Practically speaking,
//     this means that JSON input with syntactic errors do not result
//     in any mutations of the target Go value. In contrast, the v2 semantic
//     is to perform a streaming decode and gradually unmarshal the JSON input
//     into the target Go value, which means that the Go value may be
//     partially mutated when a syntactic error is encountered.
//
//   - When unmarshaling, a semantic error does not immediately terminate the
//     unmarshal procedure, but rather evaluation continues.
//     When unmarshal returns, only the first semantic error is reported.
//     In contrast, the v2 semantic is to terminate unmarshal the moment
//     an error is encountered.
//
// This affects either marshaling or unmarshaling.
// The v1 default is true.
func ReportErrorsWithLegacySemantics(v bool) Options {
	if v {
		return jsonflags.ReportErrorsWithLegacySemantics | 1
	} else {
		return jsonflags.ReportErrorsWithLegacySemantics | 0
	}
}

// StringifyWithLegacySemantics specifies that the `string` tag option
// may stringify bools and string values. It only takes effect on fields
// where the top-level type is a bool, string, numeric kind, or a pointer to
// such a kind. Specifically, `string` will not stringify bool, string,
// or numeric kinds within a composite data type
// (e.g., array, slice, struct, map, or interface).
//
// When marshaling, such Go values are serialized as their usual
// JSON representation, but quoted within a JSON string.
// When unmarshaling, such Go values must be deserialized from
// a JSON string containing their usual JSON representation.
// A JSON null quoted in a JSON string is a valid substitute for JSON null
// while unmarshaling into a Go value that `string` takes effect on.
//
// This affects either marshaling or unmarshaling.
// The v1 default is true.
func StringifyWithLegacySemantics(v bool) Options {
	if v {
		return jsonflags.StringifyWithLegacySemantics | 1
	} else {
		return jsonflags.StringifyWithLegacySemantics | 0
	}
}

// UnmarshalArrayFromAnyLength specifies that Go arrays can be unmarshaled
// from input JSON arrays of any length. If the JSON array is too short,
// then the remaining Go array elements are zeroed. If the JSON array
// is too long, then the excess JSON array elements are skipped over.
//
// This only affects unmarshaling and is ignored when marshaling.
// The v1 default is true.
func UnmarshalArrayFromAnyLength(v bool) Options {
	if v {
		return jsonflags.UnmarshalArrayFromAnyLength | 1
	} else {
		return jsonflags.UnmarshalArrayFromAnyLength | 0
	}
}

// unmarshalAnyWithRawNumber specifies that unmarshaling a JSON number into
// an empty Go interface should use the Number type instead of a float64.
func unmarshalAnyWithRawNumber(v bool) Options {
	if v {
		return jsonflags.UnmarshalAnyWithRawNumber | 1
	} else {
		return jsonflags.UnmarshalAnyWithRawNumber | 0
	}
}
