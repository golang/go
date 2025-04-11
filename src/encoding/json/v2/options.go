// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.jsonv2

package json

import (
	"fmt"

	"encoding/json/internal"
	"encoding/json/internal/jsonflags"
	"encoding/json/internal/jsonopts"
)

// Options configure [Marshal], [MarshalWrite], [MarshalEncode],
// [Unmarshal], [UnmarshalRead], and [UnmarshalDecode] with specific features.
// Each function takes in a variadic list of options, where properties
// set in later options override the value of previously set properties.
//
// The Options type is identical to [encoding/json.Options] and
// [encoding/json/jsontext.Options]. Options from the other packages can
// be used interchangeably with functionality in this package.
//
// Options represent either a singular option or a set of options.
// It can be functionally thought of as a Go map of option properties
// (even though the underlying implementation avoids Go maps for performance).
//
// The constructors (e.g., [Deterministic]) return a singular option value:
//
//	opt := Deterministic(true)
//
// which is analogous to creating a single entry map:
//
//	opt := Options{"Deterministic": true}
//
// [JoinOptions] composes multiple options values to together:
//
//	out := JoinOptions(opts...)
//
// which is analogous to making a new map and copying the options over:
//
//	out := make(Options)
//	for _, m := range opts {
//		for k, v := range m {
//			out[k] = v
//		}
//	}
//
// [GetOption] looks up the value of options parameter:
//
//	v, ok := GetOption(opts, Deterministic)
//
// which is analogous to a Go map lookup:
//
//	v, ok := Options["Deterministic"]
//
// There is a single Options type, which is used with both marshal and unmarshal.
// Some options affect both operations, while others only affect one operation:
//
//   - [StringifyNumbers] affects marshaling and unmarshaling
//   - [Deterministic] affects marshaling only
//   - [FormatNilSliceAsNull] affects marshaling only
//   - [FormatNilMapAsNull] affects marshaling only
//   - [OmitZeroStructFields] affects marshaling only
//   - [MatchCaseInsensitiveNames] affects marshaling and unmarshaling
//   - [DiscardUnknownMembers] affects marshaling only
//   - [RejectUnknownMembers] affects unmarshaling only
//   - [WithMarshalers] affects marshaling only
//   - [WithUnmarshalers] affects unmarshaling only
//
// Options that do not affect a particular operation are ignored.
type Options = jsonopts.Options

// JoinOptions coalesces the provided list of options into a single Options.
// Properties set in later options override the value of previously set properties.
func JoinOptions(srcs ...Options) Options {
	var dst jsonopts.Struct
	dst.Join(srcs...)
	return &dst
}

// GetOption returns the value stored in opts with the provided setter,
// reporting whether the value is present.
//
// Example usage:
//
//	v, ok := json.GetOption(opts, json.Deterministic)
//
// Options are most commonly introspected to alter the JSON representation of
// [MarshalerTo.MarshalJSONTo] and [UnmarshalerFrom.UnmarshalJSONFrom] methods, and
// [MarshalToFunc] and [UnmarshalFromFunc] functions.
// In such cases, the presence bit should generally be ignored.
func GetOption[T any](opts Options, setter func(T) Options) (T, bool) {
	return jsonopts.GetOption(opts, setter)
}

// DefaultOptionsV2 is the full set of all options that define v2 semantics.
// It is equivalent to all options under [Options], [encoding/json.Options],
// and [encoding/json/jsontext.Options] being set to false or the zero value,
// except for the options related to whitespace formatting.
func DefaultOptionsV2() Options {
	return &jsonopts.DefaultOptionsV2
}

// StringifyNumbers specifies that numeric Go types should be marshaled
// as a JSON string containing the equivalent JSON number value.
// When unmarshaling, numeric Go types are parsed from a JSON string
// containing the JSON number without any surrounding whitespace.
//
// According to RFC 8259, section 6, a JSON implementation may choose to
// limit the representation of a JSON number to an IEEE 754 binary64 value.
// This may cause decoders to lose precision for int64 and uint64 types.
// Quoting JSON numbers as a JSON string preserves the exact precision.
//
// This affects either marshaling or unmarshaling.
func StringifyNumbers(v bool) Options {
	if v {
		return jsonflags.StringifyNumbers | 1
	} else {
		return jsonflags.StringifyNumbers | 0
	}
}

// Deterministic specifies that the same input value will be serialized
// as the exact same output bytes. Different processes of
// the same program will serialize equal values to the same bytes,
// but different versions of the same program are not guaranteed
// to produce the exact same sequence of bytes.
//
// This only affects marshaling and is ignored when unmarshaling.
func Deterministic(v bool) Options {
	if v {
		return jsonflags.Deterministic | 1
	} else {
		return jsonflags.Deterministic | 0
	}
}

// FormatNilSliceAsNull specifies that a nil Go slice should marshal as a
// JSON null instead of the default representation as an empty JSON array
// (or an empty JSON string in the case of ~[]byte).
// Slice fields explicitly marked with `format:emitempty` still marshal
// as an empty JSON array.
//
// This only affects marshaling and is ignored when unmarshaling.
func FormatNilSliceAsNull(v bool) Options {
	if v {
		return jsonflags.FormatNilSliceAsNull | 1
	} else {
		return jsonflags.FormatNilSliceAsNull | 0
	}
}

// FormatNilMapAsNull specifies that a nil Go map should marshal as a
// JSON null instead of the default representation as an empty JSON object.
// Map fields explicitly marked with `format:emitempty` still marshal
// as an empty JSON object.
//
// This only affects marshaling and is ignored when unmarshaling.
func FormatNilMapAsNull(v bool) Options {
	if v {
		return jsonflags.FormatNilMapAsNull | 1
	} else {
		return jsonflags.FormatNilMapAsNull | 0
	}
}

// OmitZeroStructFields specifies that a Go struct should marshal in such a way
// that all struct fields that are zero are omitted from the marshaled output
// if the value is zero as determined by the "IsZero() bool" method if present,
// otherwise based on whether the field is the zero Go value.
// This is semantically equivalent to specifying the `omitzero` tag option
// on every field in a Go struct.
//
// This only affects marshaling and is ignored when unmarshaling.
func OmitZeroStructFields(v bool) Options {
	if v {
		return jsonflags.OmitZeroStructFields | 1
	} else {
		return jsonflags.OmitZeroStructFields | 0
	}
}

// MatchCaseInsensitiveNames specifies that JSON object members are matched
// against Go struct fields using a case-insensitive match of the name.
// Go struct fields explicitly marked with `case:strict` or `case:ignore`
// always use case-sensitive (or case-insensitive) name matching,
// regardless of the value of this option.
//
// This affects either marshaling or unmarshaling.
// For marshaling, this option may alter the detection of duplicate names
// (assuming [jsontext.AllowDuplicateNames] is false) from inlined fields
// if it matches one of the declared fields in the Go struct.
func MatchCaseInsensitiveNames(v bool) Options {
	if v {
		return jsonflags.MatchCaseInsensitiveNames | 1
	} else {
		return jsonflags.MatchCaseInsensitiveNames | 0
	}
}

// DiscardUnknownMembers specifies that marshaling should ignore any
// JSON object members stored in Go struct fields dedicated to storing
// unknown JSON object members.
//
// This only affects marshaling and is ignored when unmarshaling.
func DiscardUnknownMembers(v bool) Options {
	if v {
		return jsonflags.DiscardUnknownMembers | 1
	} else {
		return jsonflags.DiscardUnknownMembers | 0
	}
}

// RejectUnknownMembers specifies that unknown members should be rejected
// when unmarshaling a JSON object, regardless of whether there is a field
// to store unknown members.
//
// This only affects unmarshaling and is ignored when marshaling.
func RejectUnknownMembers(v bool) Options {
	if v {
		return jsonflags.RejectUnknownMembers | 1
	} else {
		return jsonflags.RejectUnknownMembers | 0
	}
}

// WithMarshalers specifies a list of type-specific marshalers to use,
// which can be used to override the default marshal behavior for values
// of particular types.
//
// This only affects marshaling and is ignored when unmarshaling.
func WithMarshalers(v *Marshalers) Options {
	return (*marshalersOption)(v)
}

// WithUnmarshalers specifies a list of type-specific unmarshalers to use,
// which can be used to override the default unmarshal behavior for values
// of particular types.
//
// This only affects unmarshaling and is ignored when marshaling.
func WithUnmarshalers(v *Unmarshalers) Options {
	return (*unmarshalersOption)(v)
}

// These option types are declared here instead of "jsonopts"
// to avoid a dependency on "reflect" from "jsonopts".
type (
	marshalersOption   Marshalers
	unmarshalersOption Unmarshalers
)

func (*marshalersOption) JSONOptions(internal.NotForPublicUse)   {}
func (*unmarshalersOption) JSONOptions(internal.NotForPublicUse) {}

// Inject support into "jsonopts" to handle these types.
func init() {
	jsonopts.GetUnknownOption = func(src *jsonopts.Struct, zero jsonopts.Options) (any, bool) {
		switch zero.(type) {
		case *marshalersOption:
			if !src.Flags.Has(jsonflags.Marshalers) {
				return (*Marshalers)(nil), false
			}
			return src.Marshalers.(*Marshalers), true
		case *unmarshalersOption:
			if !src.Flags.Has(jsonflags.Unmarshalers) {
				return (*Unmarshalers)(nil), false
			}
			return src.Unmarshalers.(*Unmarshalers), true
		default:
			panic(fmt.Sprintf("unknown option %T", zero))
		}
	}
	jsonopts.JoinUnknownOption = func(dst *jsonopts.Struct, src jsonopts.Options) {
		switch src := src.(type) {
		case *marshalersOption:
			dst.Flags.Set(jsonflags.Marshalers | 1)
			dst.Marshalers = (*Marshalers)(src)
		case *unmarshalersOption:
			dst.Flags.Set(jsonflags.Unmarshalers | 1)
			dst.Unmarshalers = (*Unmarshalers)(src)
		default:
			panic(fmt.Sprintf("unknown option %T", src))
		}
	}
}
