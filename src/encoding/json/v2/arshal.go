// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.jsonv2

package json

import (
	"bytes"
	"encoding"
	"io"
	"reflect"
	"slices"
	"strings"
	"sync"
	"time"

	"encoding/json/internal"
	"encoding/json/internal/jsonflags"
	"encoding/json/internal/jsonopts"
	"encoding/json/jsontext"
)

// Reference encoding and time packages to assist pkgsite
// in being able to hotlink references to those packages.
var (
	_ encoding.TextMarshaler
	_ encoding.TextAppender
	_ encoding.TextUnmarshaler
	_ time.Time
	_ time.Duration
)

// export exposes internal functionality of the "jsontext" package.
var export = jsontext.Internal.Export(&internal.AllowInternalUse)

// Marshal serializes a Go value as a []byte according to the provided
// marshal and encode options (while ignoring unmarshal or decode options).
// It does not terminate the output with a newline.
//
// Type-specific marshal functions and methods take precedence
// over the default representation of a value.
// Functions or methods that operate on *T are only called when encoding
// a value of type T (by taking its address) or a non-nil value of *T.
// Marshal ensures that a value is always addressable
// (by boxing it on the heap if necessary) so that
// these functions and methods can be consistently called. For performance,
// it is recommended that Marshal be passed a non-nil pointer to the value.
//
// The input value is encoded as JSON according the following rules:
//
//   - If any type-specific functions in a [WithMarshalers] option match
//     the value type, then those functions are called to encode the value.
//     If all applicable functions return [SkipFunc],
//     then the value is encoded according to subsequent rules.
//
//   - If the value type implements [MarshalerTo],
//     then the MarshalJSONTo method is called to encode the value.
//
//   - If the value type implements [Marshaler],
//     then the MarshalJSON method is called to encode the value.
//
//   - If the value type implements [encoding.TextAppender],
//     then the AppendText method is called to encode the value and
//     subsequently encode its result as a JSON string.
//
//   - If the value type implements [encoding.TextMarshaler],
//     then the MarshalText method is called to encode the value and
//     subsequently encode its result as a JSON string.
//
//   - Otherwise, the value is encoded according to the value's type
//     as described in detail below.
//
// Most Go types have a default JSON representation.
// Certain types support specialized formatting according to
// a format flag optionally specified in the Go struct tag
// for the struct field that contains the current value
// (see the “JSON Representation of Go structs” section for more details).
//
// The representation of each type is as follows:
//
//   - A Go boolean is encoded as a JSON boolean (e.g., true or false).
//     It does not support any custom format flags.
//
//   - A Go string is encoded as a JSON string.
//     It does not support any custom format flags.
//
//   - A Go []byte or [N]byte is encoded as a JSON string containing
//     the binary value encoded using RFC 4648.
//     If the format is "base64" or unspecified, then this uses RFC 4648, section 4.
//     If the format is "base64url", then this uses RFC 4648, section 5.
//     If the format is "base32", then this uses RFC 4648, section 6.
//     If the format is "base32hex", then this uses RFC 4648, section 7.
//     If the format is "base16" or "hex", then this uses RFC 4648, section 8.
//     If the format is "array", then the bytes value is encoded as a JSON array
//     where each byte is recursively JSON-encoded as each JSON array element.
//
//   - A Go integer is encoded as a JSON number without fractions or exponents.
//     If [StringifyNumbers] is specified or encoding a JSON object name,
//     then the JSON number is encoded within a JSON string.
//     It does not support any custom format flags.
//
//   - A Go float is encoded as a JSON number.
//     If [StringifyNumbers] is specified or encoding a JSON object name,
//     then the JSON number is encoded within a JSON string.
//     If the format is "nonfinite", then NaN, +Inf, and -Inf are encoded as
//     the JSON strings "NaN", "Infinity", and "-Infinity", respectively.
//     Otherwise, the presence of non-finite numbers results in a [SemanticError].
//
//   - A Go map is encoded as a JSON object, where each Go map key and value
//     is recursively encoded as a name and value pair in the JSON object.
//     The Go map key must encode as a JSON string, otherwise this results
//     in a [SemanticError]. The Go map is traversed in a non-deterministic order.
//     For deterministic encoding, consider using the [Deterministic] option.
//     If the format is "emitnull", then a nil map is encoded as a JSON null.
//     If the format is "emitempty", then a nil map is encoded as an empty JSON object,
//     regardless of whether [FormatNilMapAsNull] is specified.
//     Otherwise by default, a nil map is encoded as an empty JSON object.
//
//   - A Go struct is encoded as a JSON object.
//     See the “JSON Representation of Go structs” section
//     in the package-level documentation for more details.
//
//   - A Go slice is encoded as a JSON array, where each Go slice element
//     is recursively JSON-encoded as the elements of the JSON array.
//     If the format is "emitnull", then a nil slice is encoded as a JSON null.
//     If the format is "emitempty", then a nil slice is encoded as an empty JSON array,
//     regardless of whether [FormatNilSliceAsNull] is specified.
//     Otherwise by default, a nil slice is encoded as an empty JSON array.
//
//   - A Go array is encoded as a JSON array, where each Go array element
//     is recursively JSON-encoded as the elements of the JSON array.
//     The JSON array length is always identical to the Go array length.
//     It does not support any custom format flags.
//
//   - A Go pointer is encoded as a JSON null if nil, otherwise it is
//     the recursively JSON-encoded representation of the underlying value.
//     Format flags are forwarded to the encoding of the underlying value.
//
//   - A Go interface is encoded as a JSON null if nil, otherwise it is
//     the recursively JSON-encoded representation of the underlying value.
//     It does not support any custom format flags.
//
//   - A Go [time.Time] is encoded as a JSON string containing the timestamp
//     formatted in RFC 3339 with nanosecond precision.
//     If the format matches one of the format constants declared
//     in the time package (e.g., RFC1123), then that format is used.
//     If the format is "unix", "unixmilli", "unixmicro", or "unixnano",
//     then the timestamp is encoded as a possibly fractional JSON number
//     of the number of seconds (or milliseconds, microseconds, or nanoseconds)
//     since the Unix epoch, which is January 1st, 1970 at 00:00:00 UTC.
//     To avoid a fractional component, round the timestamp to the relevant unit.
//     Otherwise, the format is used as-is with [time.Time.Format] if non-empty.
//
//   - A Go [time.Duration] currently has no default representation and
//     requires an explicit format to be specified.
//     If the format is "sec", "milli", "micro", or "nano",
//     then the duration is encoded as a possibly fractional JSON number
//     of the number of seconds (or milliseconds, microseconds, or nanoseconds).
//     To avoid a fractional component, round the duration to the relevant unit.
//     If the format is "units", it is encoded as a JSON string formatted using
//     [time.Duration.String] (e.g., "1h30m" for 1 hour 30 minutes).
//     If the format is "iso8601", it is encoded as a JSON string using the
//     ISO 8601 standard for durations (e.g., "PT1H30M" for 1 hour 30 minutes)
//     using only accurate units of hours, minutes, and seconds.
//
//   - All other Go types (e.g., complex numbers, channels, and functions)
//     have no default representation and result in a [SemanticError].
//
// JSON cannot represent cyclic data structures and Marshal does not handle them.
// Passing cyclic structures will result in an error.
func Marshal(in any, opts ...Options) (out []byte, err error) {
	enc := export.GetBufferedEncoder(opts...)
	defer export.PutBufferedEncoder(enc)
	xe := export.Encoder(enc)
	xe.Flags.Set(jsonflags.OmitTopLevelNewline | 1)
	err = marshalEncode(enc, in, &xe.Struct)
	if err != nil && xe.Flags.Get(jsonflags.ReportErrorsWithLegacySemantics) {
		return nil, internal.TransformMarshalError(in, err)
	}
	return bytes.Clone(xe.Buf), err
}

// MarshalWrite serializes a Go value into an [io.Writer] according to the provided
// marshal and encode options (while ignoring unmarshal or decode options).
// It does not terminate the output with a newline.
// See [Marshal] for details about the conversion of a Go value into JSON.
func MarshalWrite(out io.Writer, in any, opts ...Options) (err error) {
	enc := export.GetStreamingEncoder(out, opts...)
	defer export.PutStreamingEncoder(enc)
	xe := export.Encoder(enc)
	xe.Flags.Set(jsonflags.OmitTopLevelNewline | 1)
	err = marshalEncode(enc, in, &xe.Struct)
	if err != nil && xe.Flags.Get(jsonflags.ReportErrorsWithLegacySemantics) {
		return internal.TransformMarshalError(in, err)
	}
	return err
}

// MarshalEncode serializes a Go value into an [jsontext.Encoder] according to
// the provided marshal options (while ignoring unmarshal, encode, or decode options).
// Any marshal-relevant options already specified on the [jsontext.Encoder]
// take lower precedence than the set of options provided by the caller.
// Unlike [Marshal] and [MarshalWrite], encode options are ignored because
// they must have already been specified on the provided [jsontext.Encoder].
//
// See [Marshal] for details about the conversion of a Go value into JSON.
func MarshalEncode(out *jsontext.Encoder, in any, opts ...Options) (err error) {
	xe := export.Encoder(out)
	if len(opts) > 0 {
		optsOriginal := xe.Struct
		defer func() { xe.Struct = optsOriginal }()
		xe.Struct.JoinWithoutCoderOptions(opts...)
	}
	err = marshalEncode(out, in, &xe.Struct)
	if err != nil && xe.Flags.Get(jsonflags.ReportErrorsWithLegacySemantics) {
		return internal.TransformMarshalError(in, err)
	}
	return err
}

func marshalEncode(out *jsontext.Encoder, in any, mo *jsonopts.Struct) (err error) {
	v := reflect.ValueOf(in)
	if !v.IsValid() || (v.Kind() == reflect.Pointer && v.IsNil()) {
		return out.WriteToken(jsontext.Null)
	}
	// Shallow copy non-pointer values to obtain an addressable value.
	// It is beneficial to performance to always pass pointers to avoid this.
	forceAddr := v.Kind() != reflect.Pointer
	if forceAddr {
		v2 := reflect.New(v.Type())
		v2.Elem().Set(v)
		v = v2
	}
	va := addressableValue{v.Elem(), forceAddr} // dereferenced pointer is always addressable
	t := va.Type()

	// Lookup and call the marshal function for this type.
	marshal := lookupArshaler(t).marshal
	if mo.Marshalers != nil {
		marshal, _ = mo.Marshalers.(*Marshalers).lookup(marshal, t)
	}
	if err := marshal(out, va, mo); err != nil {
		if !mo.Flags.Get(jsonflags.AllowDuplicateNames) {
			export.Encoder(out).Tokens.InvalidateDisabledNamespaces()
		}
		return err
	}
	return nil
}

// Unmarshal decodes a []byte input into a Go value according to the provided
// unmarshal and decode options (while ignoring marshal or encode options).
// The input must be a single JSON value with optional whitespace interspersed.
// The output must be a non-nil pointer.
//
// Type-specific unmarshal functions and methods take precedence
// over the default representation of a value.
// Functions or methods that operate on *T are only called when decoding
// a value of type T (by taking its address) or a non-nil value of *T.
// Unmarshal ensures that a value is always addressable
// (by boxing it on the heap if necessary) so that
// these functions and methods can be consistently called.
//
// The input is decoded into the output according the following rules:
//
//   - If any type-specific functions in a [WithUnmarshalers] option match
//     the value type, then those functions are called to decode the JSON
//     value. If all applicable functions return [SkipFunc],
//     then the input is decoded according to subsequent rules.
//
//   - If the value type implements [UnmarshalerFrom],
//     then the UnmarshalJSONFrom method is called to decode the JSON value.
//
//   - If the value type implements [Unmarshaler],
//     then the UnmarshalJSON method is called to decode the JSON value.
//
//   - If the value type implements [encoding.TextUnmarshaler],
//     then the input is decoded as a JSON string and
//     the UnmarshalText method is called with the decoded string value.
//     This fails with a [SemanticError] if the input is not a JSON string.
//
//   - Otherwise, the JSON value is decoded according to the value's type
//     as described in detail below.
//
// Most Go types have a default JSON representation.
// Certain types support specialized formatting according to
// a format flag optionally specified in the Go struct tag
// for the struct field that contains the current value
// (see the “JSON Representation of Go structs” section for more details).
// A JSON null may be decoded into every supported Go value where
// it is equivalent to storing the zero value of the Go value.
// If the input JSON kind is not handled by the current Go value type,
// then this fails with a [SemanticError]. Unless otherwise specified,
// the decoded value replaces any pre-existing value.
//
// The representation of each type is as follows:
//
//   - A Go boolean is decoded from a JSON boolean (e.g., true or false).
//     It does not support any custom format flags.
//
//   - A Go string is decoded from a JSON string.
//     It does not support any custom format flags.
//
//   - A Go []byte or [N]byte is decoded from a JSON string
//     containing the binary value encoded using RFC 4648.
//     If the format is "base64" or unspecified, then this uses RFC 4648, section 4.
//     If the format is "base64url", then this uses RFC 4648, section 5.
//     If the format is "base32", then this uses RFC 4648, section 6.
//     If the format is "base32hex", then this uses RFC 4648, section 7.
//     If the format is "base16" or "hex", then this uses RFC 4648, section 8.
//     If the format is "array", then the Go slice or array is decoded from a
//     JSON array where each JSON element is recursively decoded for each byte.
//     When decoding into a non-nil []byte, the slice length is reset to zero
//     and the decoded input is appended to it.
//     When decoding into a [N]byte, the input must decode to exactly N bytes,
//     otherwise it fails with a [SemanticError].
//
//   - A Go integer is decoded from a JSON number.
//     It must be decoded from a JSON string containing a JSON number
//     if [StringifyNumbers] is specified or decoding a JSON object name.
//     It fails with a [SemanticError] if the JSON number
//     has a fractional or exponent component.
//     It also fails if it overflows the representation of the Go integer type.
//     It does not support any custom format flags.
//
//   - A Go float is decoded from a JSON number.
//     It must be decoded from a JSON string containing a JSON number
//     if [StringifyNumbers] is specified or decoding a JSON object name.
//     It fails if it overflows the representation of the Go float type.
//     If the format is "nonfinite", then the JSON strings
//     "NaN", "Infinity", and "-Infinity" are decoded as NaN, +Inf, and -Inf.
//     Otherwise, the presence of such strings results in a [SemanticError].
//
//   - A Go map is decoded from a JSON object,
//     where each JSON object name and value pair is recursively decoded
//     as the Go map key and value. Maps are not cleared.
//     If the Go map is nil, then a new map is allocated to decode into.
//     If the decoded key matches an existing Go map entry, the entry value
//     is reused by decoding the JSON object value into it.
//     The formats "emitnull" and "emitempty" have no effect when decoding.
//
//   - A Go struct is decoded from a JSON object.
//     See the “JSON Representation of Go structs” section
//     in the package-level documentation for more details.
//
//   - A Go slice is decoded from a JSON array, where each JSON element
//     is recursively decoded and appended to the Go slice.
//     Before appending into a Go slice, a new slice is allocated if it is nil,
//     otherwise the slice length is reset to zero.
//     The formats "emitnull" and "emitempty" have no effect when decoding.
//
//   - A Go array is decoded from a JSON array, where each JSON array element
//     is recursively decoded as each corresponding Go array element.
//     Each Go array element is zeroed before decoding into it.
//     It fails with a [SemanticError] if the JSON array does not contain
//     the exact same number of elements as the Go array.
//     It does not support any custom format flags.
//
//   - A Go pointer is decoded based on the JSON kind and underlying Go type.
//     If the input is a JSON null, then this stores a nil pointer.
//     Otherwise, it allocates a new underlying value if the pointer is nil,
//     and recursively JSON decodes into the underlying value.
//     Format flags are forwarded to the decoding of the underlying type.
//
//   - A Go interface is decoded based on the JSON kind and underlying Go type.
//     If the input is a JSON null, then this stores a nil interface value.
//     Otherwise, a nil interface value of an empty interface type is initialized
//     with a zero Go bool, string, float64, map[string]any, or []any if the
//     input is a JSON boolean, string, number, object, or array, respectively.
//     If the interface value is still nil, then this fails with a [SemanticError]
//     since decoding could not determine an appropriate Go type to decode into.
//     For example, unmarshaling into a nil io.Reader fails since
//     there is no concrete type to populate the interface value with.
//     Otherwise an underlying value exists and it recursively decodes
//     the JSON input into it. It does not support any custom format flags.
//
//   - A Go [time.Time] is decoded from a JSON string containing the time
//     formatted in RFC 3339 with nanosecond precision.
//     If the format matches one of the format constants declared in
//     the time package (e.g., RFC1123), then that format is used for parsing.
//     If the format is "unix", "unixmilli", "unixmicro", or "unixnano",
//     then the timestamp is decoded from an optionally fractional JSON number
//     of the number of seconds (or milliseconds, microseconds, or nanoseconds)
//     since the Unix epoch, which is January 1st, 1970 at 00:00:00 UTC.
//     Otherwise, the format is used as-is with [time.Time.Parse] if non-empty.
//
//   - A Go [time.Duration] currently has no default representation and
//     requires an explicit format to be specified.
//     If the format is "sec", "milli", "micro", or "nano",
//     then the duration is decoded from an optionally fractional JSON number
//     of the number of seconds (or milliseconds, microseconds, or nanoseconds).
//     If the format is "units", it is decoded from a JSON string parsed using
//     [time.ParseDuration] (e.g., "1h30m" for 1 hour 30 minutes).
//     If the format is "iso8601", it is decoded from a JSON string using the
//     ISO 8601 standard for durations (e.g., "PT1H30M" for 1 hour 30 minutes)
//     accepting only accurate units of hours, minutes, or seconds.
//
//   - All other Go types (e.g., complex numbers, channels, and functions)
//     have no default representation and result in a [SemanticError].
//
// In general, unmarshaling follows merge semantics (similar to RFC 7396)
// where the decoded Go value replaces the destination value
// for any JSON kind other than an object.
// For JSON objects, the input object is merged into the destination value
// where matching object members recursively apply merge semantics.
func Unmarshal(in []byte, out any, opts ...Options) (err error) {
	dec := export.GetBufferedDecoder(in, opts...)
	defer export.PutBufferedDecoder(dec)
	xd := export.Decoder(dec)
	err = unmarshalFull(dec, out, &xd.Struct)
	if err != nil && xd.Flags.Get(jsonflags.ReportErrorsWithLegacySemantics) {
		return internal.TransformUnmarshalError(out, err)
	}
	return err
}

// UnmarshalRead deserializes a Go value from an [io.Reader] according to the
// provided unmarshal and decode options (while ignoring marshal or encode options).
// The input must be a single JSON value with optional whitespace interspersed.
// It consumes the entirety of [io.Reader] until [io.EOF] is encountered,
// without reporting an error for EOF. The output must be a non-nil pointer.
// See [Unmarshal] for details about the conversion of JSON into a Go value.
func UnmarshalRead(in io.Reader, out any, opts ...Options) (err error) {
	dec := export.GetStreamingDecoder(in, opts...)
	defer export.PutStreamingDecoder(dec)
	xd := export.Decoder(dec)
	err = unmarshalFull(dec, out, &xd.Struct)
	if err != nil && xd.Flags.Get(jsonflags.ReportErrorsWithLegacySemantics) {
		return internal.TransformUnmarshalError(out, err)
	}
	return err
}

func unmarshalFull(in *jsontext.Decoder, out any, uo *jsonopts.Struct) error {
	switch err := unmarshalDecode(in, out, uo); err {
	case nil:
		return export.Decoder(in).CheckEOF()
	case io.EOF:
		offset := in.InputOffset() + int64(len(in.UnreadBuffer()))
		return &jsontext.SyntacticError{ByteOffset: offset, Err: io.ErrUnexpectedEOF}
	default:
		return err
	}
}

// UnmarshalDecode deserializes a Go value from a [jsontext.Decoder] according to
// the provided unmarshal options (while ignoring marshal, encode, or decode options).
// Any unmarshal options already specified on the [jsontext.Decoder]
// take lower precedence than the set of options provided by the caller.
// Unlike [Unmarshal] and [UnmarshalRead], decode options are ignored because
// they must have already been specified on the provided [jsontext.Decoder].
//
// The input may be a stream of one or more JSON values,
// where this only unmarshals the next JSON value in the stream.
// The output must be a non-nil pointer.
// See [Unmarshal] for details about the conversion of JSON into a Go value.
func UnmarshalDecode(in *jsontext.Decoder, out any, opts ...Options) (err error) {
	xd := export.Decoder(in)
	if len(opts) > 0 {
		optsOriginal := xd.Struct
		defer func() { xd.Struct = optsOriginal }()
		xd.Struct.JoinWithoutCoderOptions(opts...)
	}
	err = unmarshalDecode(in, out, &xd.Struct)
	if err != nil && xd.Flags.Get(jsonflags.ReportErrorsWithLegacySemantics) {
		return internal.TransformUnmarshalError(out, err)
	}
	return err
}

func unmarshalDecode(in *jsontext.Decoder, out any, uo *jsonopts.Struct) (err error) {
	v := reflect.ValueOf(out)
	if v.Kind() != reflect.Pointer || v.IsNil() {
		return &SemanticError{action: "unmarshal", GoType: reflect.TypeOf(out), Err: internal.ErrNonNilReference}
	}
	va := addressableValue{v.Elem(), false} // dereferenced pointer is always addressable
	t := va.Type()

	// In legacy semantics, the entirety of the next JSON value
	// was validated before attempting to unmarshal it.
	if uo.Flags.Get(jsonflags.ReportErrorsWithLegacySemantics) {
		if err := export.Decoder(in).CheckNextValue(); err != nil {
			return err
		}
	}

	// Lookup and call the unmarshal function for this type.
	unmarshal := lookupArshaler(t).unmarshal
	if uo.Unmarshalers != nil {
		unmarshal, _ = uo.Unmarshalers.(*Unmarshalers).lookup(unmarshal, t)
	}
	if err := unmarshal(in, va, uo); err != nil {
		if !uo.Flags.Get(jsonflags.AllowDuplicateNames) {
			export.Decoder(in).Tokens.InvalidateDisabledNamespaces()
		}
		return err
	}
	return nil
}

// addressableValue is a reflect.Value that is guaranteed to be addressable
// such that calling the Addr and Set methods do not panic.
//
// There is no compile magic that enforces this property,
// but rather the need to construct this type makes it easier to examine each
// construction site to ensure that this property is upheld.
type addressableValue struct {
	reflect.Value

	// forcedAddr reports whether this value is addressable
	// only through the use of [newAddressableValue].
	// This is only used for [jsonflags.CallMethodsWithLegacySemantics].
	forcedAddr bool
}

// newAddressableValue constructs a new addressable value of type t.
func newAddressableValue(t reflect.Type) addressableValue {
	return addressableValue{reflect.New(t).Elem(), true}
}

// TODO: Remove *jsonopts.Struct argument from [marshaler] and [unmarshaler].
// This can be directly accessed on the encoder or decoder.

// All marshal and unmarshal behavior is implemented using these signatures.
// The *jsonopts.Struct argument is guaranteed to identical to or at least
// a strict super-set of the options in Encoder.Struct or Decoder.Struct.
// It is identical for Marshal, Unmarshal, MarshalWrite, and UnmarshalRead.
// It is a super-set for MarshalEncode and UnmarshalDecode.
type (
	marshaler   = func(*jsontext.Encoder, addressableValue, *jsonopts.Struct) error
	unmarshaler = func(*jsontext.Decoder, addressableValue, *jsonopts.Struct) error
)

type arshaler struct {
	marshal    marshaler
	unmarshal  unmarshaler
	nonDefault bool
}

var lookupArshalerCache sync.Map // map[reflect.Type]*arshaler

func lookupArshaler(t reflect.Type) *arshaler {
	if v, ok := lookupArshalerCache.Load(t); ok {
		return v.(*arshaler)
	}

	fncs := makeDefaultArshaler(t)
	fncs = makeMethodArshaler(fncs, t)
	fncs = makeTimeArshaler(fncs, t)

	// Use the last stored so that duplicate arshalers can be garbage collected.
	v, _ := lookupArshalerCache.LoadOrStore(t, fncs)
	return v.(*arshaler)
}

var stringsPools = &sync.Pool{New: func() any { return new(stringSlice) }}

type stringSlice []string

// getStrings returns a non-nil pointer to a slice with length n.
func getStrings(n int) *stringSlice {
	s := stringsPools.Get().(*stringSlice)
	if cap(*s) < n {
		*s = make([]string, n)
	}
	*s = (*s)[:n]
	return s
}

func putStrings(s *stringSlice) {
	if cap(*s) > 1<<10 {
		*s = nil // avoid pinning arbitrarily large amounts of memory
	}
	stringsPools.Put(s)
}

func (ss *stringSlice) Sort() {
	slices.SortFunc(*ss, func(x, y string) int { return strings.Compare(x, y) })
}
