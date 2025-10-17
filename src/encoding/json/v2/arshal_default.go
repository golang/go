// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.jsonv2

package json

import (
	"bytes"
	"cmp"
	"encoding"
	"encoding/base32"
	"encoding/base64"
	"encoding/hex"
	"errors"
	"fmt"
	"math"
	"reflect"
	"slices"
	"strconv"
	"strings"
	"sync"

	"encoding/json/internal"
	"encoding/json/internal/jsonflags"
	"encoding/json/internal/jsonopts"
	"encoding/json/internal/jsonwire"
	"encoding/json/jsontext"
)

// optimizeCommon specifies whether to use optimizations targeted for certain
// common patterns, rather than using the slower, but more general logic.
// All tests should pass regardless of whether this is true or not.
const optimizeCommon = true

var (
	// Most natural Go type that correspond with each JSON type.
	anyType          = reflect.TypeFor[any]()            // JSON value
	boolType         = reflect.TypeFor[bool]()           // JSON bool
	stringType       = reflect.TypeFor[string]()         // JSON string
	float64Type      = reflect.TypeFor[float64]()        // JSON number
	mapStringAnyType = reflect.TypeFor[map[string]any]() // JSON object
	sliceAnyType     = reflect.TypeFor[[]any]()          // JSON array

	bytesType       = reflect.TypeFor[[]byte]()
	emptyStructType = reflect.TypeFor[struct{}]()
)

const startDetectingCyclesAfter = 1000

type seenPointers = map[any]struct{}

type typedPointer struct {
	typ reflect.Type
	ptr any // always stores unsafe.Pointer, but avoids depending on unsafe
	len int // remember slice length to avoid false positives
}

// visitPointer visits pointer p of type t, reporting an error if seen before.
// If successfully visited, then the caller must eventually call leave.
func visitPointer(m *seenPointers, v reflect.Value) error {
	p := typedPointer{v.Type(), v.UnsafePointer(), sliceLen(v)}
	if _, ok := (*m)[p]; ok {
		return internal.ErrCycle
	}
	if *m == nil {
		*m = make(seenPointers)
	}
	(*m)[p] = struct{}{}
	return nil
}
func leavePointer(m *seenPointers, v reflect.Value) {
	p := typedPointer{v.Type(), v.UnsafePointer(), sliceLen(v)}
	delete(*m, p)
}

func sliceLen(v reflect.Value) int {
	if v.Kind() == reflect.Slice {
		return v.Len()
	}
	return 0
}

func len64[Bytes ~[]byte | ~string](in Bytes) int64 {
	return int64(len(in))
}

func makeDefaultArshaler(t reflect.Type) *arshaler {
	switch t.Kind() {
	case reflect.Bool:
		return makeBoolArshaler(t)
	case reflect.String:
		return makeStringArshaler(t)
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		return makeIntArshaler(t)
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr:
		return makeUintArshaler(t)
	case reflect.Float32, reflect.Float64:
		return makeFloatArshaler(t)
	case reflect.Map:
		return makeMapArshaler(t)
	case reflect.Struct:
		return makeStructArshaler(t)
	case reflect.Slice:
		fncs := makeSliceArshaler(t)
		if t.Elem().Kind() == reflect.Uint8 {
			return makeBytesArshaler(t, fncs)
		}
		return fncs
	case reflect.Array:
		fncs := makeArrayArshaler(t)
		if t.Elem().Kind() == reflect.Uint8 {
			return makeBytesArshaler(t, fncs)
		}
		return fncs
	case reflect.Pointer:
		return makePointerArshaler(t)
	case reflect.Interface:
		return makeInterfaceArshaler(t)
	default:
		return makeInvalidArshaler(t)
	}
}

func makeBoolArshaler(t reflect.Type) *arshaler {
	var fncs arshaler
	fncs.marshal = func(enc *jsontext.Encoder, va addressableValue, mo *jsonopts.Struct) error {
		xe := export.Encoder(enc)
		if mo.Format != "" && mo.FormatDepth == xe.Tokens.Depth() {
			return newInvalidFormatError(enc, t)
		}

		// Optimize for marshaling without preceding whitespace.
		if optimizeCommon && !mo.Flags.Get(jsonflags.AnyWhitespace|jsonflags.StringifyBoolsAndStrings) && !xe.Tokens.Last.NeedObjectName() {
			xe.Buf = strconv.AppendBool(xe.Tokens.MayAppendDelim(xe.Buf, 't'), va.Bool())
			xe.Tokens.Last.Increment()
			if xe.NeedFlush() {
				return xe.Flush()
			}
			return nil
		}

		if mo.Flags.Get(jsonflags.StringifyBoolsAndStrings) {
			if va.Bool() {
				return enc.WriteToken(jsontext.String("true"))
			} else {
				return enc.WriteToken(jsontext.String("false"))
			}
		}
		return enc.WriteToken(jsontext.Bool(va.Bool()))
	}
	fncs.unmarshal = func(dec *jsontext.Decoder, va addressableValue, uo *jsonopts.Struct) error {
		xd := export.Decoder(dec)
		if uo.Format != "" && uo.FormatDepth == xd.Tokens.Depth() {
			return newInvalidFormatError(dec, t)
		}
		tok, err := dec.ReadToken()
		if err != nil {
			return err
		}
		k := tok.Kind()
		switch k {
		case 'n':
			if !uo.Flags.Get(jsonflags.MergeWithLegacySemantics) {
				va.SetBool(false)
			}
			return nil
		case 't', 'f':
			if !uo.Flags.Get(jsonflags.StringifyBoolsAndStrings) {
				va.SetBool(tok.Bool())
				return nil
			}
		case '"':
			if uo.Flags.Get(jsonflags.StringifyBoolsAndStrings) {
				switch tok.String() {
				case "true":
					va.SetBool(true)
				case "false":
					va.SetBool(false)
				default:
					if uo.Flags.Get(jsonflags.StringifyWithLegacySemantics) && tok.String() == "null" {
						if !uo.Flags.Get(jsonflags.MergeWithLegacySemantics) {
							va.SetBool(false)
						}
						return nil
					}
					return newUnmarshalErrorAfterWithValue(dec, t, strconv.ErrSyntax)
				}
				return nil
			}
		}
		return newUnmarshalErrorAfterWithSkipping(dec, t, nil)
	}
	return &fncs
}

func makeStringArshaler(t reflect.Type) *arshaler {
	var fncs arshaler
	fncs.marshal = func(enc *jsontext.Encoder, va addressableValue, mo *jsonopts.Struct) error {
		xe := export.Encoder(enc)
		if mo.Format != "" && mo.FormatDepth == xe.Tokens.Depth() {
			return newInvalidFormatError(enc, t)
		}

		// Optimize for marshaling without preceding whitespace.
		s := va.String()
		if optimizeCommon && !mo.Flags.Get(jsonflags.AnyWhitespace|jsonflags.StringifyBoolsAndStrings) && !xe.Tokens.Last.NeedObjectName() {
			b := xe.Buf
			b = xe.Tokens.MayAppendDelim(b, '"')
			b, err := jsonwire.AppendQuote(b, s, &mo.Flags)
			if err == nil {
				xe.Buf = b
				xe.Tokens.Last.Increment()
				if xe.NeedFlush() {
					return xe.Flush()
				}
				return nil
			}
			// Otherwise, the string contains invalid UTF-8,
			// so let the logic below construct the proper error.
		}

		if mo.Flags.Get(jsonflags.StringifyBoolsAndStrings) {
			b, err := jsonwire.AppendQuote(nil, s, &mo.Flags)
			if err != nil {
				return newMarshalErrorBefore(enc, t, &jsontext.SyntacticError{Err: err})
			}
			q, err := jsontext.AppendQuote(nil, b)
			if err != nil {
				panic("BUG: second AppendQuote should never fail: " + err.Error())
			}
			return enc.WriteValue(q)
		}
		return enc.WriteToken(jsontext.String(s))
	}
	fncs.unmarshal = func(dec *jsontext.Decoder, va addressableValue, uo *jsonopts.Struct) error {
		xd := export.Decoder(dec)
		if uo.Format != "" && uo.FormatDepth == xd.Tokens.Depth() {
			return newInvalidFormatError(dec, t)
		}
		var flags jsonwire.ValueFlags
		val, err := xd.ReadValue(&flags)
		if err != nil {
			return err
		}
		k := val.Kind()
		switch k {
		case 'n':
			if !uo.Flags.Get(jsonflags.MergeWithLegacySemantics) {
				va.SetString("")
			}
			return nil
		case '"':
			val = jsonwire.UnquoteMayCopy(val, flags.IsVerbatim())
			if uo.Flags.Get(jsonflags.StringifyBoolsAndStrings) {
				val, err = jsontext.AppendUnquote(nil, val)
				if err != nil {
					return newUnmarshalErrorAfter(dec, t, err)
				}
				if uo.Flags.Get(jsonflags.StringifyWithLegacySemantics) && string(val) == "null" {
					if !uo.Flags.Get(jsonflags.MergeWithLegacySemantics) {
						va.SetString("")
					}
					return nil
				}
			}
			if xd.StringCache == nil {
				xd.StringCache = new(stringCache)
			}
			str := makeString(xd.StringCache, val)
			va.SetString(str)
			return nil
		}
		return newUnmarshalErrorAfter(dec, t, nil)
	}
	return &fncs
}

var (
	appendEncodeBase16    = hex.AppendEncode
	appendEncodeBase32    = base32.StdEncoding.AppendEncode
	appendEncodeBase32Hex = base32.HexEncoding.AppendEncode
	appendEncodeBase64    = base64.StdEncoding.AppendEncode
	appendEncodeBase64URL = base64.URLEncoding.AppendEncode
	encodedLenBase16      = hex.EncodedLen
	encodedLenBase32      = base32.StdEncoding.EncodedLen
	encodedLenBase32Hex   = base32.HexEncoding.EncodedLen
	encodedLenBase64      = base64.StdEncoding.EncodedLen
	encodedLenBase64URL   = base64.URLEncoding.EncodedLen
	appendDecodeBase16    = hex.AppendDecode
	appendDecodeBase32    = base32.StdEncoding.AppendDecode
	appendDecodeBase32Hex = base32.HexEncoding.AppendDecode
	appendDecodeBase64    = base64.StdEncoding.AppendDecode
	appendDecodeBase64URL = base64.URLEncoding.AppendDecode
)

func makeBytesArshaler(t reflect.Type, fncs *arshaler) *arshaler {
	// NOTE: This handles both []~byte and [N]~byte.
	// The v2 default is to treat a []namedByte as equivalent to []T
	// since being able to convert []namedByte to []byte relies on
	// dubious Go reflection behavior (see https://go.dev/issue/24746).
	// For v1 emulation, we use jsonflags.FormatBytesWithLegacySemantics
	// to forcibly treat []namedByte as a []byte.
	marshalArray := fncs.marshal
	isNamedByte := t.Elem().PkgPath() != ""
	hasMarshaler := implementsAny(t.Elem(), allMarshalerTypes...)
	fncs.marshal = func(enc *jsontext.Encoder, va addressableValue, mo *jsonopts.Struct) error {
		if !mo.Flags.Get(jsonflags.FormatBytesWithLegacySemantics) && isNamedByte {
			return marshalArray(enc, va, mo) // treat as []T or [N]T
		}
		xe := export.Encoder(enc)
		appendEncode := appendEncodeBase64
		if mo.Format != "" && mo.FormatDepth == xe.Tokens.Depth() {
			switch mo.Format {
			case "base64":
				appendEncode = appendEncodeBase64
			case "base64url":
				appendEncode = appendEncodeBase64URL
			case "base32":
				appendEncode = appendEncodeBase32
			case "base32hex":
				appendEncode = appendEncodeBase32Hex
			case "base16", "hex":
				appendEncode = appendEncodeBase16
			case "array":
				mo.Format = ""
				return marshalArray(enc, va, mo)
			default:
				return newInvalidFormatError(enc, t)
			}
		} else if mo.Flags.Get(jsonflags.FormatByteArrayAsArray) && va.Kind() == reflect.Array {
			return marshalArray(enc, va, mo)
		} else if mo.Flags.Get(jsonflags.FormatBytesWithLegacySemantics) && hasMarshaler {
			return marshalArray(enc, va, mo)
		}
		if mo.Flags.Get(jsonflags.FormatNilSliceAsNull) && va.Kind() == reflect.Slice && va.IsNil() {
			// TODO: Provide a "emitempty" format override?
			return enc.WriteToken(jsontext.Null)
		}
		return xe.AppendRaw('"', true, func(b []byte) ([]byte, error) {
			return appendEncode(b, va.Bytes()), nil
		})
	}
	unmarshalArray := fncs.unmarshal
	fncs.unmarshal = func(dec *jsontext.Decoder, va addressableValue, uo *jsonopts.Struct) error {
		if !uo.Flags.Get(jsonflags.FormatBytesWithLegacySemantics) && isNamedByte {
			return unmarshalArray(dec, va, uo) // treat as []T or [N]T
		}
		xd := export.Decoder(dec)
		appendDecode, encodedLen := appendDecodeBase64, encodedLenBase64
		if uo.Format != "" && uo.FormatDepth == xd.Tokens.Depth() {
			switch uo.Format {
			case "base64":
				appendDecode, encodedLen = appendDecodeBase64, encodedLenBase64
			case "base64url":
				appendDecode, encodedLen = appendDecodeBase64URL, encodedLenBase64URL
			case "base32":
				appendDecode, encodedLen = appendDecodeBase32, encodedLenBase32
			case "base32hex":
				appendDecode, encodedLen = appendDecodeBase32Hex, encodedLenBase32Hex
			case "base16", "hex":
				appendDecode, encodedLen = appendDecodeBase16, encodedLenBase16
			case "array":
				uo.Format = ""
				return unmarshalArray(dec, va, uo)
			default:
				return newInvalidFormatError(dec, t)
			}
		} else if uo.Flags.Get(jsonflags.FormatByteArrayAsArray) && va.Kind() == reflect.Array {
			return unmarshalArray(dec, va, uo)
		} else if uo.Flags.Get(jsonflags.FormatBytesWithLegacySemantics) && dec.PeekKind() == '[' {
			return unmarshalArray(dec, va, uo)
		}
		var flags jsonwire.ValueFlags
		val, err := xd.ReadValue(&flags)
		if err != nil {
			return err
		}
		k := val.Kind()
		switch k {
		case 'n':
			if !uo.Flags.Get(jsonflags.MergeWithLegacySemantics) || va.Kind() != reflect.Array {
				va.SetZero()
			}
			return nil
		case '"':
			// NOTE: The v2 default is to strictly comply with RFC 4648.
			// Section 3.2 specifies that padding is required.
			// Section 3.3 specifies that non-alphabet characters
			// (e.g., '\r' or '\n') must be rejected.
			// Section 3.5 specifies that unnecessary non-zero bits in
			// the last quantum may be rejected. Since this is optional,
			// we do not reject such inputs.
			val = jsonwire.UnquoteMayCopy(val, flags.IsVerbatim())
			b, err := appendDecode(va.Bytes()[:0], val)
			if err != nil {
				return newUnmarshalErrorAfter(dec, t, err)
			}
			if len(val) != encodedLen(len(b)) && !uo.Flags.Get(jsonflags.ParseBytesWithLooseRFC4648) {
				// TODO(https://go.dev/issue/53845): RFC 4648, section 3.3,
				// specifies that non-alphabet characters must be rejected.
				// Unfortunately, the "base32" and "base64" packages allow
				// '\r' and '\n' characters by default.
				i := bytes.IndexAny(val, "\r\n")
				err := fmt.Errorf("illegal character %s at offset %d", jsonwire.QuoteRune(val[i:]), i)
				return newUnmarshalErrorAfter(dec, t, err)
			}

			if va.Kind() == reflect.Array {
				dst := va.Bytes()
				clear(dst[copy(dst, b):]) // noop if len(b) <= len(dst)
				if len(b) != len(dst) && !uo.Flags.Get(jsonflags.UnmarshalArrayFromAnyLength) {
					err := fmt.Errorf("decoded length of %d mismatches array length of %d", len(b), len(dst))
					return newUnmarshalErrorAfter(dec, t, err)
				}
			} else {
				if b == nil {
					b = []byte{}
				}
				va.SetBytes(b)
			}
			return nil
		}
		return newUnmarshalErrorAfter(dec, t, nil)
	}
	return fncs
}

func makeIntArshaler(t reflect.Type) *arshaler {
	var fncs arshaler
	bits := t.Bits()
	fncs.marshal = func(enc *jsontext.Encoder, va addressableValue, mo *jsonopts.Struct) error {
		xe := export.Encoder(enc)
		if mo.Format != "" && mo.FormatDepth == xe.Tokens.Depth() {
			return newInvalidFormatError(enc, t)
		}

		// Optimize for marshaling without preceding whitespace or string escaping.
		if optimizeCommon && !mo.Flags.Get(jsonflags.AnyWhitespace|jsonflags.StringifyNumbers) && !xe.Tokens.Last.NeedObjectName() {
			xe.Buf = strconv.AppendInt(xe.Tokens.MayAppendDelim(xe.Buf, '0'), va.Int(), 10)
			xe.Tokens.Last.Increment()
			if xe.NeedFlush() {
				return xe.Flush()
			}
			return nil
		}

		k := stringOrNumberKind(xe.Tokens.Last.NeedObjectName() || mo.Flags.Get(jsonflags.StringifyNumbers))
		return xe.AppendRaw(k, true, func(b []byte) ([]byte, error) {
			return strconv.AppendInt(b, va.Int(), 10), nil
		})
	}
	fncs.unmarshal = func(dec *jsontext.Decoder, va addressableValue, uo *jsonopts.Struct) error {
		xd := export.Decoder(dec)
		if uo.Format != "" && uo.FormatDepth == xd.Tokens.Depth() {
			return newInvalidFormatError(dec, t)
		}
		stringify := xd.Tokens.Last.NeedObjectName() || uo.Flags.Get(jsonflags.StringifyNumbers)
		var flags jsonwire.ValueFlags
		val, err := xd.ReadValue(&flags)
		if err != nil {
			return err
		}
		k := val.Kind()
		switch k {
		case 'n':
			if !uo.Flags.Get(jsonflags.MergeWithLegacySemantics) {
				va.SetInt(0)
			}
			return nil
		case '"':
			if !stringify {
				break
			}
			val = jsonwire.UnquoteMayCopy(val, flags.IsVerbatim())
			if uo.Flags.Get(jsonflags.StringifyWithLegacySemantics) {
				// For historical reasons, v1 parsed a quoted number
				// according to the Go syntax and permitted a quoted null.
				// See https://go.dev/issue/75619
				n, err := strconv.ParseInt(string(val), 10, bits)
				if err != nil {
					if string(val) == "null" {
						if !uo.Flags.Get(jsonflags.MergeWithLegacySemantics) {
							va.SetInt(0)
						}
						return nil
					}
					return newUnmarshalErrorAfterWithValue(dec, t, errors.Unwrap(err))
				}
				va.SetInt(n)
				return nil
			}
			fallthrough
		case '0':
			if stringify && k == '0' {
				break
			}
			var negOffset int
			neg := len(val) > 0 && val[0] == '-'
			if neg {
				negOffset = 1
			}
			n, ok := jsonwire.ParseUint(val[negOffset:])
			maxInt := uint64(1) << (bits - 1)
			overflow := (neg && n > maxInt) || (!neg && n > maxInt-1)
			if !ok {
				if n != math.MaxUint64 {
					return newUnmarshalErrorAfterWithValue(dec, t, strconv.ErrSyntax)
				}
				overflow = true
			}
			if overflow {
				return newUnmarshalErrorAfterWithValue(dec, t, strconv.ErrRange)
			}
			if neg {
				va.SetInt(int64(-n))
			} else {
				va.SetInt(int64(+n))
			}
			return nil
		}
		return newUnmarshalErrorAfter(dec, t, nil)
	}
	return &fncs
}

func makeUintArshaler(t reflect.Type) *arshaler {
	var fncs arshaler
	bits := t.Bits()
	fncs.marshal = func(enc *jsontext.Encoder, va addressableValue, mo *jsonopts.Struct) error {
		xe := export.Encoder(enc)
		if mo.Format != "" && mo.FormatDepth == xe.Tokens.Depth() {
			return newInvalidFormatError(enc, t)
		}

		// Optimize for marshaling without preceding whitespace or string escaping.
		if optimizeCommon && !mo.Flags.Get(jsonflags.AnyWhitespace|jsonflags.StringifyNumbers) && !xe.Tokens.Last.NeedObjectName() {
			xe.Buf = strconv.AppendUint(xe.Tokens.MayAppendDelim(xe.Buf, '0'), va.Uint(), 10)
			xe.Tokens.Last.Increment()
			if xe.NeedFlush() {
				return xe.Flush()
			}
			return nil
		}

		k := stringOrNumberKind(xe.Tokens.Last.NeedObjectName() || mo.Flags.Get(jsonflags.StringifyNumbers))
		return xe.AppendRaw(k, true, func(b []byte) ([]byte, error) {
			return strconv.AppendUint(b, va.Uint(), 10), nil
		})
	}
	fncs.unmarshal = func(dec *jsontext.Decoder, va addressableValue, uo *jsonopts.Struct) error {
		xd := export.Decoder(dec)
		if uo.Format != "" && uo.FormatDepth == xd.Tokens.Depth() {
			return newInvalidFormatError(dec, t)
		}
		stringify := xd.Tokens.Last.NeedObjectName() || uo.Flags.Get(jsonflags.StringifyNumbers)
		var flags jsonwire.ValueFlags
		val, err := xd.ReadValue(&flags)
		if err != nil {
			return err
		}
		k := val.Kind()
		switch k {
		case 'n':
			if !uo.Flags.Get(jsonflags.MergeWithLegacySemantics) {
				va.SetUint(0)
			}
			return nil
		case '"':
			if !stringify {
				break
			}
			val = jsonwire.UnquoteMayCopy(val, flags.IsVerbatim())
			if uo.Flags.Get(jsonflags.StringifyWithLegacySemantics) {
				// For historical reasons, v1 parsed a quoted number
				// according to the Go syntax and permitted a quoted null.
				// See https://go.dev/issue/75619
				n, err := strconv.ParseUint(string(val), 10, bits)
				if err != nil {
					if string(val) == "null" {
						if !uo.Flags.Get(jsonflags.MergeWithLegacySemantics) {
							va.SetUint(0)
						}
						return nil
					}
					return newUnmarshalErrorAfterWithValue(dec, t, errors.Unwrap(err))
				}
				va.SetUint(n)
				return nil
			}
			fallthrough
		case '0':
			if stringify && k == '0' {
				break
			}
			n, ok := jsonwire.ParseUint(val)
			maxUint := uint64(1) << bits
			overflow := n > maxUint-1
			if !ok {
				if n != math.MaxUint64 {
					return newUnmarshalErrorAfterWithValue(dec, t, strconv.ErrSyntax)
				}
				overflow = true
			}
			if overflow {
				return newUnmarshalErrorAfterWithValue(dec, t, strconv.ErrRange)
			}
			va.SetUint(n)
			return nil
		}
		return newUnmarshalErrorAfter(dec, t, nil)
	}
	return &fncs
}

func makeFloatArshaler(t reflect.Type) *arshaler {
	var fncs arshaler
	bits := t.Bits()
	fncs.marshal = func(enc *jsontext.Encoder, va addressableValue, mo *jsonopts.Struct) error {
		xe := export.Encoder(enc)
		var allowNonFinite bool
		if mo.Format != "" && mo.FormatDepth == xe.Tokens.Depth() {
			if mo.Format == "nonfinite" {
				allowNonFinite = true
			} else {
				return newInvalidFormatError(enc, t)
			}
		}

		fv := va.Float()
		if math.IsNaN(fv) || math.IsInf(fv, 0) {
			if !allowNonFinite {
				err := fmt.Errorf("unsupported value: %v", fv)
				return newMarshalErrorBefore(enc, t, err)
			}
			return enc.WriteToken(jsontext.Float(fv))
		}

		// Optimize for marshaling without preceding whitespace or string escaping.
		if optimizeCommon && !mo.Flags.Get(jsonflags.AnyWhitespace|jsonflags.StringifyNumbers) && !xe.Tokens.Last.NeedObjectName() {
			xe.Buf = jsonwire.AppendFloat(xe.Tokens.MayAppendDelim(xe.Buf, '0'), fv, bits)
			xe.Tokens.Last.Increment()
			if xe.NeedFlush() {
				return xe.Flush()
			}
			return nil
		}

		k := stringOrNumberKind(xe.Tokens.Last.NeedObjectName() || mo.Flags.Get(jsonflags.StringifyNumbers))
		return xe.AppendRaw(k, true, func(b []byte) ([]byte, error) {
			return jsonwire.AppendFloat(b, va.Float(), bits), nil
		})
	}
	fncs.unmarshal = func(dec *jsontext.Decoder, va addressableValue, uo *jsonopts.Struct) error {
		xd := export.Decoder(dec)
		var allowNonFinite bool
		if uo.Format != "" && uo.FormatDepth == xd.Tokens.Depth() {
			if uo.Format == "nonfinite" {
				allowNonFinite = true
			} else {
				return newInvalidFormatError(dec, t)
			}
		}
		stringify := xd.Tokens.Last.NeedObjectName() || uo.Flags.Get(jsonflags.StringifyNumbers)
		var flags jsonwire.ValueFlags
		val, err := xd.ReadValue(&flags)
		if err != nil {
			return err
		}
		k := val.Kind()
		switch k {
		case 'n':
			if !uo.Flags.Get(jsonflags.MergeWithLegacySemantics) {
				va.SetFloat(0)
			}
			return nil
		case '"':
			val = jsonwire.UnquoteMayCopy(val, flags.IsVerbatim())
			if allowNonFinite {
				switch string(val) {
				case "NaN":
					va.SetFloat(math.NaN())
					return nil
				case "Infinity":
					va.SetFloat(math.Inf(+1))
					return nil
				case "-Infinity":
					va.SetFloat(math.Inf(-1))
					return nil
				}
			}
			if !stringify {
				break
			}
			if uo.Flags.Get(jsonflags.StringifyWithLegacySemantics) {
				// For historical reasons, v1 parsed a quoted number
				// according to the Go syntax and permitted a quoted null.
				// See https://go.dev/issue/75619
				n, err := strconv.ParseFloat(string(val), bits)
				if err != nil {
					if string(val) == "null" {
						if !uo.Flags.Get(jsonflags.MergeWithLegacySemantics) {
							va.SetFloat(0)
						}
						return nil
					}
					return newUnmarshalErrorAfterWithValue(dec, t, errors.Unwrap(err))
				}
				va.SetFloat(n)
				return nil
			}
			if n, err := jsonwire.ConsumeNumber(val); n != len(val) || err != nil {
				return newUnmarshalErrorAfterWithValue(dec, t, strconv.ErrSyntax)
			}
			fallthrough
		case '0':
			if stringify && k == '0' {
				break
			}
			fv, ok := jsonwire.ParseFloat(val, bits)
			va.SetFloat(fv)
			if !ok {
				return newUnmarshalErrorAfterWithValue(dec, t, strconv.ErrRange)
			}
			return nil
		}
		return newUnmarshalErrorAfter(dec, t, nil)
	}
	return &fncs
}

func makeMapArshaler(t reflect.Type) *arshaler {
	// NOTE: The logic below disables namespaces for tracking duplicate names
	// when handling map keys with a unique representation.

	// NOTE: Values retrieved from a map are not addressable,
	// so we shallow copy the values to make them addressable and
	// store them back into the map afterwards.

	var fncs arshaler
	var (
		once    sync.Once
		keyFncs *arshaler
		valFncs *arshaler
	)
	init := func() {
		keyFncs = lookupArshaler(t.Key())
		valFncs = lookupArshaler(t.Elem())
	}
	nillableLegacyKey := t.Key().Kind() == reflect.Pointer &&
		implementsAny(t.Key(), textMarshalerType, textAppenderType)
	fncs.marshal = func(enc *jsontext.Encoder, va addressableValue, mo *jsonopts.Struct) error {
		// Check for cycles.
		xe := export.Encoder(enc)
		if xe.Tokens.Depth() > startDetectingCyclesAfter {
			if err := visitPointer(&xe.SeenPointers, va.Value); err != nil {
				return newMarshalErrorBefore(enc, t, err)
			}
			defer leavePointer(&xe.SeenPointers, va.Value)
		}

		emitNull := mo.Flags.Get(jsonflags.FormatNilMapAsNull)
		if mo.Format != "" && mo.FormatDepth == xe.Tokens.Depth() {
			switch mo.Format {
			case "emitnull":
				emitNull = true
				mo.Format = ""
			case "emitempty":
				emitNull = false
				mo.Format = ""
			default:
				return newInvalidFormatError(enc, t)
			}
		}

		// Handle empty maps.
		n := va.Len()
		if n == 0 {
			if emitNull && va.IsNil() {
				return enc.WriteToken(jsontext.Null)
			}
			// Optimize for marshaling an empty map without any preceding whitespace.
			if optimizeCommon && !mo.Flags.Get(jsonflags.AnyWhitespace) && !xe.Tokens.Last.NeedObjectName() {
				xe.Buf = append(xe.Tokens.MayAppendDelim(xe.Buf, '{'), "{}"...)
				xe.Tokens.Last.Increment()
				if xe.NeedFlush() {
					return xe.Flush()
				}
				return nil
			}
		}

		once.Do(init)
		if err := enc.WriteToken(jsontext.BeginObject); err != nil {
			return err
		}
		if n > 0 {
			nonDefaultKey := keyFncs.nonDefault
			marshalKey := keyFncs.marshal
			marshalVal := valFncs.marshal
			if mo.Marshalers != nil {
				var ok bool
				marshalKey, ok = mo.Marshalers.(*Marshalers).lookup(marshalKey, t.Key())
				marshalVal, _ = mo.Marshalers.(*Marshalers).lookup(marshalVal, t.Elem())
				nonDefaultKey = nonDefaultKey || ok
			}
			k := newAddressableValue(t.Key())
			v := newAddressableValue(t.Elem())

			// A Go map guarantees that each entry has a unique key.
			// As such, disable the expensive duplicate name check if we know
			// that every Go key will serialize as a unique JSON string.
			if !nonDefaultKey && mapKeyWithUniqueRepresentation(k.Kind(), mo.Flags.Get(jsonflags.AllowInvalidUTF8)) {
				xe.Tokens.Last.DisableNamespace()
			}

			switch {
			case !mo.Flags.Get(jsonflags.Deterministic) || n <= 1:
				for iter := va.Value.MapRange(); iter.Next(); {
					k.SetIterKey(iter)
					err := marshalKey(enc, k, mo)
					if err != nil {
						if mo.Flags.Get(jsonflags.CallMethodsWithLegacySemantics) &&
							errors.Is(err, jsontext.ErrNonStringName) && nillableLegacyKey && k.IsNil() {
							err = enc.WriteToken(jsontext.String(""))
						}
						if err != nil {
							if serr, ok := err.(*jsontext.SyntacticError); ok && serr.Err == jsontext.ErrNonStringName {
								err = newMarshalErrorBefore(enc, k.Type(), err)
							}
							return err
						}
					}
					v.SetIterValue(iter)
					if err := marshalVal(enc, v, mo); err != nil {
						return err
					}
				}
			case !nonDefaultKey && t.Key().Kind() == reflect.String:
				names := getStrings(n)
				for i, iter := 0, va.Value.MapRange(); i < n && iter.Next(); i++ {
					k.SetIterKey(iter)
					(*names)[i] = k.String()
				}
				names.Sort()
				for _, name := range *names {
					if err := enc.WriteToken(jsontext.String(name)); err != nil {
						return err
					}
					// TODO(https://go.dev/issue/57061): Use v.SetMapIndexOf.
					k.SetString(name)
					v.Set(va.MapIndex(k.Value))
					if err := marshalVal(enc, v, mo); err != nil {
						return err
					}
				}
				putStrings(names)
			default:
				type member struct {
					name string // unquoted name
					key  addressableValue
					val  addressableValue
				}
				members := make([]member, n)
				keys := reflect.MakeSlice(reflect.SliceOf(t.Key()), n, n)
				vals := reflect.MakeSlice(reflect.SliceOf(t.Elem()), n, n)
				for i, iter := 0, va.Value.MapRange(); i < n && iter.Next(); i++ {
					// Marshal the member name.
					k := addressableValue{keys.Index(i), true} // indexed slice element is always addressable
					k.SetIterKey(iter)
					v := addressableValue{vals.Index(i), true} // indexed slice element is always addressable
					v.SetIterValue(iter)
					err := marshalKey(enc, k, mo)
					if err != nil {
						if mo.Flags.Get(jsonflags.CallMethodsWithLegacySemantics) &&
							errors.Is(err, jsontext.ErrNonStringName) && nillableLegacyKey && k.IsNil() {
							err = enc.WriteToken(jsontext.String(""))
						}
						if err != nil {
							if serr, ok := err.(*jsontext.SyntacticError); ok && serr.Err == jsontext.ErrNonStringName {
								err = newMarshalErrorBefore(enc, k.Type(), err)
							}
							return err
						}
					}
					name := xe.UnwriteOnlyObjectMemberName()
					members[i] = member{name, k, v}
				}
				// TODO: If AllowDuplicateNames is enabled, then sort according
				// to reflect.Value as well if the names are equal.
				// See internal/fmtsort.
				slices.SortFunc(members, func(x, y member) int {
					return strings.Compare(x.name, y.name)
				})
				for _, member := range members {
					if err := enc.WriteToken(jsontext.String(member.name)); err != nil {
						return err
					}
					if err := marshalVal(enc, member.val, mo); err != nil {
						return err
					}
				}
			}
		}
		if err := enc.WriteToken(jsontext.EndObject); err != nil {
			return err
		}
		return nil
	}
	fncs.unmarshal = func(dec *jsontext.Decoder, va addressableValue, uo *jsonopts.Struct) error {
		xd := export.Decoder(dec)
		if uo.Format != "" && uo.FormatDepth == xd.Tokens.Depth() {
			switch uo.Format {
			case "emitnull", "emitempty":
				uo.Format = "" // only relevant for marshaling
			default:
				return newInvalidFormatError(dec, t)
			}
		}
		tok, err := dec.ReadToken()
		if err != nil {
			return err
		}
		k := tok.Kind()
		switch k {
		case 'n':
			va.SetZero()
			return nil
		case '{':
			once.Do(init)
			if va.IsNil() {
				va.Set(reflect.MakeMap(t))
			}

			nonDefaultKey := keyFncs.nonDefault
			unmarshalKey := keyFncs.unmarshal
			unmarshalVal := valFncs.unmarshal
			if uo.Unmarshalers != nil {
				var ok bool
				unmarshalKey, ok = uo.Unmarshalers.(*Unmarshalers).lookup(unmarshalKey, t.Key())
				unmarshalVal, _ = uo.Unmarshalers.(*Unmarshalers).lookup(unmarshalVal, t.Elem())
				nonDefaultKey = nonDefaultKey || ok
			}
			k := newAddressableValue(t.Key())
			v := newAddressableValue(t.Elem())

			// Manually check for duplicate entries by virtue of whether the
			// unmarshaled key already exists in the destination Go map.
			// Consequently, syntactically different names (e.g., "0" and "-0")
			// will be rejected as duplicates since they semantically refer
			// to the same Go value. This is an unusual interaction
			// between syntax and semantics, but is more correct.
			if !nonDefaultKey && mapKeyWithUniqueRepresentation(k.Kind(), uo.Flags.Get(jsonflags.AllowInvalidUTF8)) {
				xd.Tokens.Last.DisableNamespace()
			}

			// In the rare case where the map is not already empty,
			// then we need to manually track which keys we already saw
			// since existing presence alone is insufficient to indicate
			// whether the input had a duplicate name.
			var seen reflect.Value
			if !uo.Flags.Get(jsonflags.AllowDuplicateNames) && va.Len() > 0 {
				seen = reflect.MakeMap(reflect.MapOf(k.Type(), emptyStructType))
			}

			var errUnmarshal error
			for dec.PeekKind() != '}' {
				// Unmarshal the map entry key.
				k.SetZero()
				err := unmarshalKey(dec, k, uo)
				if err != nil {
					if isFatalError(err, uo.Flags) {
						return err
					}
					if err := dec.SkipValue(); err != nil {
						return err
					}
					errUnmarshal = cmp.Or(errUnmarshal, err)
					continue
				}
				if k.Kind() == reflect.Interface && !k.IsNil() && !k.Elem().Type().Comparable() {
					err := newUnmarshalErrorAfter(dec, t, fmt.Errorf("invalid incomparable key type %v", k.Elem().Type()))
					if !uo.Flags.Get(jsonflags.ReportErrorsWithLegacySemantics) {
						return err
					}
					if err2 := dec.SkipValue(); err2 != nil {
						return err2
					}
					errUnmarshal = cmp.Or(errUnmarshal, err)
					continue
				}

				// Check if a pre-existing map entry value exists for this key.
				if v2 := va.MapIndex(k.Value); v2.IsValid() {
					if !uo.Flags.Get(jsonflags.AllowDuplicateNames) && (!seen.IsValid() || seen.MapIndex(k.Value).IsValid()) {
						// TODO: Unread the object name.
						name := xd.PreviousTokenOrValue()
						return newDuplicateNameError(dec.StackPointer(), nil, dec.InputOffset()-len64(name))
					}
					if !uo.Flags.Get(jsonflags.MergeWithLegacySemantics) {
						v.Set(v2)
					} else {
						v.SetZero()
					}
				} else {
					v.SetZero()
				}

				// Unmarshal the map entry value.
				err = unmarshalVal(dec, v, uo)
				va.SetMapIndex(k.Value, v.Value)
				if seen.IsValid() {
					seen.SetMapIndex(k.Value, reflect.Zero(emptyStructType))
				}
				if err != nil {
					if isFatalError(err, uo.Flags) {
						return err
					}
					errUnmarshal = cmp.Or(errUnmarshal, err)
				}
			}
			if _, err := dec.ReadToken(); err != nil {
				return err
			}
			return errUnmarshal
		}
		return newUnmarshalErrorAfterWithSkipping(dec, t, nil)
	}
	return &fncs
}

// mapKeyWithUniqueRepresentation reports whether all possible values of k
// marshal to a different JSON value, and whether all possible JSON values
// that can unmarshal into k unmarshal to different Go values.
// In other words, the representation must be a bijective.
func mapKeyWithUniqueRepresentation(k reflect.Kind, allowInvalidUTF8 bool) bool {
	switch k {
	case reflect.Bool,
		reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
		reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr:
		return true
	case reflect.String:
		// For strings, we have to be careful since names with invalid UTF-8
		// maybe unescape to the same Go string value.
		return !allowInvalidUTF8
	default:
		// Floating-point kinds are not listed above since NaNs
		// can appear multiple times and all serialize as "NaN".
		return false
	}
}

var errNilField = errors.New("cannot set embedded pointer to unexported struct type")

func makeStructArshaler(t reflect.Type) *arshaler {
	// NOTE: The logic below disables namespaces for tracking duplicate names
	// and does the tracking locally with an efficient bit-set based on which
	// Go struct fields were seen.

	var fncs arshaler
	var (
		once    sync.Once
		fields  structFields
		errInit *SemanticError
	)
	init := func() {
		fields, errInit = makeStructFields(t)
	}
	fncs.marshal = func(enc *jsontext.Encoder, va addressableValue, mo *jsonopts.Struct) error {
		xe := export.Encoder(enc)
		if mo.Format != "" && mo.FormatDepth == xe.Tokens.Depth() {
			return newInvalidFormatError(enc, t)
		}
		once.Do(init)
		if errInit != nil && !mo.Flags.Get(jsonflags.ReportErrorsWithLegacySemantics) {
			return newMarshalErrorBefore(enc, errInit.GoType, errInit.Err)
		}
		if err := enc.WriteToken(jsontext.BeginObject); err != nil {
			return err
		}
		var seenIdxs uintSet
		prevIdx := -1
		xe.Tokens.Last.DisableNamespace() // we manually ensure unique names below
		for i := range fields.flattened {
			f := &fields.flattened[i]
			v := addressableValue{va.Field(f.index0), va.forcedAddr} // addressable if struct value is addressable
			if len(f.index) > 0 {
				v = v.fieldByIndex(f.index, false)
				if !v.IsValid() {
					continue // implies a nil inlined field
				}
			}

			// OmitZero skips the field if the Go value is zero,
			// which we can determine up front without calling the marshaler.
			if (f.omitzero || mo.Flags.Get(jsonflags.OmitZeroStructFields)) &&
				((f.isZero == nil && v.IsZero()) || (f.isZero != nil && f.isZero(v))) {
				continue
			}

			// Check for the legacy definition of omitempty.
			if f.omitempty && mo.Flags.Get(jsonflags.OmitEmptyWithLegacySemantics) && isLegacyEmpty(v) {
				continue
			}

			marshal := f.fncs.marshal
			nonDefault := f.fncs.nonDefault
			if mo.Marshalers != nil {
				var ok bool
				marshal, ok = mo.Marshalers.(*Marshalers).lookup(marshal, f.typ)
				nonDefault = nonDefault || ok
			}

			// OmitEmpty skips the field if the marshaled JSON value is empty,
			// which we can know up front if there are no custom marshalers,
			// otherwise we must marshal the value and unwrite it if empty.
			if f.omitempty && !mo.Flags.Get(jsonflags.OmitEmptyWithLegacySemantics) &&
				!nonDefault && f.isEmpty != nil && f.isEmpty(v) {
				continue // fast path for omitempty
			}

			// Write the object member name.
			//
			// The logic below is semantically equivalent to:
			//	enc.WriteToken(String(f.name))
			// but specialized and simplified because:
			//	1. The Encoder must be expecting an object name.
			//	2. The object namespace is guaranteed to be disabled.
			//	3. The object name is guaranteed to be valid and pre-escaped.
			//	4. There is no need to flush the buffer (for unwrite purposes).
			//	5. There is no possibility of an error occurring.
			if optimizeCommon {
				// Append any delimiters or optional whitespace.
				b := xe.Buf
				if xe.Tokens.Last.Length() > 0 {
					b = append(b, ',')
					if mo.Flags.Get(jsonflags.SpaceAfterComma) {
						b = append(b, ' ')
					}
				}
				if mo.Flags.Get(jsonflags.Multiline) {
					b = xe.AppendIndent(b, xe.Tokens.NeedIndent('"'))
				}

				// Append the token to the output and to the state machine.
				n0 := len(b) // offset before calling AppendQuote
				if !f.nameNeedEscape {
					b = append(b, f.quotedName...)
				} else {
					b, _ = jsonwire.AppendQuote(b, f.name, &mo.Flags)
				}
				xe.Buf = b
				xe.Names.ReplaceLastQuotedOffset(n0)
				xe.Tokens.Last.Increment()
			} else {
				if err := enc.WriteToken(jsontext.String(f.name)); err != nil {
					return err
				}
			}

			// Write the object member value.
			flagsOriginal := mo.Flags
			if f.string {
				if !mo.Flags.Get(jsonflags.StringifyWithLegacySemantics) {
					mo.Flags.Set(jsonflags.StringifyNumbers | 1)
				} else if canLegacyStringify(f.typ) {
					mo.Flags.Set(jsonflags.StringifyNumbers | jsonflags.StringifyBoolsAndStrings | 1)
				}
			}
			if f.format != "" {
				mo.FormatDepth = xe.Tokens.Depth()
				mo.Format = f.format
			}
			err := marshal(enc, v, mo)
			mo.Flags = flagsOriginal
			mo.Format = ""
			if err != nil {
				return err
			}

			// Try unwriting the member if empty (slow path for omitempty).
			if f.omitempty && !mo.Flags.Get(jsonflags.OmitEmptyWithLegacySemantics) {
				var prevName *string
				if prevIdx >= 0 {
					prevName = &fields.flattened[prevIdx].name
				}
				if xe.UnwriteEmptyObjectMember(prevName) {
					continue
				}
			}

			// Remember the previous written object member.
			// The set of seen fields only needs to be updated to detect
			// duplicate names with those from the inlined fallback.
			if !mo.Flags.Get(jsonflags.AllowDuplicateNames) && fields.inlinedFallback != nil {
				seenIdxs.insert(uint(f.id))
			}
			prevIdx = f.id
		}
		if fields.inlinedFallback != nil && !(mo.Flags.Get(jsonflags.DiscardUnknownMembers) && fields.inlinedFallback.unknown) {
			var insertUnquotedName func([]byte) bool
			if !mo.Flags.Get(jsonflags.AllowDuplicateNames) {
				insertUnquotedName = func(name []byte) bool {
					// Check that the name from inlined fallback does not match
					// one of the previously marshaled names from known fields.
					if foldedFields := fields.lookupByFoldedName(name); len(foldedFields) > 0 {
						if f := fields.byActualName[string(name)]; f != nil {
							return seenIdxs.insert(uint(f.id))
						}
						for _, f := range foldedFields {
							if f.matchFoldedName(name, &mo.Flags) {
								return seenIdxs.insert(uint(f.id))
							}
						}
					}

					// Check that the name does not match any other name
					// previously marshaled from the inlined fallback.
					return xe.Namespaces.Last().InsertUnquoted(name)
				}
			}
			if err := marshalInlinedFallbackAll(enc, va, mo, fields.inlinedFallback, insertUnquotedName); err != nil {
				return err
			}
		}
		if err := enc.WriteToken(jsontext.EndObject); err != nil {
			return err
		}
		return nil
	}
	fncs.unmarshal = func(dec *jsontext.Decoder, va addressableValue, uo *jsonopts.Struct) error {
		xd := export.Decoder(dec)
		if uo.Format != "" && uo.FormatDepth == xd.Tokens.Depth() {
			return newInvalidFormatError(dec, t)
		}
		tok, err := dec.ReadToken()
		if err != nil {
			return err
		}
		k := tok.Kind()
		switch k {
		case 'n':
			if !uo.Flags.Get(jsonflags.MergeWithLegacySemantics) {
				va.SetZero()
			}
			return nil
		case '{':
			once.Do(init)
			if errInit != nil && !uo.Flags.Get(jsonflags.ReportErrorsWithLegacySemantics) {
				return newUnmarshalErrorAfter(dec, errInit.GoType, errInit.Err)
			}
			var seenIdxs uintSet
			xd.Tokens.Last.DisableNamespace()
			var errUnmarshal error
			for dec.PeekKind() != '}' {
				// Process the object member name.
				var flags jsonwire.ValueFlags
				val, err := xd.ReadValue(&flags)
				if err != nil {
					return err
				}
				name := jsonwire.UnquoteMayCopy(val, flags.IsVerbatim())
				f := fields.byActualName[string(name)]
				if f == nil {
					for _, f2 := range fields.lookupByFoldedName(name) {
						if f2.matchFoldedName(name, &uo.Flags) {
							f = f2
							break
						}
					}
					if f == nil {
						if uo.Flags.Get(jsonflags.RejectUnknownMembers) && (fields.inlinedFallback == nil || fields.inlinedFallback.unknown) {
							err := newUnmarshalErrorAfter(dec, t, ErrUnknownName)
							if !uo.Flags.Get(jsonflags.ReportErrorsWithLegacySemantics) {
								return err
							}
							errUnmarshal = cmp.Or(errUnmarshal, err)
						}
						if !uo.Flags.Get(jsonflags.AllowDuplicateNames) && !xd.Namespaces.Last().InsertUnquoted(name) {
							// TODO: Unread the object name.
							return newDuplicateNameError(dec.StackPointer(), nil, dec.InputOffset()-len64(val))
						}

						if fields.inlinedFallback == nil {
							// Skip unknown value since we have no place to store it.
							if err := dec.SkipValue(); err != nil {
								return err
							}
						} else {
							// Marshal into value capable of storing arbitrary object members.
							if err := unmarshalInlinedFallbackNext(dec, va, uo, fields.inlinedFallback, val, name); err != nil {
								if isFatalError(err, uo.Flags) {
									return err
								}
								errUnmarshal = cmp.Or(errUnmarshal, err)
							}
						}
						continue
					}
				}
				if !uo.Flags.Get(jsonflags.AllowDuplicateNames) && !seenIdxs.insert(uint(f.id)) {
					// TODO: Unread the object name.
					return newDuplicateNameError(dec.StackPointer(), nil, dec.InputOffset()-len64(val))
				}

				// Process the object member value.
				unmarshal := f.fncs.unmarshal
				if uo.Unmarshalers != nil {
					unmarshal, _ = uo.Unmarshalers.(*Unmarshalers).lookup(unmarshal, f.typ)
				}
				flagsOriginal := uo.Flags
				if f.string {
					if !uo.Flags.Get(jsonflags.StringifyWithLegacySemantics) {
						uo.Flags.Set(jsonflags.StringifyNumbers | 1)
					} else if canLegacyStringify(f.typ) {
						uo.Flags.Set(jsonflags.StringifyNumbers | jsonflags.StringifyBoolsAndStrings | 1)
					}
				}
				if f.format != "" {
					uo.FormatDepth = xd.Tokens.Depth()
					uo.Format = f.format
				}
				v := addressableValue{va.Field(f.index0), va.forcedAddr} // addressable if struct value is addressable
				if len(f.index) > 0 {
					v = v.fieldByIndex(f.index, true)
					if !v.IsValid() {
						err := newUnmarshalErrorBefore(dec, t, errNilField)
						if !uo.Flags.Get(jsonflags.ReportErrorsWithLegacySemantics) {
							return err
						}
						errUnmarshal = cmp.Or(errUnmarshal, err)
						unmarshal = func(dec *jsontext.Decoder, _ addressableValue, _ *jsonopts.Struct) error {
							return dec.SkipValue()
						}
					}
				}
				err = unmarshal(dec, v, uo)
				uo.Flags = flagsOriginal
				uo.Format = ""
				if err != nil {
					if isFatalError(err, uo.Flags) {
						return err
					}
					errUnmarshal = cmp.Or(errUnmarshal, err)
				}
			}
			if _, err := dec.ReadToken(); err != nil {
				return err
			}
			return errUnmarshal
		}
		return newUnmarshalErrorAfterWithSkipping(dec, t, nil)
	}
	return &fncs
}

func (va addressableValue) fieldByIndex(index []int, mayAlloc bool) addressableValue {
	for _, i := range index {
		va = va.indirect(mayAlloc)
		if !va.IsValid() {
			return va
		}
		va = addressableValue{va.Field(i), va.forcedAddr} // addressable if struct value is addressable
	}
	return va
}

func (va addressableValue) indirect(mayAlloc bool) addressableValue {
	if va.Kind() == reflect.Pointer {
		if va.IsNil() {
			if !mayAlloc || !va.CanSet() {
				return addressableValue{}
			}
			va.Set(reflect.New(va.Type().Elem()))
		}
		va = addressableValue{va.Elem(), false} // dereferenced pointer is always addressable
	}
	return va
}

// isLegacyEmpty reports whether a value is empty according to the v1 definition.
func isLegacyEmpty(v addressableValue) bool {
	// Equivalent to encoding/json.isEmptyValue@v1.21.0.
	switch v.Kind() {
	case reflect.Bool:
		return v.Bool() == false
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		return v.Int() == 0
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr:
		return v.Uint() == 0
	case reflect.Float32, reflect.Float64:
		return v.Float() == 0
	case reflect.String, reflect.Map, reflect.Slice, reflect.Array:
		return v.Len() == 0
	case reflect.Pointer, reflect.Interface:
		return v.IsNil()
	}
	return false
}

// canLegacyStringify reports whether t can be stringified according to v1,
// where t is a bool, string, or number (or unnamed pointer to such).
// In v1, the `string` option does not apply recursively to nested types within
// a composite Go type (e.g., an array, slice, struct, map, or interface).
func canLegacyStringify(t reflect.Type) bool {
	// Based on encoding/json.typeFields#L1126-L1143@v1.23.0
	if t.Name() == "" && t.Kind() == reflect.Ptr {
		t = t.Elem()
	}
	switch t.Kind() {
	case reflect.Bool, reflect.String,
		reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
		reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr,
		reflect.Float32, reflect.Float64:
		return true
	}
	return false
}

func makeSliceArshaler(t reflect.Type) *arshaler {
	var fncs arshaler
	var (
		once    sync.Once
		valFncs *arshaler
	)
	init := func() {
		valFncs = lookupArshaler(t.Elem())
	}
	fncs.marshal = func(enc *jsontext.Encoder, va addressableValue, mo *jsonopts.Struct) error {
		// Check for cycles.
		xe := export.Encoder(enc)
		if xe.Tokens.Depth() > startDetectingCyclesAfter {
			if err := visitPointer(&xe.SeenPointers, va.Value); err != nil {
				return newMarshalErrorBefore(enc, t, err)
			}
			defer leavePointer(&xe.SeenPointers, va.Value)
		}

		emitNull := mo.Flags.Get(jsonflags.FormatNilSliceAsNull)
		if mo.Format != "" && mo.FormatDepth == xe.Tokens.Depth() {
			switch mo.Format {
			case "emitnull":
				emitNull = true
				mo.Format = ""
			case "emitempty":
				emitNull = false
				mo.Format = ""
			default:
				return newInvalidFormatError(enc, t)
			}
		}

		// Handle empty slices.
		n := va.Len()
		if n == 0 {
			if emitNull && va.IsNil() {
				return enc.WriteToken(jsontext.Null)
			}
			// Optimize for marshaling an empty slice without any preceding whitespace.
			if optimizeCommon && !mo.Flags.Get(jsonflags.AnyWhitespace) && !xe.Tokens.Last.NeedObjectName() {
				xe.Buf = append(xe.Tokens.MayAppendDelim(xe.Buf, '['), "[]"...)
				xe.Tokens.Last.Increment()
				if xe.NeedFlush() {
					return xe.Flush()
				}
				return nil
			}
		}

		once.Do(init)
		if err := enc.WriteToken(jsontext.BeginArray); err != nil {
			return err
		}
		marshal := valFncs.marshal
		if mo.Marshalers != nil {
			marshal, _ = mo.Marshalers.(*Marshalers).lookup(marshal, t.Elem())
		}
		for i := range n {
			v := addressableValue{va.Index(i), false} // indexed slice element is always addressable
			if err := marshal(enc, v, mo); err != nil {
				return err
			}
		}
		if err := enc.WriteToken(jsontext.EndArray); err != nil {
			return err
		}
		return nil
	}
	emptySlice := reflect.MakeSlice(t, 0, 0)
	fncs.unmarshal = func(dec *jsontext.Decoder, va addressableValue, uo *jsonopts.Struct) error {
		xd := export.Decoder(dec)
		if uo.Format != "" && uo.FormatDepth == xd.Tokens.Depth() {
			switch uo.Format {
			case "emitnull", "emitempty":
				uo.Format = "" // only relevant for marshaling
			default:
				return newInvalidFormatError(dec, t)
			}
		}

		tok, err := dec.ReadToken()
		if err != nil {
			return err
		}
		k := tok.Kind()
		switch k {
		case 'n':
			va.SetZero()
			return nil
		case '[':
			once.Do(init)
			unmarshal := valFncs.unmarshal
			if uo.Unmarshalers != nil {
				unmarshal, _ = uo.Unmarshalers.(*Unmarshalers).lookup(unmarshal, t.Elem())
			}
			mustZero := true // we do not know the cleanliness of unused capacity
			cap := va.Cap()
			if cap > 0 {
				va.SetLen(cap)
			}
			var i int
			var errUnmarshal error
			for dec.PeekKind() != ']' {
				if i == cap {
					va.Value.Grow(1)
					cap = va.Cap()
					va.SetLen(cap)
					mustZero = false // reflect.Value.Grow ensures new capacity is zero-initialized
				}
				v := addressableValue{va.Index(i), false} // indexed slice element is always addressable
				i++
				if mustZero && !uo.Flags.Get(jsonflags.MergeWithLegacySemantics) {
					v.SetZero()
				}
				if err := unmarshal(dec, v, uo); err != nil {
					if isFatalError(err, uo.Flags) {
						va.SetLen(i)
						return err
					}
					errUnmarshal = cmp.Or(errUnmarshal, err)
				}
			}
			if i == 0 {
				va.Set(emptySlice)
			} else {
				va.SetLen(i)
			}
			if _, err := dec.ReadToken(); err != nil {
				return err
			}
			return errUnmarshal
		}
		return newUnmarshalErrorAfterWithSkipping(dec, t, nil)
	}
	return &fncs
}

var errArrayUnderflow = errors.New("too few array elements")
var errArrayOverflow = errors.New("too many array elements")

func makeArrayArshaler(t reflect.Type) *arshaler {
	var fncs arshaler
	var (
		once    sync.Once
		valFncs *arshaler
	)
	init := func() {
		valFncs = lookupArshaler(t.Elem())
	}
	n := t.Len()
	fncs.marshal = func(enc *jsontext.Encoder, va addressableValue, mo *jsonopts.Struct) error {
		xe := export.Encoder(enc)
		if mo.Format != "" && mo.FormatDepth == xe.Tokens.Depth() {
			return newInvalidFormatError(enc, t)
		}
		once.Do(init)
		if err := enc.WriteToken(jsontext.BeginArray); err != nil {
			return err
		}
		marshal := valFncs.marshal
		if mo.Marshalers != nil {
			marshal, _ = mo.Marshalers.(*Marshalers).lookup(marshal, t.Elem())
		}
		for i := range n {
			v := addressableValue{va.Index(i), va.forcedAddr} // indexed array element is addressable if array is addressable
			if err := marshal(enc, v, mo); err != nil {
				return err
			}
		}
		if err := enc.WriteToken(jsontext.EndArray); err != nil {
			return err
		}
		return nil
	}
	fncs.unmarshal = func(dec *jsontext.Decoder, va addressableValue, uo *jsonopts.Struct) error {
		xd := export.Decoder(dec)
		if uo.Format != "" && uo.FormatDepth == xd.Tokens.Depth() {
			return newInvalidFormatError(dec, t)
		}
		tok, err := dec.ReadToken()
		if err != nil {
			return err
		}
		k := tok.Kind()
		switch k {
		case 'n':
			if !uo.Flags.Get(jsonflags.MergeWithLegacySemantics) {
				va.SetZero()
			}
			return nil
		case '[':
			once.Do(init)
			unmarshal := valFncs.unmarshal
			if uo.Unmarshalers != nil {
				unmarshal, _ = uo.Unmarshalers.(*Unmarshalers).lookup(unmarshal, t.Elem())
			}
			var i int
			var errUnmarshal error
			for dec.PeekKind() != ']' {
				if i >= n {
					if err := dec.SkipValue(); err != nil {
						return err
					}
					err = errArrayOverflow
					continue
				}
				v := addressableValue{va.Index(i), va.forcedAddr} // indexed array element is addressable if array is addressable
				if !uo.Flags.Get(jsonflags.MergeWithLegacySemantics) {
					v.SetZero()
				}
				if err := unmarshal(dec, v, uo); err != nil {
					if isFatalError(err, uo.Flags) {
						return err
					}
					errUnmarshal = cmp.Or(errUnmarshal, err)
				}
				i++
			}
			for ; i < n; i++ {
				va.Index(i).SetZero()
				err = errArrayUnderflow
			}
			if _, err := dec.ReadToken(); err != nil {
				return err
			}
			if err != nil && !uo.Flags.Get(jsonflags.UnmarshalArrayFromAnyLength) {
				return newUnmarshalErrorAfter(dec, t, err)
			}
			return errUnmarshal
		}
		return newUnmarshalErrorAfterWithSkipping(dec, t, nil)
	}
	return &fncs
}

func makePointerArshaler(t reflect.Type) *arshaler {
	var fncs arshaler
	var (
		once    sync.Once
		valFncs *arshaler
	)
	init := func() {
		valFncs = lookupArshaler(t.Elem())
	}
	fncs.marshal = func(enc *jsontext.Encoder, va addressableValue, mo *jsonopts.Struct) error {
		// Check for cycles.
		xe := export.Encoder(enc)
		if xe.Tokens.Depth() > startDetectingCyclesAfter {
			if err := visitPointer(&xe.SeenPointers, va.Value); err != nil {
				return newMarshalErrorBefore(enc, t, err)
			}
			defer leavePointer(&xe.SeenPointers, va.Value)
		}

		// NOTE: Struct.Format is forwarded to underlying marshal.
		if va.IsNil() {
			return enc.WriteToken(jsontext.Null)
		}
		once.Do(init)
		marshal := valFncs.marshal
		if mo.Marshalers != nil {
			marshal, _ = mo.Marshalers.(*Marshalers).lookup(marshal, t.Elem())
		}
		v := addressableValue{va.Elem(), false} // dereferenced pointer is always addressable
		return marshal(enc, v, mo)
	}
	fncs.unmarshal = func(dec *jsontext.Decoder, va addressableValue, uo *jsonopts.Struct) error {
		// NOTE: Struct.Format is forwarded to underlying unmarshal.
		if dec.PeekKind() == 'n' {
			if _, err := dec.ReadToken(); err != nil {
				return err
			}
			va.SetZero()
			return nil
		}
		once.Do(init)
		unmarshal := valFncs.unmarshal
		if uo.Unmarshalers != nil {
			unmarshal, _ = uo.Unmarshalers.(*Unmarshalers).lookup(unmarshal, t.Elem())
		}
		if va.IsNil() {
			va.Set(reflect.New(t.Elem()))
		}
		v := addressableValue{va.Elem(), false} // dereferenced pointer is always addressable
		if err := unmarshal(dec, v, uo); err != nil {
			return err
		}
		if uo.Flags.Get(jsonflags.StringifyWithLegacySemantics) &&
			uo.Flags.Get(jsonflags.StringifyNumbers|jsonflags.StringifyBoolsAndStrings) {
			// A JSON null quoted within a JSON string should take effect
			// within the pointer value, rather than the indirect value.
			//
			// TODO: This does not correctly handle escaped nulls
			// (e.g., "\u006e\u0075\u006c\u006c"), but is good enough
			// for such an esoteric use case of the `string` option.
			if string(export.Decoder(dec).PreviousTokenOrValue()) == `"null"` {
				va.SetZero()
			}
		}
		return nil
	}
	return &fncs
}

func makeInterfaceArshaler(t reflect.Type) *arshaler {
	// NOTE: Values retrieved from an interface are not addressable,
	// so we shallow copy the values to make them addressable and
	// store them back into the interface afterwards.

	var fncs arshaler
	var whichMarshaler reflect.Type
	for _, iface := range allMarshalerTypes {
		if t.Implements(iface) {
			whichMarshaler = t
			break
		}
	}
	fncs.marshal = func(enc *jsontext.Encoder, va addressableValue, mo *jsonopts.Struct) error {
		xe := export.Encoder(enc)
		if mo.Format != "" && mo.FormatDepth == xe.Tokens.Depth() {
			return newInvalidFormatError(enc, t)
		}
		if va.IsNil() {
			return enc.WriteToken(jsontext.Null)
		} else if mo.Flags.Get(jsonflags.CallMethodsWithLegacySemantics) && whichMarshaler != nil {
			// The marshaler for a pointer never calls the method on a nil receiver.
			// Wrap the nil pointer within a struct type so that marshal
			// instead appears on a value receiver and may be called.
			if va.Elem().Kind() == reflect.Pointer && va.Elem().IsNil() {
				v2 := newAddressableValue(whichMarshaler)
				switch whichMarshaler {
				case jsonMarshalerToType:
					v2.Set(reflect.ValueOf(struct{ MarshalerTo }{va.Elem().Interface().(MarshalerTo)}))
				case jsonMarshalerType:
					v2.Set(reflect.ValueOf(struct{ Marshaler }{va.Elem().Interface().(Marshaler)}))
				case textAppenderType:
					v2.Set(reflect.ValueOf(struct{ encoding.TextAppender }{va.Elem().Interface().(encoding.TextAppender)}))
				case textMarshalerType:
					v2.Set(reflect.ValueOf(struct{ encoding.TextMarshaler }{va.Elem().Interface().(encoding.TextMarshaler)}))
				}
				va = v2
			}
		}
		v := newAddressableValue(va.Elem().Type())
		v.Set(va.Elem())
		marshal := lookupArshaler(v.Type()).marshal
		if mo.Marshalers != nil {
			marshal, _ = mo.Marshalers.(*Marshalers).lookup(marshal, v.Type())
		}
		// Optimize for the any type if there are no special options.
		if optimizeCommon &&
			t == anyType && !mo.Flags.Get(jsonflags.StringifyNumbers|jsonflags.StringifyBoolsAndStrings) && mo.Format == "" &&
			(mo.Marshalers == nil || !mo.Marshalers.(*Marshalers).fromAny) {
			return marshalValueAny(enc, va.Elem().Interface(), mo)
		}
		return marshal(enc, v, mo)
	}
	fncs.unmarshal = func(dec *jsontext.Decoder, va addressableValue, uo *jsonopts.Struct) error {
		xd := export.Decoder(dec)
		if uo.Format != "" && uo.FormatDepth == xd.Tokens.Depth() {
			return newInvalidFormatError(dec, t)
		}
		if uo.Flags.Get(jsonflags.MergeWithLegacySemantics) && !va.IsNil() {
			// Legacy merge behavior is difficult to explain.
			// In general, it only merges for non-nil pointer kinds.
			// As a special case, unmarshaling a JSON null into a pointer
			// sets a concrete nil pointer of the underlying type
			// (rather than setting the interface value itself to nil).
			e := va.Elem()
			if e.Kind() == reflect.Pointer && !e.IsNil() {
				if dec.PeekKind() == 'n' && e.Elem().Kind() == reflect.Pointer {
					if _, err := dec.ReadToken(); err != nil {
						return err
					}
					va.Elem().Elem().SetZero()
					return nil
				}
			} else {
				va.SetZero()
			}
		}
		if dec.PeekKind() == 'n' {
			if _, err := dec.ReadToken(); err != nil {
				return err
			}
			va.SetZero()
			return nil
		}
		var v addressableValue
		if va.IsNil() {
			// Optimize for the any type if there are no special options.
			// We do not care about stringified numbers since JSON strings
			// are always unmarshaled into an any value as Go strings.
			// Duplicate name check must be enforced since unmarshalValueAny
			// does not implement merge semantics.
			if optimizeCommon &&
				t == anyType && !uo.Flags.Get(jsonflags.AllowDuplicateNames) && uo.Format == "" &&
				(uo.Unmarshalers == nil || !uo.Unmarshalers.(*Unmarshalers).fromAny) {
				v, err := unmarshalValueAny(dec, uo)
				// We must check for nil interface values up front.
				// See https://go.dev/issue/52310.
				if v != nil {
					va.Set(reflect.ValueOf(v))
				}
				return err
			}

			k := dec.PeekKind()
			if !isAnyType(t) {
				return newUnmarshalErrorBeforeWithSkipping(dec, t, internal.ErrNilInterface)
			}
			switch k {
			case 'f', 't':
				v = newAddressableValue(boolType)
			case '"':
				v = newAddressableValue(stringType)
			case '0':
				if uo.Flags.Get(jsonflags.UnmarshalAnyWithRawNumber) {
					v = addressableValue{reflect.ValueOf(internal.NewRawNumber()).Elem(), true}
				} else {
					v = newAddressableValue(float64Type)
				}
			case '{':
				v = newAddressableValue(mapStringAnyType)
			case '[':
				v = newAddressableValue(sliceAnyType)
			default:
				// If k is invalid (e.g., due to an I/O or syntax error), then
				// that will be cached by PeekKind and returned by ReadValue.
				// If k is '}' or ']', then ReadValue must error since
				// those are invalid kinds at the start of a JSON value.
				_, err := dec.ReadValue()
				return err
			}
		} else {
			// Shallow copy the existing value to keep it addressable.
			// Any mutations at the top-level of the value will be observable
			// since we always store this value back into the interface value.
			v = newAddressableValue(va.Elem().Type())
			v.Set(va.Elem())
		}
		unmarshal := lookupArshaler(v.Type()).unmarshal
		if uo.Unmarshalers != nil {
			unmarshal, _ = uo.Unmarshalers.(*Unmarshalers).lookup(unmarshal, v.Type())
		}
		err := unmarshal(dec, v, uo)
		va.Set(v.Value)
		return err
	}
	return &fncs
}

// isAnyType reports wether t is equivalent to the any interface type.
func isAnyType(t reflect.Type) bool {
	// This is forward compatible if the Go language permits type sets within
	// ordinary interfaces where an interface with zero methods does not
	// necessarily mean it can hold every possible Go type.
	// See https://go.dev/issue/45346.
	return t == anyType || anyType.Implements(t)
}

func makeInvalidArshaler(t reflect.Type) *arshaler {
	var fncs arshaler
	fncs.marshal = func(enc *jsontext.Encoder, va addressableValue, mo *jsonopts.Struct) error {
		return newMarshalErrorBefore(enc, t, nil)
	}
	fncs.unmarshal = func(dec *jsontext.Decoder, va addressableValue, uo *jsonopts.Struct) error {
		return newUnmarshalErrorBefore(dec, t, nil)
	}
	return &fncs
}

func stringOrNumberKind(isString bool) jsontext.Kind {
	if isString {
		return '"'
	} else {
		return '0'
	}
}

type uintSet64 uint64

func (s uintSet64) has(i uint) bool { return s&(1<<i) > 0 }
func (s *uintSet64) set(i uint)     { *s |= 1 << i }

// uintSet is a set of unsigned integers.
// It is optimized for most integers being close to zero.
type uintSet struct {
	lo uintSet64
	hi []uintSet64
}

// has reports whether i is in the set.
func (s *uintSet) has(i uint) bool {
	if i < 64 {
		return s.lo.has(i)
	} else {
		i -= 64
		iHi, iLo := int(i/64), i%64
		return iHi < len(s.hi) && s.hi[iHi].has(iLo)
	}
}

// insert inserts i into the set and reports whether it was the first insertion.
func (s *uintSet) insert(i uint) bool {
	// TODO: Make this inlinable at least for the lower 64-bit case.
	if i < 64 {
		has := s.lo.has(i)
		s.lo.set(i)
		return !has
	} else {
		i -= 64
		iHi, iLo := int(i/64), i%64
		if iHi >= len(s.hi) {
			s.hi = append(s.hi, make([]uintSet64, iHi+1-len(s.hi))...)
			s.hi = s.hi[:cap(s.hi)]
		}
		has := s.hi[iHi].has(iLo)
		s.hi[iHi].set(iLo)
		return !has
	}
}
