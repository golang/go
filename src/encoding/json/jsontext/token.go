// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.jsonv2

package jsontext

import (
	"bytes"
	"errors"
	"math"
	"strconv"

	"encoding/json/internal/jsonflags"
	"encoding/json/internal/jsonwire"
)

// NOTE: Token is analogous to v1 json.Token.

const (
	maxInt64  = math.MaxInt64
	minInt64  = math.MinInt64
	maxUint64 = math.MaxUint64
	minUint64 = 0 // for consistency and readability purposes

	invalidTokenPanic = "invalid jsontext.Token; it has been voided by a subsequent json.Decoder call"
)

var errInvalidToken = errors.New("invalid jsontext.Token")

// Token represents a lexical JSON token, which may be one of the following:
//   - a JSON literal (i.e., null, true, or false)
//   - a JSON string (e.g., "hello, world!")
//   - a JSON number (e.g., 123.456)
//   - a begin or end delimiter for a JSON object (i.e., { or } )
//   - a begin or end delimiter for a JSON array (i.e., [ or ] )
//
// A Token cannot represent entire array or object values, while a [Value] can.
// There is no Token to represent commas and colons since
// these structural tokens can be inferred from the surrounding context.
//
// A Token stores data in one of two forms:
//
//   - As raw JSON text: backed by the internal buffer of the [Decoder]
//     and only ever produced by [Decoder.ReadToken].
//     Such a token is only valid until the next call to any method on that
//     [Decoder] (e.g., [Decoder.PeekKind], [Decoder.ReadToken],
//     [Decoder.ReadValue], or [Decoder.SkipValue]).
//     Call [Token.Clone] to copy the raw text into an independent allocation
//     that persists beyond subsequent [Decoder] calls.
//
//   - As a typed Go value: a self-contained representation produced by
//     the constructor functions (e.g., [String], [Int], [Uint], [Float]).
//     Such tokens are valid indefinitely and do not need to be cloned.
type Token struct {
	nonComparable

	// Tokens can exist in either a "raw" or an "exact" form.
	// Tokens produced by the Decoder are in the "raw" form.
	// Tokens returned by constructors are usually in the "exact" form.
	// The Encoder accepts Tokens in either the "raw" or "exact" form.
	//
	// The following chart shows the possible values for each Token type:
	//	╔══════════════════╦════════════╤════════════╤════════════╗
	//	║ Token type       ║ raw field  │ str field  │ num field  ║
	//	╠══════════════════╬════════════╪════════════╪════════════╣
	//	║ null   (raw)     ║ "null"     │ ""         │ 0          ║
	//	║ false  (raw)     ║ "false"    │ ""         │ 0          ║
	//	║ true   (raw)     ║ "true"     │ ""         │ 0          ║
	//	║ string (raw)     ║ non-empty  │ ""         │ offset     ║
	//	║ string (string)  ║ nil        │ non-empty  │ 0          ║
	//	║ number (raw)     ║ non-empty  │ ""         │ offset     ║
	//	║ number (float32) ║ nil        │ "F"        │ non-zero   ║
	//	║ number (float64) ║ nil        │ "f"        │ non-zero   ║
	//	║ number (int64)   ║ nil        │ "i"        │ non-zero   ║
	//	║ number (uint64)  ║ nil        │ "u"        │ non-zero   ║
	//	║ object (delim)   ║ "{" or "}" │ ""         │ 0          ║
	//	║ array  (delim)   ║ "[" or "]" │ ""         │ 0          ║
	//	╚══════════════════╩════════════╧════════════╧════════════╝
	//
	// Notes:
	//   - For tokens stored in "raw" form, the num field contains the
	//     absolute offset determined by raw.previousOffsetStart().
	//     The buffer itself is stored in raw.previousBuffer().
	//   - JSON literals and structural characters are always in the "raw" form.
	//   - JSON strings and numbers can be in either "raw" or "exact" forms.
	//   - The exact zero value of JSON strings and numbers in the "exact" forms
	//     have ambiguous representation. Thus, they are always represented
	//     in the "raw" form.

	// raw contains a reference to the raw decode buffer.
	// If non-nil, then its value takes precedence over str and num.
	// It is only valid if num == raw.previousOffsetStart().
	raw *decodeBuffer

	// str is the unescaped JSON string if num is zero.
	// Otherwise, it is "F", "f", "i", or "u" if num should be interpreted
	// as a float32, float64, int64, or uint64, respectively.
	str string

	// num is a float32, float64, int64, or uint64 stored as a uint64 value.
	// For floating-point values, it stores the raw IEEE-754 bit-pattern.
	// It is non-zero for any JSON number in the "exact" form.
	num uint64
}

// TODO: Does representing 1-byte delimiters as *decodeBuffer cause performance issues?

var (
	Null  Token = rawToken("null")
	False Token = rawToken("false")
	True  Token = rawToken("true")

	BeginObject Token = rawToken("{")
	EndObject   Token = rawToken("}")
	BeginArray  Token = rawToken("[")
	EndArray    Token = rawToken("]")

	zeroString Token = rawToken(`""`)
	zeroNumber Token = rawToken(`0`)

	nanString  Token = String("NaN")
	pinfString Token = String("Infinity")
	ninfString Token = String("-Infinity")
)

func rawToken(s string) Token {
	return Token{raw: &decodeBuffer{buf: []byte(s), prevStart: 0, prevEnd: len(s)}}
}

// Bool constructs a Token representing a JSON boolean.
func Bool(b bool) Token {
	if b {
		return True
	}
	return False
}

// String constructs a Token representing a JSON string.
// The provided string should contain valid UTF-8, otherwise invalid characters
// may be mangled as the Unicode replacement character.
func String(s string) Token {
	if len(s) == 0 {
		return zeroString
	}
	return Token{str: s}
}

// Float32 constructs a Token representing a JSON number as
// a 32-bit floating-point number formatted according to
// ECMA-262, 10th edition, section 7.1.12.1,
// with the exception that -0 is still formatted as -0.
// The values NaN, +Inf, and -Inf will be represented
// as a JSON string with the values "NaN", "Infinity", and "-Infinity".
//
// Note that most JSON libraries and standards assume that JSON numbers
// are 64-bit floating-point numbers. Use of 32-bit precision should
// only be used if the corresponding decoder knows that
// this JSON number token is expected to only have 32-bit precision.
// For all other situations, prefer using the [Float] constructor instead.
func Float32(n float32) Token {
	if n != 0 && !math.IsNaN(float64(n)) && !math.IsInf(float64(n), 0) {
		return Token{str: "F", num: uint64(math.Float32bits(n))}
	}
	return Float(float64(n)) // handles ±0, NaN, and ±Inf
}

// Float constructs a Token representing a JSON number as
// a 64-bit floating-point number formatted according to
// ECMA-262, 10th edition, section 7.1.12.1 and RFC 8785, section 3.2.2.3.
// with the exception that -0 is still formatted as -0.
// The values NaN, +Inf, and -Inf will be represented
// as a JSON string with the values "NaN", "Infinity", and "-Infinity".
func Float(n float64) Token {
	switch {
	case math.Float64bits(n) == 0:
		return zeroNumber
	case math.IsNaN(n):
		return nanString
	case math.IsInf(n, +1):
		return pinfString
	case math.IsInf(n, -1):
		return ninfString
	}
	return Token{str: "f", num: math.Float64bits(n)}
}

// Int constructs a Token representing a JSON number from an int64.
func Int(n int64) Token {
	if n == 0 {
		return zeroNumber
	}
	return Token{str: "i", num: uint64(n)}
}

// Uint constructs a Token representing a JSON number from a uint64.
func Uint(n uint64) Token {
	if n == 0 {
		return zeroNumber
	}
	return Token{str: "u", num: uint64(n)}
}

// Clone returns a copy of the token with a value that is not backed by the
// [Decoder] buffer and therefore remains valid past subsequent [Decoder] calls.
// It has no effect on tokens produced by constructor functions,
// since those are already self-contained.
func (t Token) Clone() Token {
	// TODO: Allow caller to avoid any allocations?
	if raw := t.raw; raw != nil {
		// Avoid copying globals.
		if t.raw.prevStart == 0 {
			switch t.raw {
			case Null.raw:
				return Null
			case False.raw:
				return False
			case True.raw:
				return True
			case BeginObject.raw:
				return BeginObject
			case EndObject.raw:
				return EndObject
			case BeginArray.raw:
				return BeginArray
			case EndArray.raw:
				return EndArray
			}
		}

		// This only ever needs to clone strings and numbers.
		if uint64(raw.previousOffsetStart()) != t.num {
			panic(invalidTokenPanic)
		}
		buf := bytes.Clone(raw.previousBuffer())
		return Token{raw: &decodeBuffer{buf: buf, prevStart: 0, prevEnd: len(buf)}}
	}
	return t
}

// Bool returns the value for a JSON boolean.
// It panics if the token kind is not a JSON boolean.
func (t Token) Bool() bool {
	switch t.raw {
	case True.raw:
		return true
	case False.raw:
		return false
	default:
		panic("invalid JSON token kind: " + t.Kind().String())
	}
}

// appendString appends a JSON string to dst and returns it.
// It panics if t is not a JSON string.
func (t Token) appendString(dst []byte, flags *jsonflags.Flags) ([]byte, error) {
	if raw := t.raw; raw != nil {
		// Handle raw string value.
		buf := raw.previousBuffer()
		if Kind(buf[0]) == '"' {
			if jsonwire.ConsumeSimpleString(buf) == len(buf) {
				return append(dst, buf...), nil
			}
			dst, _, err := jsonwire.ReformatString(dst, buf, flags)
			return dst, err
		}
	} else if len(t.str) != 0 && t.num == 0 {
		// Handle exact string value.
		return jsonwire.AppendQuote(dst, []byte(t.str), flags)
	}

	panic("invalid JSON token kind: " + t.Kind().String())
}

// String returns the unescaped string value for a JSON string.
// For other JSON kinds, this returns the raw JSON representation.
func (t Token) String() string {
	// This is inlinable to take advantage of "function outlining".
	// This avoids an allocation for the string(b) conversion
	// if the caller does not use the string in an escaping manner.
	// See https://blog.filippo.io/efficient-go-apis-with-the-inliner/
	s, b := t.string()
	if len(b) > 0 {
		return string(b)
	}
	return s
}
func (t Token) string() (string, []byte) {
	if raw := t.raw; raw != nil {
		if uint64(raw.previousOffsetStart()) != t.num {
			panic(invalidTokenPanic)
		}
		buf := raw.previousBuffer()
		if buf[0] == '"' {
			// TODO: Preserve ValueFlags in Token?
			isVerbatim := jsonwire.ConsumeSimpleString(buf) == len(buf)
			return "", jsonwire.UnquoteMayCopy(buf, isVerbatim)
		}
		// Handle tokens that are not JSON strings for fmt.Stringer.
		return "", buf
	}
	if len(t.str) != 0 && t.num == 0 {
		return t.str, nil
	}
	// Handle tokens that are not JSON strings for fmt.Stringer.
	if t.num > 0 {
		switch t.str[0] {
		case 'F':
			return string(jsonwire.AppendFloat(nil, float64(math.Float32frombits(uint32(t.num))), 32)), nil
		case 'f':
			return string(jsonwire.AppendFloat(nil, float64(math.Float64frombits(uint64(t.num))), 64)), nil
		case 'i':
			return strconv.FormatInt(int64(t.num), 10), nil
		case 'u':
			return strconv.FormatUint(uint64(t.num), 10), nil
		}
	}
	return "<invalid jsontext.Token>", nil
}

// appendNumber appends a JSON number to dst and returns it.
// It panics if t is not a JSON number.
func (t Token) appendNumber(dst []byte, flags *jsonflags.Flags) ([]byte, error) {
	if raw := t.raw; raw != nil {
		// Handle raw number value.
		buf := raw.previousBuffer()
		if Kind(buf[0]).normalize() == '0' {
			dst, _, err := jsonwire.ReformatNumber(dst, buf, flags)
			return dst, err
		}
	} else if t.num != 0 {
		// Handle exact number value.
		switch t.str[0] {
		case 'F':
			return jsonwire.AppendFloat(dst, float64(math.Float32frombits(uint32(t.num))), 32), nil
		case 'f':
			return jsonwire.AppendFloat(dst, float64(math.Float64frombits(uint64(t.num))), 64), nil
		case 'i':
			return strconv.AppendInt(dst, int64(t.num), 10), nil
		case 'u':
			return strconv.AppendUint(dst, uint64(t.num), 10), nil
		}
	}

	panic("invalid JSON token kind: " + t.Kind().String())
}

// Float32 returns the floating-point value for a JSON number
// parsed according to 32 bits of precision.
//
// If the JSON number is outside the representable range of a float32,
// it returns +Inf or -Inf along with an error
// that matches [strconv.ErrRange] according to [errors.Is].
//
// It returns a NaN, +Inf, or -Inf value for any JSON string
// with the values "NaN", "Infinity", or "-Infinity".
//
// It panics if the token kind is not a JSON number
// or a JSON string with the aforementioned values.
//
// Note that most JSON libraries and standards assume that JSON numbers
// are 64-bit floating-point numbers.
// This method should only be used if the caller knows
// from other context that this token is a JSON number
// formatted only to 32 bits of precision (such as being encoded
// using the [Float32] constructor). For all other situations,
// prefer using the [Token.Float] accessor instead.
func (t Token) Float32() (float32, error) {
	f, err := t.float(32)
	return float32(f), err
}

// Float returns the floating-point value for a JSON number
// parsed according to 64 bits of precision.
//
// If the JSON number is outside the representable range of a float64,
// it returns +Inf or -Inf along with an error
// that matches [strconv.ErrRange] according to [errors.Is].
//
// It returns a NaN, +Inf, or -Inf value for any JSON string
// with the values "NaN", "Infinity", or "-Infinity".
//
// It panics if the token kind is not a JSON number
// or a JSON string with the aforementioned values.
func (t Token) Float() (float64, error) {
	f, err := t.float(64)
	return float64(f), err
}

func (t Token) float(bits int) (float64, error) {
	if raw := t.raw; raw != nil {
		// Handle raw JSON number value.
		if uint64(raw.previousOffsetStart()) != t.num {
			panic(invalidTokenPanic)
		}
		buf := raw.previousBuffer()
		if Kind(buf[0]).normalize() == '0' {
			fv, err := strconv.ParseFloat(string(buf), bits)
			if err != nil {
				err = &numError{accessor: "Float", value: t.String(), err: errors.Unwrap(err)} // only ever ErrRange
			}
			return fv, err
		}
	} else if t.num != 0 {
		// Handle typed Go number value.
		switch t.str[0] {
		case 'F':
			return float64(math.Float32frombits(uint32(t.num))), nil
		case 'f':
			f64 := float64(math.Float64frombits(uint64(t.num)))
			if bits == 32 && !math.IsInf(f64, 0) && math.IsInf(float64(float32(f64)), 0) {
				return f64, &numError{accessor: "Float", value: t.String(), err: strconv.ErrRange}
			}
			return f64, nil
		case 'i':
			return float64(int64(t.num)), nil // NOTE: This may lead to loss of precision.
		case 'u':
			return float64(uint64(t.num)), nil // NOTE: This may lead to loss of precision.
		}
	}

	// Handle string values with "NaN", "Infinity", or "-Infinity".
	if t.Kind() == '"' {
		switch t.String() {
		case "NaN":
			return math.NaN(), nil
		case "Infinity":
			return math.Inf(+1), nil
		case "-Infinity":
			return math.Inf(-1), nil
		}
		// TODO: Should this be a error instead of a panic?
		// We can safely switch from a panic to an error in the future.
	}

	panic("invalid JSON token kind: " + t.Kind().String())
}

// Int returns the signed integer value for a JSON number.
//
// It reports an error that matches [strconv.ErrSyntax] according to [errors.Is]
// if the JSON number does not match the restricted grammar of just a signed integer.
// It reports an error that matches [strconv.ErrRange] according to [errors.Is]
// if the JSON number is a signed integer, but outside the range of an int64.
// Even if an error is reported, a reasonable value is still returned.
// The fractional component of any number is ignored (truncation toward zero).
// Any number beyond the representation of an int64 will be saturated
// to the closest representable value.
//
// It panics if the token kind is not a JSON number.
func (t Token) Int() (int64, error) {
	if raw := t.raw; raw != nil {
		// Handle raw JSON number value.
		if uint64(raw.previousOffsetStart()) != t.num {
			panic(invalidTokenPanic)
		}
		buf := raw.previousBuffer()
		if len(buf) > 0 && buf[0] == '-' {
			// Prospectively parse a negative integer.
			switch abs, ok := jsonwire.ParseUint(buf[len("-"):]); {
			case abs > -minInt64:
				return minInt64, &numError{accessor: "Int", value: t.String(), err: strconv.ErrRange}
			case ok:
				return -1 * int64(abs), nil
			}
		} else {
			// Prospectively parse a non-negative integer.
			switch abs, ok := jsonwire.ParseUint(buf); {
			case abs > +maxInt64:
				return maxInt64, &numError{accessor: "Int", value: t.String(), err: strconv.ErrRange}
			case ok:
				return +1 * int64(abs), nil
			}
		}
		// This is not a signed integer, which implies ErrSyntax.
		if Kind(buf[0]).normalize() == '0' {
			f64, _ := strconv.ParseFloat(string(buf), 64)
			return f64toi64(f64), &numError{accessor: "Int", value: t.String(), err: strconv.ErrSyntax}
		}
	} else if t.num != 0 {
		// Handle typed Go number value.
		switch t.str[0] {
		case 'i':
			return int64(t.num), nil
		case 'u':
			if t.num > maxInt64 {
				return maxInt64, &numError{accessor: "Int", value: t.String(), err: strconv.ErrRange}
			}
			return int64(t.num), nil
		case 'f', 'F':
			f64 := float64(math.Float64frombits(uint64(t.num)))
			if t.str[0] == 'F' {
				f64 = float64(math.Float32frombits(uint32(t.num)))
			}
			switch i64 := f64toi64(f64); {
			case math.IsNaN(f64), math.Trunc(f64) != f64:
				return i64, &numError{accessor: "Int", value: t.String(), err: strconv.ErrSyntax}
			case (i64 == minInt64 && f64 < minInt64) || (i64 == maxInt64 && f64 > maxInt64):
				return i64, &numError{accessor: "Int", value: t.String(), err: strconv.ErrRange}
			default:
				return i64, nil
			}
		}
	}

	panic("invalid JSON token kind: " + t.Kind().String())
}

func f64toi64(f64 float64) int64 {
	switch {
	case math.IsNaN(f64):
		return 0
	case f64 >= maxInt64+1:
		return maxInt64
	case f64 < minInt64:
		return minInt64
	default:
		return int64(f64) // NOTE: This may lead to loss of precision.
	}
}

// Uint returns the unsigned integer value for a JSON number.
//
// It reports an error that matches [strconv.ErrSyntax] if the JSON number
// does not match the restricted grammar of just an unsigned integer.
// It reports an error that matches [strconv.ErrRange] if the JSON number
// is an unsigned integer, but outside the representable range of a uint64.
// Even if an error is reported, a reasonable value is still returned.
// The fractional component of any number is ignored (truncation toward zero).
// Any number beyond the representation of a uint64 will be saturated
// to the closest representable value.
//
// It panics if the token kind is not a JSON number.
func (t Token) Uint() (uint64, error) {
	// NOTE: This accessor returns 0 for any negative JSON number,
	// which might be surprising, but is at least consistent with the behavior
	// of saturating out-of-bounds numbers to the closest representable number.
	// We report ErrSyntax instead of ErrRange since the grammar for
	// an unsigned integer does not permit a negative sign.

	if raw := t.raw; raw != nil {
		// Handle raw JSON number value.
		if uint64(raw.previousOffsetStart()) != t.num {
			panic(invalidTokenPanic)
		}
		buf := raw.previousBuffer()
		// Prospectively parse an unsigned integer.
		switch abs, ok := jsonwire.ParseUint(buf); {
		case ok:
			return abs, nil
		case abs == maxUint64: // implies overflows
			return maxUint64, &numError{accessor: "Uint", value: t.String(), err: strconv.ErrRange}
		}
		// This is not an unsigned integer, which implies ErrSyntax.
		if Kind(buf[0]).normalize() == '0' {
			f64, _ := strconv.ParseFloat(string(buf), 64)
			return f64tou64(f64), &numError{accessor: "Uint", value: t.String(), err: strconv.ErrSyntax}
		}
	} else if t.num != 0 {
		// Handle typed Go number value.
		switch t.str[0] {
		case 'u':
			return t.num, nil
		case 'i':
			if int64(t.num) < minUint64 {
				return minUint64, &numError{accessor: "Uint", value: t.String(), err: strconv.ErrSyntax}
			}
			return uint64(int64(t.num)), nil
		case 'f', 'F':
			f64 := float64(math.Float64frombits(uint64(t.num)))
			if t.str[0] == 'F' {
				f64 = float64(math.Float32frombits(uint32(t.num)))
			}
			switch u64 := f64tou64(f64); {
			case math.IsNaN(f64), math.Trunc(f64) != f64, math.Signbit(f64):
				return u64, &numError{accessor: "Uint", value: t.String(), err: strconv.ErrSyntax}
			case (u64 == minUint64 && f64 < minUint64) || (u64 == maxUint64 && f64 > maxUint64):
				return u64, &numError{accessor: "Uint", value: t.String(), err: strconv.ErrRange}
			default:
				return u64, nil
			}
		}
	}

	panic("invalid JSON token kind: " + t.Kind().String())
}

func f64tou64(f64 float64) uint64 {
	switch {
	case math.IsNaN(f64):
		return 0
	case f64 >= maxUint64+1:
		return maxUint64
	case f64 < minUint64:
		return minUint64
	default:
		return uint64(f64) // NOTE: This may lead to loss of precision.
	}
}

// Kind returns the token kind.
func (t Token) Kind() Kind {
	switch {
	case t.raw != nil:
		raw := t.raw
		if uint64(raw.previousOffsetStart()) != t.num {
			panic(invalidTokenPanic)
		}
		return Kind(t.raw.buf[raw.prevStart]).normalize()
	case t.num != 0:
		return '0'
	case len(t.str) != 0:
		return '"'
	default:
		return invalidKind
	}
}

// A Kind represents the kind of a JSON token.
//
// Kind represents each possible JSON token kind with a single byte,
// which is conveniently the first byte of that kind's grammar
// with the restriction that numbers always be represented with '0'.
type Kind byte

const (
	KindInvalid     Kind = 0   // invalid kind
	KindNull        Kind = 'n' // null
	KindFalse       Kind = 'f' // false
	KindTrue        Kind = 't' // true
	KindString      Kind = '"' // string
	KindNumber      Kind = '0' // number
	KindBeginObject Kind = '{' // begin object
	KindEndObject   Kind = '}' // end object
	KindBeginArray  Kind = '[' // begin array
	KindEndArray    Kind = ']' // end array
)

const invalidKind Kind = 0

// String prints the kind in a humanly readable fashion.
func (k Kind) String() string {
	switch k {
	case 0:
		return "invalid"
	case 'n':
		return "null"
	case 'f':
		return "false"
	case 't':
		return "true"
	case '"':
		return "string"
	case '0':
		return "number"
	case '{':
		return "{"
	case '}':
		return "}"
	case '[':
		return "["
	case ']':
		return "]"
	default:
		return "<invalid jsontext.Kind: " + jsonwire.QuoteRune([]byte{byte(k)}) + ">"
	}
}

var normKind = [256]Kind{
	'n': 'n',
	'f': 'f',
	't': 't',
	'"': '"',
	'{': '{',
	'}': '}',
	'[': '[',
	']': ']',
	'-': '0',
	'0': '0',
	'1': '0',
	'2': '0',
	'3': '0',
	'4': '0',
	'5': '0',
	'6': '0',
	'7': '0',
	'8': '0',
	'9': '0',
}

// normalize coalesces all possible starting characters of a number as just '0',
// and converts all invalid kinds to 0.
func (k Kind) normalize() Kind {
	// A lookup table keeps the inlining cost as low as possible.
	return normKind[k]
}
