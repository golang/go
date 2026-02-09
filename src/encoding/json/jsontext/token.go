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
type Token struct {
	nonComparable

	// Tokens can exist in either a "raw" or an "exact" form.
	// Tokens produced by the Decoder are in the "raw" form.
	// Tokens returned by constructors are usually in the "exact" form.
	// The Encoder accepts Tokens in either the "raw" or "exact" form.
	//
	// The following chart shows the possible values for each Token type:
	//	╔═════════════════╦════════════╤════════════╤════════════╗
	//	║ Token type      ║ raw field  │ str field  │ num field  ║
	//	╠═════════════════╬════════════╪════════════╪════════════╣
	//	║ null   (raw)    ║ "null"     │ ""         │ 0          ║
	//	║ false  (raw)    ║ "false"    │ ""         │ 0          ║
	//	║ true   (raw)    ║ "true"     │ ""         │ 0          ║
	//	║ string (raw)    ║ non-empty  │ ""         │ offset     ║
	//	║ string (string) ║ nil        │ non-empty  │ 0          ║
	//	║ number (raw)    ║ non-empty  │ ""         │ offset     ║
	//	║ number (float)  ║ nil        │ "f"        │ non-zero   ║
	//	║ number (int64)  ║ nil        │ "i"        │ non-zero   ║
	//	║ number (uint64) ║ nil        │ "u"        │ non-zero   ║
	//	║ object (delim)  ║ "{" or "}" │ ""         │ 0          ║
	//	║ array  (delim)  ║ "[" or "]" │ ""         │ 0          ║
	//	╚═════════════════╩════════════╧════════════╧════════════╝
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
	// Otherwise, it is "f", "i", or "u" if num should be interpreted
	// as a float64, int64, or uint64, respectively.
	str string

	// num is a float64, int64, or uint64 stored as a uint64 value.
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

// Float constructs a Token representing a JSON number.
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

// Clone makes a copy of the Token such that its value remains valid
// even after a subsequent [Decoder.Read] call.
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
		return jsonwire.AppendQuote(dst, t.str, flags)
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
		case 'f':
			return string(jsonwire.AppendFloat(nil, math.Float64frombits(t.num), 64)), nil
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
		case 'f':
			return jsonwire.AppendFloat(dst, math.Float64frombits(t.num), 64), nil
		case 'i':
			return strconv.AppendInt(dst, int64(t.num), 10), nil
		case 'u':
			return strconv.AppendUint(dst, uint64(t.num), 10), nil
		}
	}

	panic("invalid JSON token kind: " + t.Kind().String())
}

// Float returns the floating-point value for a JSON number.
// It returns a NaN, +Inf, or -Inf value for any JSON string
// with the values "NaN", "Infinity", or "-Infinity".
// It panics for all other cases.
func (t Token) Float() float64 {
	if raw := t.raw; raw != nil {
		// Handle raw number value.
		if uint64(raw.previousOffsetStart()) != t.num {
			panic(invalidTokenPanic)
		}
		buf := raw.previousBuffer()
		if Kind(buf[0]).normalize() == '0' {
			fv, _ := jsonwire.ParseFloat(buf, 64)
			return fv
		}
	} else if t.num != 0 {
		// Handle exact number value.
		switch t.str[0] {
		case 'f':
			return math.Float64frombits(t.num)
		case 'i':
			return float64(int64(t.num))
		case 'u':
			return float64(uint64(t.num))
		}
	}

	// Handle string values with "NaN", "Infinity", or "-Infinity".
	if t.Kind() == '"' {
		switch t.String() {
		case "NaN":
			return math.NaN()
		case "Infinity":
			return math.Inf(+1)
		case "-Infinity":
			return math.Inf(-1)
		}
	}

	panic("invalid JSON token kind: " + t.Kind().String())
}

// Int returns the signed integer value for a JSON number.
// The fractional component of any number is ignored (truncation toward zero).
// Any number beyond the representation of an int64 will be saturated
// to the closest representable value.
// It panics if the token kind is not a JSON number.
func (t Token) Int() int64 {
	if raw := t.raw; raw != nil {
		// Handle raw integer value.
		if uint64(raw.previousOffsetStart()) != t.num {
			panic(invalidTokenPanic)
		}
		neg := false
		buf := raw.previousBuffer()
		if len(buf) > 0 && buf[0] == '-' {
			neg, buf = true, buf[1:]
		}
		if numAbs, ok := jsonwire.ParseUint(buf); ok {
			if neg {
				if numAbs > -minInt64 {
					return minInt64
				}
				return -1 * int64(numAbs)
			} else {
				if numAbs > +maxInt64 {
					return maxInt64
				}
				return +1 * int64(numAbs)
			}
		}
	} else if t.num != 0 {
		// Handle exact integer value.
		switch t.str[0] {
		case 'i':
			return int64(t.num)
		case 'u':
			if t.num > maxInt64 {
				return maxInt64
			}
			return int64(t.num)
		}
	}

	// Handle JSON number that is a floating-point value.
	if t.Kind() == '0' {
		switch fv := t.Float(); {
		case fv >= maxInt64:
			return maxInt64
		case fv <= minInt64:
			return minInt64
		default:
			return int64(fv) // truncation toward zero
		}
	}

	panic("invalid JSON token kind: " + t.Kind().String())
}

// Uint returns the unsigned integer value for a JSON number.
// The fractional component of any number is ignored (truncation toward zero).
// Any number beyond the representation of an uint64 will be saturated
// to the closest representable value.
// It panics if the token kind is not a JSON number.
func (t Token) Uint() uint64 {
	// NOTE: This accessor returns 0 for any negative JSON number,
	// which might be surprising, but is at least consistent with the behavior
	// of saturating out-of-bounds numbers to the closest representable number.

	if raw := t.raw; raw != nil {
		// Handle raw integer value.
		if uint64(raw.previousOffsetStart()) != t.num {
			panic(invalidTokenPanic)
		}
		neg := false
		buf := raw.previousBuffer()
		if len(buf) > 0 && buf[0] == '-' {
			neg, buf = true, buf[1:]
		}
		if num, ok := jsonwire.ParseUint(buf); ok {
			if neg {
				return minUint64
			}
			return num
		}
	} else if t.num != 0 {
		// Handle exact integer value.
		switch t.str[0] {
		case 'u':
			return t.num
		case 'i':
			if int64(t.num) < minUint64 {
				return minUint64
			}
			return uint64(int64(t.num))
		}
	}

	// Handle JSON number that is a floating-point value.
	if t.Kind() == '0' {
		switch fv := t.Float(); {
		case fv >= maxUint64:
			return maxUint64
		case fv <= minUint64:
			return minUint64
		default:
			return uint64(fv) // truncation toward zero
		}
	}

	panic("invalid JSON token kind: " + t.Kind().String())
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
		return "<invalid jsontext.Kind: " + jsonwire.QuoteRune(string(k)) + ">"
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
