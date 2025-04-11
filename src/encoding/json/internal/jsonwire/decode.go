// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.jsonv2

package jsonwire

import (
	"io"
	"math"
	"slices"
	"strconv"
	"unicode/utf16"
	"unicode/utf8"
)

type ValueFlags uint

const (
	_ ValueFlags = (1 << iota) / 2 // powers of two starting with zero

	stringNonVerbatim  // string cannot be naively treated as valid UTF-8
	stringNonCanonical // string not formatted according to RFC 8785, section 3.2.2.2.
	// TODO: Track whether a number is a non-integer?
)

func (f *ValueFlags) Join(f2 ValueFlags) { *f |= f2 }
func (f ValueFlags) IsVerbatim() bool    { return f&stringNonVerbatim == 0 }
func (f ValueFlags) IsCanonical() bool   { return f&stringNonCanonical == 0 }

// ConsumeWhitespace consumes leading JSON whitespace per RFC 7159, section 2.
func ConsumeWhitespace(b []byte) (n int) {
	// NOTE: The arguments and logic are kept simple to keep this inlinable.
	for len(b) > n && (b[n] == ' ' || b[n] == '\t' || b[n] == '\r' || b[n] == '\n') {
		n++
	}
	return n
}

// ConsumeNull consumes the next JSON null literal per RFC 7159, section 3.
// It returns 0 if it is invalid, in which case consumeLiteral should be used.
func ConsumeNull(b []byte) int {
	// NOTE: The arguments and logic are kept simple to keep this inlinable.
	const literal = "null"
	if len(b) >= len(literal) && string(b[:len(literal)]) == literal {
		return len(literal)
	}
	return 0
}

// ConsumeFalse consumes the next JSON false literal per RFC 7159, section 3.
// It returns 0 if it is invalid, in which case consumeLiteral should be used.
func ConsumeFalse(b []byte) int {
	// NOTE: The arguments and logic are kept simple to keep this inlinable.
	const literal = "false"
	if len(b) >= len(literal) && string(b[:len(literal)]) == literal {
		return len(literal)
	}
	return 0
}

// ConsumeTrue consumes the next JSON true literal per RFC 7159, section 3.
// It returns 0 if it is invalid, in which case consumeLiteral should be used.
func ConsumeTrue(b []byte) int {
	// NOTE: The arguments and logic are kept simple to keep this inlinable.
	const literal = "true"
	if len(b) >= len(literal) && string(b[:len(literal)]) == literal {
		return len(literal)
	}
	return 0
}

// ConsumeLiteral consumes the next JSON literal per RFC 7159, section 3.
// If the input appears truncated, it returns io.ErrUnexpectedEOF.
func ConsumeLiteral(b []byte, lit string) (n int, err error) {
	for i := 0; i < len(b) && i < len(lit); i++ {
		if b[i] != lit[i] {
			return i, NewInvalidCharacterError(b[i:], "in literal "+lit+" (expecting "+strconv.QuoteRune(rune(lit[i]))+")")
		}
	}
	if len(b) < len(lit) {
		return len(b), io.ErrUnexpectedEOF
	}
	return len(lit), nil
}

// ConsumeSimpleString consumes the next JSON string per RFC 7159, section 7
// but is limited to the grammar for an ASCII string without escape sequences.
// It returns 0 if it is invalid or more complicated than a simple string,
// in which case consumeString should be called.
//
// It rejects '<', '>', and '&' for compatibility reasons since these were
// always escaped in the v1 implementation. Thus, if this function reports
// non-zero then we know that the string would be encoded the same way
// under both v1 or v2 escape semantics.
func ConsumeSimpleString(b []byte) (n int) {
	// NOTE: The arguments and logic are kept simple to keep this inlinable.
	if len(b) > 0 && b[0] == '"' {
		n++
		for len(b) > n && b[n] < utf8.RuneSelf && escapeASCII[b[n]] == 0 {
			n++
		}
		if uint(len(b)) > uint(n) && b[n] == '"' {
			n++
			return n
		}
	}
	return 0
}

// ConsumeString consumes the next JSON string per RFC 7159, section 7.
// If validateUTF8 is false, then this allows the presence of invalid UTF-8
// characters within the string itself.
// It reports the number of bytes consumed and whether an error was encountered.
// If the input appears truncated, it returns io.ErrUnexpectedEOF.
func ConsumeString(flags *ValueFlags, b []byte, validateUTF8 bool) (n int, err error) {
	return ConsumeStringResumable(flags, b, 0, validateUTF8)
}

// ConsumeStringResumable is identical to consumeString but supports resuming
// from a previous call that returned io.ErrUnexpectedEOF.
func ConsumeStringResumable(flags *ValueFlags, b []byte, resumeOffset int, validateUTF8 bool) (n int, err error) {
	// Consume the leading double quote.
	switch {
	case resumeOffset > 0:
		n = resumeOffset // already handled the leading quote
	case uint(len(b)) == 0:
		return n, io.ErrUnexpectedEOF
	case b[0] == '"':
		n++
	default:
		return n, NewInvalidCharacterError(b[n:], `at start of string (expecting '"')`)
	}

	// Consume every character in the string.
	for uint(len(b)) > uint(n) {
		// Optimize for long sequences of unescaped characters.
		noEscape := func(c byte) bool {
			return c < utf8.RuneSelf && ' ' <= c && c != '\\' && c != '"'
		}
		for uint(len(b)) > uint(n) && noEscape(b[n]) {
			n++
		}
		if uint(len(b)) <= uint(n) {
			return n, io.ErrUnexpectedEOF
		}

		// Check for terminating double quote.
		if b[n] == '"' {
			n++
			return n, nil
		}

		switch r, rn := utf8.DecodeRune(b[n:]); {
		// Handle UTF-8 encoded byte sequence.
		// Due to specialized handling of ASCII above, we know that
		// all normal sequences at this point must be 2 bytes or larger.
		case rn > 1:
			n += rn
		// Handle escape sequence.
		case r == '\\':
			flags.Join(stringNonVerbatim)
			resumeOffset = n
			if uint(len(b)) < uint(n+2) {
				return resumeOffset, io.ErrUnexpectedEOF
			}
			switch r := b[n+1]; r {
			case '/':
				// Forward slash is the only character with 3 representations.
				// Per RFC 8785, section 3.2.2.2., this must not be escaped.
				flags.Join(stringNonCanonical)
				n += 2
			case '"', '\\', 'b', 'f', 'n', 'r', 't':
				n += 2
			case 'u':
				if uint(len(b)) < uint(n+6) {
					if hasEscapedUTF16Prefix(b[n:], false) {
						return resumeOffset, io.ErrUnexpectedEOF
					}
					flags.Join(stringNonCanonical)
					return n, NewInvalidEscapeSequenceError(b[n:])
				}
				v1, ok := parseHexUint16(b[n+2 : n+6])
				if !ok {
					flags.Join(stringNonCanonical)
					return n, NewInvalidEscapeSequenceError(b[n : n+6])
				}
				// Only certain control characters can use the \uFFFF notation
				// for canonical formatting (per RFC 8785, section 3.2.2.2.).
				switch v1 {
				// \uFFFF notation not permitted for these characters.
				case '\b', '\f', '\n', '\r', '\t':
					flags.Join(stringNonCanonical)
				default:
					// \uFFFF notation only permitted for control characters.
					if v1 >= ' ' {
						flags.Join(stringNonCanonical)
					} else {
						// \uFFFF notation must be lower case.
						for _, c := range b[n+2 : n+6] {
							if 'A' <= c && c <= 'F' {
								flags.Join(stringNonCanonical)
							}
						}
					}
				}
				n += 6

				r := rune(v1)
				if validateUTF8 && utf16.IsSurrogate(r) {
					if uint(len(b)) < uint(n+6) {
						if hasEscapedUTF16Prefix(b[n:], true) {
							return resumeOffset, io.ErrUnexpectedEOF
						}
						flags.Join(stringNonCanonical)
						return n - 6, NewInvalidEscapeSequenceError(b[n-6:])
					} else if v2, ok := parseHexUint16(b[n+2 : n+6]); b[n] != '\\' || b[n+1] != 'u' || !ok {
						flags.Join(stringNonCanonical)
						return n - 6, NewInvalidEscapeSequenceError(b[n-6 : n+6])
					} else if r = utf16.DecodeRune(rune(v1), rune(v2)); r == utf8.RuneError {
						flags.Join(stringNonCanonical)
						return n - 6, NewInvalidEscapeSequenceError(b[n-6 : n+6])
					} else {
						n += 6
					}
				}
			default:
				flags.Join(stringNonCanonical)
				return n, NewInvalidEscapeSequenceError(b[n : n+2])
			}
		// Handle invalid UTF-8.
		case r == utf8.RuneError:
			if !utf8.FullRune(b[n:]) {
				return n, io.ErrUnexpectedEOF
			}
			flags.Join(stringNonVerbatim | stringNonCanonical)
			if validateUTF8 {
				return n, ErrInvalidUTF8
			}
			n++
		// Handle invalid control characters.
		case r < ' ':
			flags.Join(stringNonVerbatim | stringNonCanonical)
			return n, NewInvalidCharacterError(b[n:], "in string (expecting non-control character)")
		default:
			panic("BUG: unhandled character " + QuoteRune(b[n:]))
		}
	}
	return n, io.ErrUnexpectedEOF
}

// AppendUnquote appends the unescaped form of a JSON string in src to dst.
// Any invalid UTF-8 within the string will be replaced with utf8.RuneError,
// but the error will be specified as having encountered such an error.
// The input must be an entire JSON string with no surrounding whitespace.
func AppendUnquote[Bytes ~[]byte | ~string](dst []byte, src Bytes) (v []byte, err error) {
	dst = slices.Grow(dst, len(src))

	// Consume the leading double quote.
	var i, n int
	switch {
	case uint(len(src)) == 0:
		return dst, io.ErrUnexpectedEOF
	case src[0] == '"':
		i, n = 1, 1
	default:
		return dst, NewInvalidCharacterError(src, `at start of string (expecting '"')`)
	}

	// Consume every character in the string.
	for uint(len(src)) > uint(n) {
		// Optimize for long sequences of unescaped characters.
		noEscape := func(c byte) bool {
			return c < utf8.RuneSelf && ' ' <= c && c != '\\' && c != '"'
		}
		for uint(len(src)) > uint(n) && noEscape(src[n]) {
			n++
		}
		if uint(len(src)) <= uint(n) {
			dst = append(dst, src[i:n]...)
			return dst, io.ErrUnexpectedEOF
		}

		// Check for terminating double quote.
		if src[n] == '"' {
			dst = append(dst, src[i:n]...)
			n++
			if n < len(src) {
				err = NewInvalidCharacterError(src[n:], "after string value")
			}
			return dst, err
		}

		switch r, rn := utf8.DecodeRuneInString(string(truncateMaxUTF8(src[n:]))); {
		// Handle UTF-8 encoded byte sequence.
		// Due to specialized handling of ASCII above, we know that
		// all normal sequences at this point must be 2 bytes or larger.
		case rn > 1:
			n += rn
		// Handle escape sequence.
		case r == '\\':
			dst = append(dst, src[i:n]...)

			// Handle escape sequence.
			if uint(len(src)) < uint(n+2) {
				return dst, io.ErrUnexpectedEOF
			}
			switch r := src[n+1]; r {
			case '"', '\\', '/':
				dst = append(dst, r)
				n += 2
			case 'b':
				dst = append(dst, '\b')
				n += 2
			case 'f':
				dst = append(dst, '\f')
				n += 2
			case 'n':
				dst = append(dst, '\n')
				n += 2
			case 'r':
				dst = append(dst, '\r')
				n += 2
			case 't':
				dst = append(dst, '\t')
				n += 2
			case 'u':
				if uint(len(src)) < uint(n+6) {
					if hasEscapedUTF16Prefix(src[n:], false) {
						return dst, io.ErrUnexpectedEOF
					}
					return dst, NewInvalidEscapeSequenceError(src[n:])
				}
				v1, ok := parseHexUint16(src[n+2 : n+6])
				if !ok {
					return dst, NewInvalidEscapeSequenceError(src[n : n+6])
				}
				n += 6

				// Check whether this is a surrogate half.
				r := rune(v1)
				if utf16.IsSurrogate(r) {
					r = utf8.RuneError // assume failure unless the following succeeds
					if uint(len(src)) < uint(n+6) {
						if hasEscapedUTF16Prefix(src[n:], true) {
							return utf8.AppendRune(dst, r), io.ErrUnexpectedEOF
						}
						err = NewInvalidEscapeSequenceError(src[n-6:])
					} else if v2, ok := parseHexUint16(src[n+2 : n+6]); src[n] != '\\' || src[n+1] != 'u' || !ok {
						err = NewInvalidEscapeSequenceError(src[n-6 : n+6])
					} else if r = utf16.DecodeRune(rune(v1), rune(v2)); r == utf8.RuneError {
						err = NewInvalidEscapeSequenceError(src[n-6 : n+6])
					} else {
						n += 6
					}
				}

				dst = utf8.AppendRune(dst, r)
			default:
				return dst, NewInvalidEscapeSequenceError(src[n : n+2])
			}
			i = n
		// Handle invalid UTF-8.
		case r == utf8.RuneError:
			dst = append(dst, src[i:n]...)
			if !utf8.FullRuneInString(string(truncateMaxUTF8(src[n:]))) {
				return dst, io.ErrUnexpectedEOF
			}
			// NOTE: An unescaped string may be longer than the escaped string
			// because invalid UTF-8 bytes are being replaced.
			dst = append(dst, "\uFFFD"...)
			n += rn
			i = n
			err = ErrInvalidUTF8
		// Handle invalid control characters.
		case r < ' ':
			dst = append(dst, src[i:n]...)
			return dst, NewInvalidCharacterError(src[n:], "in string (expecting non-control character)")
		default:
			panic("BUG: unhandled character " + QuoteRune(src[n:]))
		}
	}
	dst = append(dst, src[i:n]...)
	return dst, io.ErrUnexpectedEOF
}

// hasEscapedUTF16Prefix reports whether b is possibly
// the truncated prefix of a \uFFFF escape sequence.
func hasEscapedUTF16Prefix[Bytes ~[]byte | ~string](b Bytes, lowerSurrogateHalf bool) bool {
	for i := range len(b) {
		switch c := b[i]; {
		case i == 0 && c != '\\':
			return false
		case i == 1 && c != 'u':
			return false
		case i == 2 && lowerSurrogateHalf && c != 'd' && c != 'D':
			return false // not within ['\uDC00':'\uDFFF']
		case i == 3 && lowerSurrogateHalf && !('c' <= c && c <= 'f') && !('C' <= c && c <= 'F'):
			return false // not within ['\uDC00':'\uDFFF']
		case i >= 2 && i < 6 && !('0' <= c && c <= '9') && !('a' <= c && c <= 'f') && !('A' <= c && c <= 'F'):
			return false
		}
	}
	return true
}

// UnquoteMayCopy returns the unescaped form of b.
// If there are no escaped characters, the output is simply a subslice of
// the input with the surrounding quotes removed.
// Otherwise, a new buffer is allocated for the output.
// It assumes the input is valid.
func UnquoteMayCopy(b []byte, isVerbatim bool) []byte {
	// NOTE: The arguments and logic are kept simple to keep this inlinable.
	if isVerbatim {
		return b[len(`"`) : len(b)-len(`"`)]
	}
	b, _ = AppendUnquote(nil, b)
	return b
}

// ConsumeSimpleNumber consumes the next JSON number per RFC 7159, section 6
// but is limited to the grammar for a positive integer.
// It returns 0 if it is invalid or more complicated than a simple integer,
// in which case consumeNumber should be called.
func ConsumeSimpleNumber(b []byte) (n int) {
	// NOTE: The arguments and logic are kept simple to keep this inlinable.
	if len(b) > 0 {
		if b[0] == '0' {
			n++
		} else if '1' <= b[0] && b[0] <= '9' {
			n++
			for len(b) > n && ('0' <= b[n] && b[n] <= '9') {
				n++
			}
		} else {
			return 0
		}
		if uint(len(b)) <= uint(n) || (b[n] != '.' && b[n] != 'e' && b[n] != 'E') {
			return n
		}
	}
	return 0
}

type ConsumeNumberState uint

const (
	consumeNumberInit ConsumeNumberState = iota
	beforeIntegerDigits
	withinIntegerDigits
	beforeFractionalDigits
	withinFractionalDigits
	beforeExponentDigits
	withinExponentDigits
)

// ConsumeNumber consumes the next JSON number per RFC 7159, section 6.
// It reports the number of bytes consumed and whether an error was encountered.
// If the input appears truncated, it returns io.ErrUnexpectedEOF.
//
// Note that JSON numbers are not self-terminating.
// If the entire input is consumed, then the caller needs to consider whether
// there may be subsequent unread data that may still be part of this number.
func ConsumeNumber(b []byte) (n int, err error) {
	n, _, err = ConsumeNumberResumable(b, 0, consumeNumberInit)
	return n, err
}

// ConsumeNumberResumable is identical to consumeNumber but supports resuming
// from a previous call that returned io.ErrUnexpectedEOF.
func ConsumeNumberResumable(b []byte, resumeOffset int, state ConsumeNumberState) (n int, _ ConsumeNumberState, err error) {
	// Jump to the right state when resuming from a partial consumption.
	n = resumeOffset
	if state > consumeNumberInit {
		switch state {
		case withinIntegerDigits, withinFractionalDigits, withinExponentDigits:
			// Consume leading digits.
			for uint(len(b)) > uint(n) && ('0' <= b[n] && b[n] <= '9') {
				n++
			}
			if uint(len(b)) <= uint(n) {
				return n, state, nil // still within the same state
			}
			state++ // switches "withinX" to "beforeY" where Y is the state after X
		}
		switch state {
		case beforeIntegerDigits:
			goto beforeInteger
		case beforeFractionalDigits:
			goto beforeFractional
		case beforeExponentDigits:
			goto beforeExponent
		default:
			return n, state, nil
		}
	}

	// Consume required integer component (with optional minus sign).
beforeInteger:
	resumeOffset = n
	if uint(len(b)) > 0 && b[0] == '-' {
		n++
	}
	switch {
	case uint(len(b)) <= uint(n):
		return resumeOffset, beforeIntegerDigits, io.ErrUnexpectedEOF
	case b[n] == '0':
		n++
		state = beforeFractionalDigits
	case '1' <= b[n] && b[n] <= '9':
		n++
		for uint(len(b)) > uint(n) && ('0' <= b[n] && b[n] <= '9') {
			n++
		}
		state = withinIntegerDigits
	default:
		return n, state, NewInvalidCharacterError(b[n:], "in number (expecting digit)")
	}

	// Consume optional fractional component.
beforeFractional:
	if uint(len(b)) > uint(n) && b[n] == '.' {
		resumeOffset = n
		n++
		switch {
		case uint(len(b)) <= uint(n):
			return resumeOffset, beforeFractionalDigits, io.ErrUnexpectedEOF
		case '0' <= b[n] && b[n] <= '9':
			n++
		default:
			return n, state, NewInvalidCharacterError(b[n:], "in number (expecting digit)")
		}
		for uint(len(b)) > uint(n) && ('0' <= b[n] && b[n] <= '9') {
			n++
		}
		state = withinFractionalDigits
	}

	// Consume optional exponent component.
beforeExponent:
	if uint(len(b)) > uint(n) && (b[n] == 'e' || b[n] == 'E') {
		resumeOffset = n
		n++
		if uint(len(b)) > uint(n) && (b[n] == '-' || b[n] == '+') {
			n++
		}
		switch {
		case uint(len(b)) <= uint(n):
			return resumeOffset, beforeExponentDigits, io.ErrUnexpectedEOF
		case '0' <= b[n] && b[n] <= '9':
			n++
		default:
			return n, state, NewInvalidCharacterError(b[n:], "in number (expecting digit)")
		}
		for uint(len(b)) > uint(n) && ('0' <= b[n] && b[n] <= '9') {
			n++
		}
		state = withinExponentDigits
	}

	return n, state, nil
}

// parseHexUint16 is similar to strconv.ParseUint,
// but operates directly on []byte and is optimized for base-16.
// See https://go.dev/issue/42429.
func parseHexUint16[Bytes ~[]byte | ~string](b Bytes) (v uint16, ok bool) {
	if len(b) != 4 {
		return 0, false
	}
	for i := range 4 {
		c := b[i]
		switch {
		case '0' <= c && c <= '9':
			c = c - '0'
		case 'a' <= c && c <= 'f':
			c = 10 + c - 'a'
		case 'A' <= c && c <= 'F':
			c = 10 + c - 'A'
		default:
			return 0, false
		}
		v = v*16 + uint16(c)
	}
	return v, true
}

// ParseUint parses b as a decimal unsigned integer according to
// a strict subset of the JSON number grammar, returning the value if valid.
// It returns (0, false) if there is a syntax error and
// returns (math.MaxUint64, false) if there is an overflow.
func ParseUint(b []byte) (v uint64, ok bool) {
	const unsafeWidth = 20 // len(fmt.Sprint(uint64(math.MaxUint64)))
	var n int
	for ; len(b) > n && ('0' <= b[n] && b[n] <= '9'); n++ {
		v = 10*v + uint64(b[n]-'0')
	}
	switch {
	case n == 0 || len(b) != n || (b[0] == '0' && string(b) != "0"):
		return 0, false
	case n >= unsafeWidth && (b[0] != '1' || v < 1e19 || n > unsafeWidth):
		return math.MaxUint64, false
	}
	return v, true
}

// ParseFloat parses a floating point number according to the Go float grammar.
// Note that the JSON number grammar is a strict subset.
//
// If the number overflows the finite representation of a float,
// then we return MaxFloat since any finite value will always be infinitely
// more accurate at representing another finite value than an infinite value.
func ParseFloat(b []byte, bits int) (v float64, ok bool) {
	fv, err := strconv.ParseFloat(string(b), bits)
	if math.IsInf(fv, 0) {
		switch {
		case bits == 32 && math.IsInf(fv, +1):
			fv = +math.MaxFloat32
		case bits == 64 && math.IsInf(fv, +1):
			fv = +math.MaxFloat64
		case bits == 32 && math.IsInf(fv, -1):
			fv = -math.MaxFloat32
		case bits == 64 && math.IsInf(fv, -1):
			fv = -math.MaxFloat64
		}
	}
	return fv, err == nil
}
