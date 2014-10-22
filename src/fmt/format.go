// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fmt

import (
	"math"
	"strconv"
	"unicode/utf8"
)

const (
	// %b of an int64, plus a sign.
	// Hex can add 0x and we handle it specially.
	nByte = 65

	ldigits = "0123456789abcdef"
	udigits = "0123456789ABCDEF"
)

const (
	signed   = true
	unsigned = false
)

var padZeroBytes = make([]byte, nByte)
var padSpaceBytes = make([]byte, nByte)

func init() {
	for i := 0; i < nByte; i++ {
		padZeroBytes[i] = '0'
		padSpaceBytes[i] = ' '
	}
}

// flags placed in a separate struct for easy clearing.
type fmtFlags struct {
	widPresent  bool
	precPresent bool
	minus       bool
	plus        bool
	sharp       bool
	space       bool
	unicode     bool
	uniQuote    bool // Use 'x'= prefix for %U if printable.
	zero        bool

	// For the formats %+v %#v, we set the plusV/sharpV flags
	// and clear the plus/sharp flags since %+v and %#v are in effect
	// different, flagless formats set at the top level.
	plusV  bool
	sharpV bool
}

// A fmt is the raw formatter used by Printf etc.
// It prints into a buffer that must be set up separately.
type fmt struct {
	intbuf [nByte]byte
	buf    *buffer
	// width, precision
	wid  int
	prec int
	fmtFlags
}

func (f *fmt) clearflags() {
	f.fmtFlags = fmtFlags{}
}

func (f *fmt) init(buf *buffer) {
	f.buf = buf
	f.clearflags()
}

// computePadding computes left and right padding widths (only one will be non-zero).
func (f *fmt) computePadding(width int) (padding []byte, leftWidth, rightWidth int) {
	left := !f.minus
	w := f.wid
	if w < 0 {
		left = false
		w = -w
	}
	w -= width
	if w > 0 {
		if left && f.zero {
			return padZeroBytes, w, 0
		}
		if left {
			return padSpaceBytes, w, 0
		} else {
			// can't be zero padding on the right
			return padSpaceBytes, 0, w
		}
	}
	return
}

// writePadding generates n bytes of padding.
func (f *fmt) writePadding(n int, padding []byte) {
	for n > 0 {
		m := n
		if m > nByte {
			m = nByte
		}
		f.buf.Write(padding[0:m])
		n -= m
	}
}

// pad appends b to f.buf, padded on left (w > 0) or right (w < 0 or f.minus).
func (f *fmt) pad(b []byte) {
	if !f.widPresent || f.wid == 0 {
		f.buf.Write(b)
		return
	}
	padding, left, right := f.computePadding(utf8.RuneCount(b))
	if left > 0 {
		f.writePadding(left, padding)
	}
	f.buf.Write(b)
	if right > 0 {
		f.writePadding(right, padding)
	}
}

// padString appends s to buf, padded on left (w > 0) or right (w < 0 or f.minus).
func (f *fmt) padString(s string) {
	if !f.widPresent || f.wid == 0 {
		f.buf.WriteString(s)
		return
	}
	padding, left, right := f.computePadding(utf8.RuneCountInString(s))
	if left > 0 {
		f.writePadding(left, padding)
	}
	f.buf.WriteString(s)
	if right > 0 {
		f.writePadding(right, padding)
	}
}

var (
	trueBytes  = []byte("true")
	falseBytes = []byte("false")
)

// fmt_boolean formats a boolean.
func (f *fmt) fmt_boolean(v bool) {
	if v {
		f.pad(trueBytes)
	} else {
		f.pad(falseBytes)
	}
}

// integer; interprets prec but not wid.  Once formatted, result is sent to pad()
// and then flags are cleared.
func (f *fmt) integer(a int64, base uint64, signedness bool, digits string) {
	// precision of 0 and value of 0 means "print nothing"
	if f.precPresent && f.prec == 0 && a == 0 {
		return
	}

	var buf []byte = f.intbuf[0:]
	if f.widPresent {
		width := f.wid
		if base == 16 && f.sharp {
			// Also adds "0x".
			width += 2
		}
		if width > nByte {
			// We're going to need a bigger boat.
			buf = make([]byte, width)
		}
	}

	negative := signedness == signed && a < 0
	if negative {
		a = -a
	}

	// two ways to ask for extra leading zero digits: %.3d or %03d.
	// apparently the first cancels the second.
	prec := 0
	if f.precPresent {
		prec = f.prec
		f.zero = false
	} else if f.zero && f.widPresent && !f.minus && f.wid > 0 {
		prec = f.wid
		if negative || f.plus || f.space {
			prec-- // leave room for sign
		}
	}

	// format a into buf, ending at buf[i].  (printing is easier right-to-left.)
	// a is made into unsigned ua.  we could make things
	// marginally faster by splitting the 32-bit case out into a separate
	// block but it's not worth the duplication, so ua has 64 bits.
	i := len(buf)
	ua := uint64(a)
	// use constants for the division and modulo for more efficient code.
	// switch cases ordered by popularity.
	switch base {
	case 10:
		for ua >= 10 {
			i--
			next := ua / 10
			buf[i] = byte('0' + ua - next*10)
			ua = next
		}
	case 16:
		for ua >= 16 {
			i--
			buf[i] = digits[ua&0xF]
			ua >>= 4
		}
	case 8:
		for ua >= 8 {
			i--
			buf[i] = byte('0' + ua&7)
			ua >>= 3
		}
	case 2:
		for ua >= 2 {
			i--
			buf[i] = byte('0' + ua&1)
			ua >>= 1
		}
	default:
		panic("fmt: unknown base; can't happen")
	}
	i--
	buf[i] = digits[ua]
	for i > 0 && prec > len(buf)-i {
		i--
		buf[i] = '0'
	}

	// Various prefixes: 0x, -, etc.
	if f.sharp {
		switch base {
		case 8:
			if buf[i] != '0' {
				i--
				buf[i] = '0'
			}
		case 16:
			i--
			buf[i] = 'x' + digits[10] - 'a'
			i--
			buf[i] = '0'
		}
	}
	if f.unicode {
		i--
		buf[i] = '+'
		i--
		buf[i] = 'U'
	}

	if negative {
		i--
		buf[i] = '-'
	} else if f.plus {
		i--
		buf[i] = '+'
	} else if f.space {
		i--
		buf[i] = ' '
	}

	// If we want a quoted char for %#U, move the data up to make room.
	if f.unicode && f.uniQuote && a >= 0 && a <= utf8.MaxRune && strconv.IsPrint(rune(a)) {
		runeWidth := utf8.RuneLen(rune(a))
		width := 1 + 1 + runeWidth + 1 // space, quote, rune, quote
		copy(buf[i-width:], buf[i:])   // guaranteed to have enough room.
		i -= width
		// Now put " 'x'" at the end.
		j := len(buf) - width
		buf[j] = ' '
		j++
		buf[j] = '\''
		j++
		utf8.EncodeRune(buf[j:], rune(a))
		j += runeWidth
		buf[j] = '\''
	}

	f.pad(buf[i:])
}

// truncate truncates the string to the specified precision, if present.
func (f *fmt) truncate(s string) string {
	if f.precPresent && f.prec < utf8.RuneCountInString(s) {
		n := f.prec
		for i := range s {
			if n == 0 {
				s = s[:i]
				break
			}
			n--
		}
	}
	return s
}

// fmt_s formats a string.
func (f *fmt) fmt_s(s string) {
	s = f.truncate(s)
	f.padString(s)
}

// fmt_sbx formats a string or byte slice as a hexadecimal encoding of its bytes.
func (f *fmt) fmt_sbx(s string, b []byte, digits string) {
	n := len(b)
	if b == nil {
		n = len(s)
	}
	x := digits[10] - 'a' + 'x'
	// TODO: Avoid buffer by pre-padding.
	var buf []byte
	for i := 0; i < n; i++ {
		if i > 0 && f.space {
			buf = append(buf, ' ')
		}
		if f.sharp && (f.space || i == 0) {
			buf = append(buf, '0', x)
		}
		var c byte
		if b == nil {
			c = s[i]
		} else {
			c = b[i]
		}
		buf = append(buf, digits[c>>4], digits[c&0xF])
	}
	f.pad(buf)
}

// fmt_sx formats a string as a hexadecimal encoding of its bytes.
func (f *fmt) fmt_sx(s, digits string) {
	if f.precPresent && f.prec < len(s) {
		s = s[:f.prec]
	}
	f.fmt_sbx(s, nil, digits)
}

// fmt_bx formats a byte slice as a hexadecimal encoding of its bytes.
func (f *fmt) fmt_bx(b []byte, digits string) {
	if f.precPresent && f.prec < len(b) {
		b = b[:f.prec]
	}
	f.fmt_sbx("", b, digits)
}

// fmt_q formats a string as a double-quoted, escaped Go string constant.
func (f *fmt) fmt_q(s string) {
	s = f.truncate(s)
	var quoted string
	if f.sharp && strconv.CanBackquote(s) {
		quoted = "`" + s + "`"
	} else {
		if f.plus {
			quoted = strconv.QuoteToASCII(s)
		} else {
			quoted = strconv.Quote(s)
		}
	}
	f.padString(quoted)
}

// fmt_qc formats the integer as a single-quoted, escaped Go character constant.
// If the character is not valid Unicode, it will print '\ufffd'.
func (f *fmt) fmt_qc(c int64) {
	var quoted []byte
	if f.plus {
		quoted = strconv.AppendQuoteRuneToASCII(f.intbuf[0:0], rune(c))
	} else {
		quoted = strconv.AppendQuoteRune(f.intbuf[0:0], rune(c))
	}
	f.pad(quoted)
}

// floating-point

func doPrec(f *fmt, def int) int {
	if f.precPresent {
		return f.prec
	}
	return def
}

// formatFloat formats a float64; it is an efficient equivalent to  f.pad(strconv.FormatFloat()...).
func (f *fmt) formatFloat(v float64, verb byte, prec, n int) {
	// Format number, reserving space for leading + sign if needed.
	num := strconv.AppendFloat(f.intbuf[0:1], v, verb, prec, n)
	if num[1] == '-' || num[1] == '+' {
		num = num[1:]
	} else {
		num[0] = '+'
	}
	// Special handling for infinity, which doesn't look like a number so shouldn't be padded with zeros.
	if math.IsInf(v, 0) {
		if f.zero {
			defer func() { f.zero = true }()
			f.zero = false
		}
	}
	// num is now a signed version of the number.
	// If we're zero padding, want the sign before the leading zeros.
	// Achieve this by writing the sign out and then padding the unsigned number.
	if f.zero && f.widPresent && f.wid > len(num) {
		if f.space && v >= 0 {
			f.buf.WriteByte(' ') // This is what C does: even with zero, f.space means space.
			f.wid--
		} else if f.plus || v < 0 {
			f.buf.WriteByte(num[0])
			f.wid--
		}
		f.pad(num[1:])
		return
	}
	// f.space says to replace a leading + with a space.
	if f.space && num[0] == '+' {
		num[0] = ' '
		f.pad(num)
		return
	}
	// Now we know the sign is attached directly to the number, if present at all.
	// We want a sign if asked for, if it's negative, or if it's infinity (+Inf vs. -Inf).
	if f.plus || num[0] == '-' || math.IsInf(v, 0) {
		f.pad(num)
		return
	}
	// No sign to show and the number is positive; just print the unsigned number.
	f.pad(num[1:])
}

// fmt_e64 formats a float64 in the form -1.23e+12.
func (f *fmt) fmt_e64(v float64) { f.formatFloat(v, 'e', doPrec(f, 6), 64) }

// fmt_E64 formats a float64 in the form -1.23E+12.
func (f *fmt) fmt_E64(v float64) { f.formatFloat(v, 'E', doPrec(f, 6), 64) }

// fmt_f64 formats a float64 in the form -1.23.
func (f *fmt) fmt_f64(v float64) { f.formatFloat(v, 'f', doPrec(f, 6), 64) }

// fmt_g64 formats a float64 in the 'f' or 'e' form according to size.
func (f *fmt) fmt_g64(v float64) { f.formatFloat(v, 'g', doPrec(f, -1), 64) }

// fmt_G64 formats a float64 in the 'f' or 'E' form according to size.
func (f *fmt) fmt_G64(v float64) { f.formatFloat(v, 'G', doPrec(f, -1), 64) }

// fmt_fb64 formats a float64 in the form -123p3 (exponent is power of 2).
func (f *fmt) fmt_fb64(v float64) { f.formatFloat(v, 'b', 0, 64) }

// float32
// cannot defer to float64 versions
// because it will get rounding wrong in corner cases.

// fmt_e32 formats a float32 in the form -1.23e+12.
func (f *fmt) fmt_e32(v float32) { f.formatFloat(float64(v), 'e', doPrec(f, 6), 32) }

// fmt_E32 formats a float32 in the form -1.23E+12.
func (f *fmt) fmt_E32(v float32) { f.formatFloat(float64(v), 'E', doPrec(f, 6), 32) }

// fmt_f32 formats a float32 in the form -1.23.
func (f *fmt) fmt_f32(v float32) { f.formatFloat(float64(v), 'f', doPrec(f, 6), 32) }

// fmt_g32 formats a float32 in the 'f' or 'e' form according to size.
func (f *fmt) fmt_g32(v float32) { f.formatFloat(float64(v), 'g', doPrec(f, -1), 32) }

// fmt_G32 formats a float32 in the 'f' or 'E' form according to size.
func (f *fmt) fmt_G32(v float32) { f.formatFloat(float64(v), 'G', doPrec(f, -1), 32) }

// fmt_fb32 formats a float32 in the form -123p3 (exponent is power of 2).
func (f *fmt) fmt_fb32(v float32) { f.formatFloat(float64(v), 'b', 0, 32) }

// fmt_c64 formats a complex64 according to the verb.
func (f *fmt) fmt_c64(v complex64, verb rune) {
	f.fmt_complex(float64(real(v)), float64(imag(v)), 32, verb)
}

// fmt_c128 formats a complex128 according to the verb.
func (f *fmt) fmt_c128(v complex128, verb rune) {
	f.fmt_complex(real(v), imag(v), 64, verb)
}

// fmt_complex formats a complex number as (r+ji).
func (f *fmt) fmt_complex(r, j float64, size int, verb rune) {
	f.buf.WriteByte('(')
	oldPlus := f.plus
	oldSpace := f.space
	oldWid := f.wid
	for i := 0; ; i++ {
		switch verb {
		case 'b':
			f.formatFloat(r, 'b', 0, size)
		case 'e':
			f.formatFloat(r, 'e', doPrec(f, 6), size)
		case 'E':
			f.formatFloat(r, 'E', doPrec(f, 6), size)
		case 'f', 'F':
			f.formatFloat(r, 'f', doPrec(f, 6), size)
		case 'g':
			f.formatFloat(r, 'g', doPrec(f, -1), size)
		case 'G':
			f.formatFloat(r, 'G', doPrec(f, -1), size)
		}
		if i != 0 {
			break
		}
		// Imaginary part always has a sign.
		f.plus = true
		f.space = false
		f.wid = oldWid
		r = j
	}
	f.space = oldSpace
	f.plus = oldPlus
	f.wid = oldWid
	f.buf.Write(irparenBytes)
}
