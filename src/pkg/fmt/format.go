// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fmt

import (
	"bytes"
	"strconv"
	"unicode"
	"unicode/utf8"
)

const (
	nByte = 64

	ldigits = "0123456789abcdef"
	udigits = "0123456789ABCDEF"
)

const (
	signed   = true
	unsigned = false
)

var padZeroBytes = make([]byte, nByte)
var padSpaceBytes = make([]byte, nByte)

var newline = []byte{'\n'}

func init() {
	for i := 0; i < nByte; i++ {
		padZeroBytes[i] = '0'
		padSpaceBytes[i] = ' '
	}
}

// A fmt is the raw formatter used by Printf etc.
// It prints into a bytes.Buffer that must be set up externally.
type fmt struct {
	intbuf [nByte]byte
	buf    *bytes.Buffer
	// width, precision
	wid  int
	prec int
	// flags
	widPresent  bool
	precPresent bool
	minus       bool
	plus        bool
	sharp       bool
	space       bool
	unicode     bool
	uniQuote    bool // Use 'x'= prefix for %U if printable.
	zero        bool
}

func (f *fmt) clearflags() {
	f.wid = 0
	f.widPresent = false
	f.prec = 0
	f.precPresent = false
	f.minus = false
	f.plus = false
	f.sharp = false
	f.space = false
	f.unicode = false
	f.uniQuote = false
	f.zero = false
}

func (f *fmt) init(buf *bytes.Buffer) {
	f.buf = buf
	f.clearflags()
}

// Compute left and right padding widths (only one will be non-zero).
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

// Generate n bytes of padding.
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

// Append b to f.buf, padded on left (w > 0) or right (w < 0 or f.minus)
// clear flags afterwards.
func (f *fmt) pad(b []byte) {
	var padding []byte
	var left, right int
	if f.widPresent && f.wid != 0 {
		padding, left, right = f.computePadding(len(b))
	}
	if left > 0 {
		f.writePadding(left, padding)
	}
	f.buf.Write(b)
	if right > 0 {
		f.writePadding(right, padding)
	}
}

// append s to buf, padded on left (w > 0) or right (w < 0 or f.minus).
// clear flags afterwards.
func (f *fmt) padString(s string) {
	var padding []byte
	var left, right int
	if f.widPresent && f.wid != 0 {
		padding, left, right = f.computePadding(utf8.RuneCountInString(s))
	}
	if left > 0 {
		f.writePadding(left, padding)
	}
	f.buf.WriteString(s)
	if right > 0 {
		f.writePadding(right, padding)
	}
}

func putint(buf []byte, base, val uint64, digits string) int {
	i := len(buf) - 1
	for val >= base {
		buf[i] = digits[val%base]
		i--
		val /= base
	}
	buf[i] = digits[val]
	return i - 1
}

// fmt_boolean formats a boolean.
func (f *fmt) fmt_boolean(v bool) {
	if v {
		f.padString("true")
	} else {
		f.padString("false")
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
	i := len(f.intbuf)
	ua := uint64(a)
	for ua >= base {
		i--
		buf[i] = digits[ua%base]
		ua /= base
	}
	i--
	buf[i] = digits[ua]
	for i > 0 && prec > nByte-i {
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
	if f.unicode && f.uniQuote && a >= 0 && a <= unicode.MaxRune && unicode.IsPrint(rune(a)) {
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

// fmt_sx formats a string as a hexadecimal encoding of its bytes.
func (f *fmt) fmt_sx(s string) {
	t := ""
	for i := 0; i < len(s); i++ {
		if i > 0 && f.space {
			t += " "
		}
		v := s[i]
		t += string(ldigits[v>>4])
		t += string(ldigits[v&0xF])
	}
	f.padString(t)
}

// fmt_sX formats a string as an uppercase hexadecimal encoding of its bytes.
func (f *fmt) fmt_sX(s string) {
	t := ""
	for i := 0; i < len(s); i++ {
		if i > 0 && f.space {
			t += " "
		}
		v := s[i]
		t += string(udigits[v>>4])
		t += string(udigits[v&0xF])
	}
	f.padString(t)
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
	var quoted string
	if f.plus {
		quoted = strconv.QuoteRuneToASCII(rune(c))
	} else {
		quoted = strconv.QuoteRune(rune(c))
	}
	f.padString(quoted)
}

// floating-point

func doPrec(f *fmt, def int) int {
	if f.precPresent {
		return f.prec
	}
	return def
}

// Add a plus sign or space to the floating-point string representation if missing and required.
func (f *fmt) plusSpace(s string) {
	if s[0] != '-' {
		if f.plus {
			s = "+" + s
		} else if f.space {
			s = " " + s
		}
	}
	f.padString(s)
}

// fmt_e64 formats a float64 in the form -1.23e+12.
func (f *fmt) fmt_e64(v float64) { f.plusSpace(strconv.FormatFloat(v, 'e', doPrec(f, 6), 64)) }

// fmt_E64 formats a float64 in the form -1.23E+12.
func (f *fmt) fmt_E64(v float64) { f.plusSpace(strconv.FormatFloat(v, 'E', doPrec(f, 6), 64)) }

// fmt_f64 formats a float64 in the form -1.23.
func (f *fmt) fmt_f64(v float64) { f.plusSpace(strconv.FormatFloat(v, 'f', doPrec(f, 6), 64)) }

// fmt_g64 formats a float64 in the 'f' or 'e' form according to size.
func (f *fmt) fmt_g64(v float64) { f.plusSpace(strconv.FormatFloat(v, 'g', doPrec(f, -1), 64)) }

// fmt_g64 formats a float64 in the 'f' or 'E' form according to size.
func (f *fmt) fmt_G64(v float64) { f.plusSpace(strconv.FormatFloat(v, 'G', doPrec(f, -1), 64)) }

// fmt_fb64 formats a float64 in the form -123p3 (exponent is power of 2).
func (f *fmt) fmt_fb64(v float64) { f.plusSpace(strconv.FormatFloat(v, 'b', 0, 64)) }

// float32
// cannot defer to float64 versions
// because it will get rounding wrong in corner cases.

// fmt_e32 formats a float32 in the form -1.23e+12.
func (f *fmt) fmt_e32(v float32) { f.plusSpace(strconv.FormatFloat(float64(v), 'e', doPrec(f, 6), 32)) }

// fmt_E32 formats a float32 in the form -1.23E+12.
func (f *fmt) fmt_E32(v float32) { f.plusSpace(strconv.FormatFloat(float64(v), 'E', doPrec(f, 6), 32)) }

// fmt_f32 formats a float32 in the form -1.23.
func (f *fmt) fmt_f32(v float32) { f.plusSpace(strconv.FormatFloat(float64(v), 'f', doPrec(f, 6), 32)) }

// fmt_g32 formats a float32 in the 'f' or 'e' form according to size.
func (f *fmt) fmt_g32(v float32) { f.plusSpace(strconv.FormatFloat(float64(v), 'g', doPrec(f, -1), 32)) }

// fmt_G32 formats a float32 in the 'f' or 'E' form according to size.
func (f *fmt) fmt_G32(v float32) { f.plusSpace(strconv.FormatFloat(float64(v), 'G', doPrec(f, -1), 32)) }

// fmt_fb32 formats a float32 in the form -123p3 (exponent is power of 2).
func (f *fmt) fmt_fb32(v float32) { f.padString(strconv.FormatFloat(float64(v), 'b', 0, 32)) }

// fmt_c64 formats a complex64 according to the verb.
func (f *fmt) fmt_c64(v complex64, verb rune) {
	f.buf.WriteByte('(')
	r := real(v)
	for i := 0; ; i++ {
		switch verb {
		case 'e':
			f.fmt_e32(r)
		case 'E':
			f.fmt_E32(r)
		case 'f':
			f.fmt_f32(r)
		case 'g':
			f.fmt_g32(r)
		case 'G':
			f.fmt_G32(r)
		}
		if i != 0 {
			break
		}
		f.plus = true
		r = imag(v)
	}
	f.buf.Write(irparenBytes)
}

// fmt_c128 formats a complex128 according to the verb.
func (f *fmt) fmt_c128(v complex128, verb rune) {
	f.buf.WriteByte('(')
	r := real(v)
	for i := 0; ; i++ {
		switch verb {
		case 'e':
			f.fmt_e64(r)
		case 'E':
			f.fmt_E64(r)
		case 'f':
			f.fmt_f64(r)
		case 'g':
			f.fmt_g64(r)
		case 'G':
			f.fmt_G64(r)
		}
		if i != 0 {
			break
		}
		f.plus = true
		r = imag(v)
	}
	f.buf.Write(irparenBytes)
}
