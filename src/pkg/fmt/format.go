// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fmt

import (
	"bytes";
	"strconv";
)

const (
	nByte	= 64;

	ldigits	= "0123456789abcdef";
	udigits	= "0123456789ABCDEF";
)

var padZeroBytes = make([]byte, nByte)
var padSpaceBytes = make([]byte, nByte)

var newline = []byte{'\n'}

func init() {
	for i := 0; i < nByte; i++ {
		padZeroBytes[i] = '0';
		padSpaceBytes[i] = ' ';
	}
}

// A fmt is the raw formatter used by Printf etc.
// It prints into a bytes.Buffer that must be set up externally.
type fmt struct {
	intbuf		[nByte]byte;
	buf		*bytes.Buffer;
	wid		int;
	widPresent	bool;
	prec		int;
	precPresent	bool;
	// flags
	minus	bool;
	plus	bool;
	sharp	bool;
	space	bool;
	zero	bool;
}

func (f *fmt) clearflags() {
	f.wid = 0;
	f.widPresent = false;
	f.prec = 0;
	f.precPresent = false;
	f.minus = false;
	f.plus = false;
	f.sharp = false;
	f.space = false;
	f.zero = false;
}

func (f *fmt) init(buf *bytes.Buffer) {
	f.buf = buf;
	f.clearflags();
}

// Compute left and right padding widths (only one will be non-zero).
func (f *fmt) computePadding(width int) (padding []byte, leftWidth, rightWidth int) {
	left := !f.minus;
	w := f.wid;
	if w < 0 {
		left = false;
		w = -w;
	}
	w -= width;
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
	return;
}

// Generate n bytes of padding.
func (f *fmt) writePadding(n int, padding []byte) {
	for n > 0 {
		m := n;
		if m > nByte {
			m = nByte
		}
		f.buf.Write(padding[0:m]);
		n -= m;
	}
}

// Append b to f.buf, padded on left (w > 0) or right (w < 0 or f.minus)
func (f *fmt) padBytes(b []byte) {
	var padding []byte;
	var left, right int;
	if f.widPresent && f.wid != 0 {
		padding, left, right = f.computePadding(len(b))
	}
	if left > 0 {
		f.writePadding(left, padding)
	}
	f.buf.Write(b);
	if right > 0 {
		f.writePadding(right, padding)
	}
}

// append s to buf, padded on left (w > 0) or right (w < 0 or f.minus)
func (f *fmt) pad(s string) {
	var padding []byte;
	var left, right int;
	if f.widPresent && f.wid != 0 {
		padding, left, right = f.computePadding(len(s))
	}
	if left > 0 {
		f.writePadding(left, padding)
	}
	f.buf.WriteString(s);
	if right > 0 {
		f.writePadding(right, padding)
	}
}

// format val into buf, ending at buf[i].  (printing is easier right-to-left;
// that's why the bidi languages are right-to-left except for numbers. wait,
// never mind.)  val is known to be unsigned.  we could make things maybe
// marginally faster by splitting the 32-bit case out into a separate function
// but it's not worth the duplication, so val has 64 bits.
func putint(buf []byte, base, val uint64, digits string) int {
	i := len(buf) - 1;
	for val >= base {
		buf[i] = digits[val%base];
		i--;
		val /= base;
	}
	buf[i] = digits[val];
	return i - 1;
}

// fmt_boolean formats a boolean.
func (f *fmt) fmt_boolean(v bool) {
	if v {
		f.pad("true")
	} else {
		f.pad("false")
	}
	f.clearflags();
}

// integer; interprets prec but not wid.
func (f *fmt) integer(a int64, base uint, is_signed bool, digits string) []byte {
	var buf []byte = &f.intbuf;
	negative := is_signed && a < 0;
	if negative {
		a = -a
	}

	// two ways to ask for extra leading zero digits: %.3d or %03d.
	// apparently the first cancels the second.
	prec := 0;
	if f.precPresent {
		prec = f.prec;
		f.zero = false;
	} else if f.zero && f.widPresent && !f.minus && f.wid > 0 {
		prec = f.wid;
		if negative || f.plus || f.space {
			prec--	// leave room for sign
		}
	}

	i := putint(buf, uint64(base), uint64(a), digits);
	for i > 0 && prec > (nByte-1-i) {
		buf[i] = '0';
		i--;
	}

	if f.sharp {
		switch base {
		case 8:
			if buf[i+1] != '0' {
				buf[i] = '0';
				i--;
			}
		case 16:
			buf[i] = 'x' + digits[10] - 'a';
			i--;
			buf[i] = '0';
			i--;
		}
	}

	if negative {
		buf[i] = '-';
		i--;
	} else if f.plus {
		buf[i] = '+';
		i--;
	} else if f.space {
		buf[i] = ' ';
		i--;
	}
	return buf[i+1 : nByte];
}

// fmt_d64 formats an int64 in decimal.
func (f *fmt) fmt_d64(v int64) {
	f.padBytes(f.integer(v, 10, true, ldigits));
	f.clearflags();
}

// fmt_d32 formats an int32 in decimal.
func (f *fmt) fmt_d32(v int32)	{ f.fmt_d64(int64(v)) }

// fmt_d formats an int in decimal.
func (f *fmt) fmt_d(v int)	{ f.fmt_d64(int64(v)) }

// fmt_ud64 formats a uint64 in decimal.
func (f *fmt) fmt_ud64(v uint64) *fmt {
	f.padBytes(f.integer(int64(v), 10, false, ldigits));
	f.clearflags();
	return f;
}

// fmt_ud32 formats a uint32 in decimal.
func (f *fmt) fmt_ud32(v uint32)	{ f.fmt_ud64(uint64(v)) }

// fmt_ud formats a uint in decimal.
func (f *fmt) fmt_ud(v uint)	{ f.fmt_ud64(uint64(v)) }

// fmt_x64 formats an int64 in hexadecimal.
func (f *fmt) fmt_x64(v int64) {
	f.padBytes(f.integer(v, 16, true, ldigits));
	f.clearflags();
}

// fmt_x32 formats an int32 in hexadecimal.
func (f *fmt) fmt_x32(v int32)	{ f.fmt_x64(int64(v)) }

// fmt_x formats an int in hexadecimal.
func (f *fmt) fmt_x(v int)	{ f.fmt_x64(int64(v)) }

// fmt_ux64 formats a uint64 in hexadecimal.
func (f *fmt) fmt_ux64(v uint64) {
	f.padBytes(f.integer(int64(v), 16, false, ldigits));
	f.clearflags();
}

// fmt_ux32 formats a uint32 in hexadecimal.
func (f *fmt) fmt_ux32(v uint32)	{ f.fmt_ux64(uint64(v)) }

// fmt_ux formats a uint in hexadecimal.
func (f *fmt) fmt_ux(v uint)	{ f.fmt_ux64(uint64(v)) }

// fmt_X64 formats an int64 in upper case hexadecimal.
func (f *fmt) fmt_X64(v int64) {
	f.padBytes(f.integer(v, 16, true, udigits));
	f.clearflags();
}

// fmt_X32 formats an int32 in upper case hexadecimal.
func (f *fmt) fmt_X32(v int32)	{ f.fmt_X64(int64(v)) }

// fmt_X formats an int in upper case hexadecimal.
func (f *fmt) fmt_X(v int)	{ f.fmt_X64(int64(v)) }

// fmt_uX64 formats a uint64 in upper case hexadecimal.
func (f *fmt) fmt_uX64(v uint64) {
	f.padBytes(f.integer(int64(v), 16, false, udigits));
	f.clearflags();
}

// fmt_uX32 formats a uint32 in upper case hexadecimal.
func (f *fmt) fmt_uX32(v uint32)	{ f.fmt_uX64(uint64(v)) }

// fmt_uX formats a uint in upper case hexadecimal.
func (f *fmt) fmt_uX(v uint)	{ f.fmt_uX64(uint64(v)) }

// fmt_o64 formats an int64 in octal.
func (f *fmt) fmt_o64(v int64) {
	f.padBytes(f.integer(v, 8, true, ldigits));
	f.clearflags();
}

// fmt_o32 formats an int32 in octal.
func (f *fmt) fmt_o32(v int32)	{ f.fmt_o64(int64(v)) }

// fmt_o formats an int in octal.
func (f *fmt) fmt_o(v int)	{ f.fmt_o64(int64(v)) }

// fmt_uo64 formats a uint64 in octal.
func (f *fmt) fmt_uo64(v uint64) {
	f.padBytes(f.integer(int64(v), 8, false, ldigits));
	f.clearflags();
}

// fmt_uo32 formats a uint32 in octal.
func (f *fmt) fmt_uo32(v uint32)	{ f.fmt_uo64(uint64(v)) }

// fmt_uo formats a uint in octal.
func (f *fmt) fmt_uo(v uint)	{ f.fmt_uo64(uint64(v)) }

// fmt_b64 formats a uint64 in binary.
func (f *fmt) fmt_b64(v uint64) {
	f.padBytes(f.integer(int64(v), 2, false, ldigits));
	f.clearflags();
}

// fmt_b32 formats a uint32 in binary.
func (f *fmt) fmt_b32(v uint32)	{ f.fmt_b64(uint64(v)) }

// fmt_b formats a uint in binary.
func (f *fmt) fmt_b(v uint)	{ f.fmt_b64(uint64(v)) }

// fmt_c formats a Unicode character.
func (f *fmt) fmt_c(v int) {
	f.pad(string(v));
	f.clearflags();
}

// fmt_s formats a string.
func (f *fmt) fmt_s(s string) {
	if f.precPresent {
		if f.prec < len(s) {
			s = s[0:f.prec]
		}
	}
	f.pad(s);
	f.clearflags();
}

// fmt_sx formats a string as a hexadecimal encoding of its bytes.
func (f *fmt) fmt_sx(s string) {
	t := "";
	for i := 0; i < len(s); i++ {
		if i > 0 && f.space {
			t += " "
		}
		v := s[i];
		t += string(ldigits[v>>4]);
		t += string(ldigits[v&0xF]);
	}
	f.pad(t);
	f.clearflags();
}

// fmt_sX formats a string as an uppercase hexadecimal encoding of its bytes.
func (f *fmt) fmt_sX(s string) {
	t := "";
	for i := 0; i < len(s); i++ {
		v := s[i];
		t += string(udigits[v>>4]);
		t += string(udigits[v&0xF]);
	}
	f.pad(t);
	f.clearflags();
}

// fmt_q formats a string as a double-quoted, escaped Go string constant.
func (f *fmt) fmt_q(s string) {
	var quoted string;
	if f.sharp && strconv.CanBackquote(s) {
		quoted = "`" + s + "`"
	} else {
		quoted = strconv.Quote(s)
	}
	f.pad(quoted);
	f.clearflags();
}

// floating-point

func doPrec(f *fmt, def int) int {
	if f.precPresent {
		return f.prec
	}
	return def;
}

func fmtString(f *fmt, s string) {
	f.pad(s);
	f.clearflags();
}

// Add a plus sign or space to the string if missing and required.
func (f *fmt) plusSpace(s string) {
	if s[0] != '-' {
		if f.plus {
			s = "+" + s
		} else if f.space {
			s = " " + s
		}
	}
	fmtString(f, s);
}

// fmt_e64 formats a float64 in the form -1.23e+12.
func (f *fmt) fmt_e64(v float64)	{ f.plusSpace(strconv.Ftoa64(v, 'e', doPrec(f, 6))) }

// fmt_E64 formats a float64 in the form -1.23E+12.
func (f *fmt) fmt_E64(v float64)	{ f.plusSpace(strconv.Ftoa64(v, 'E', doPrec(f, 6))) }

// fmt_f64 formats a float64 in the form -1.23.
func (f *fmt) fmt_f64(v float64)	{ f.plusSpace(strconv.Ftoa64(v, 'f', doPrec(f, 6))) }

// fmt_g64 formats a float64 in the 'f' or 'e' form according to size.
func (f *fmt) fmt_g64(v float64)	{ f.plusSpace(strconv.Ftoa64(v, 'g', doPrec(f, -1))) }

// fmt_g64 formats a float64 in the 'f' or 'E' form according to size.
func (f *fmt) fmt_G64(v float64)	{ f.plusSpace(strconv.Ftoa64(v, 'G', doPrec(f, -1))) }

// fmt_fb64 formats a float64 in the form -123p3 (exponent is power of 2).
func (f *fmt) fmt_fb64(v float64)	{ f.plusSpace(strconv.Ftoa64(v, 'b', 0)) }

// float32
// cannot defer to float64 versions
// because it will get rounding wrong in corner cases.

// fmt_e32 formats a float32 in the form -1.23e+12.
func (f *fmt) fmt_e32(v float32)	{ f.plusSpace(strconv.Ftoa32(v, 'e', doPrec(f, 6))) }

// fmt_E32 formats a float32 in the form -1.23E+12.
func (f *fmt) fmt_E32(v float32)	{ f.plusSpace(strconv.Ftoa32(v, 'E', doPrec(f, 6))) }

// fmt_f32 formats a float32 in the form -1.23.
func (f *fmt) fmt_f32(v float32)	{ f.plusSpace(strconv.Ftoa32(v, 'f', doPrec(f, 6))) }

// fmt_g32 formats a float32 in the 'f' or 'e' form according to size.
func (f *fmt) fmt_g32(v float32)	{ f.plusSpace(strconv.Ftoa32(v, 'g', doPrec(f, -1))) }

// fmt_G32 formats a float32 in the 'f' or 'E' form according to size.
func (f *fmt) fmt_G32(v float32)	{ f.plusSpace(strconv.Ftoa32(v, 'G', doPrec(f, -1))) }

// fmt_fb32 formats a float32 in the form -123p3 (exponent is power of 2).
func (f *fmt) fmt_fb32(v float32)	{ fmtString(f, strconv.Ftoa32(v, 'b', 0)) }

// float
func (x *fmt) f(a float) {
	if strconv.FloatSize == 32 {
		x.fmt_f32(float32(a))
	} else {
		x.fmt_f64(float64(a))
	}
}

func (x *fmt) e(a float) {
	if strconv.FloatSize == 32 {
		x.fmt_e32(float32(a))
	} else {
		x.fmt_e64(float64(a))
	}
}

func (x *fmt) g(a float) {
	if strconv.FloatSize == 32 {
		x.fmt_g32(float32(a))
	} else {
		x.fmt_g64(float64(a))
	}
}

func (x *fmt) fb(a float) {
	if strconv.FloatSize == 32 {
		x.fmt_fb32(float32(a))
	} else {
		x.fmt_fb64(float64(a))
	}
}
