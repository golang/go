// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fmt

import (
	"strconv";
)


const nByte = 64;
const nPows10 = 160;

var ldigits string = "0123456789abcdef"  // var not const because we take its address
var udigits string = "0123456789ABCDEF"
var pows10 [nPows10] float64;

func init() {
	pows10[0] = 1.0e0;
	pows10[1] = 1.0e1;
	for i:=2; i<nPows10; i++ {
		m := i/2;
		pows10[i] = pows10[m] * pows10[i-m];
	}
}

/*
	Fmt is the raw formatter used by Printf etc.  Not meant for normal use.
	See print.go for a more palatable interface.

	Model is to accumulate operands into an internal buffer and then
	retrieve the buffer in one hit using Str(), Putnl(), etc.  The formatting
	methods return ``self'' so the operations can be chained.

	f := fmt.New();
	print(f.Fmt_d(1234).Fmt_s("\n").Str());  // create string, print it
	f.Fmt_d(-1234).Fmt_s("\n").Put();  // print string
	f.Fmt_ud(1<<63).Putnl();  // print string with automatic newline
*/
type Fmt struct {
	buf string;
	wid int;
	wid_present bool;
	prec int;
	prec_present bool;
	// flags
	minus bool;
	plus bool;
	sharp bool;
	space bool;
	zero bool;
}

func (f *Fmt) clearflags() {
	f.wid = 0;
	f.wid_present = false;
	f.prec = 0;
	f.prec_present = false;
	f.minus = false;
	f.plus = false;
	f.sharp = false;
	f.space = false;
	f.zero = false;
}

func (f *Fmt) clearbuf() {
	f.buf = "";
}

func (f *Fmt) init() {
	f.clearbuf();
	f.clearflags();
}

// New returns a new initialized Fmt
func New() *Fmt {
	f := new(Fmt);
	f.init();
	return f;
}

// Str returns the buffered contents as a string and resets the Fmt.
func (f *Fmt) Str() string {
	s := f.buf;
	f.clearbuf();
	f.clearflags();
	f.buf = "";
	return s;
}

// Put writes the buffered contents to stdout and resets the Fmt.
func (f *Fmt) Put() {
	print(f.buf);
	f.clearbuf();
	f.clearflags();
}

// Putnl writes the buffered contents to stdout, followed by a newline, and resets the Fmt.
func (f *Fmt) Putnl() {
	print(f.buf, "\n");
	f.clearbuf();
	f.clearflags();
}

// Wp sets the width and precision for formatting the next item.
func (f *Fmt) Wp(w, p int) *Fmt {
	f.wid_present = true;
	f.wid = w;
	f.prec_present = true;
	f.prec = p;
	return f;
}

// P sets the precision for formatting the next item.
func (f *Fmt) P(p int) *Fmt {
	f.prec_present = true;
	f.prec = p;
	return f;
}

// W sets the width for formatting the next item.
func (f *Fmt) W(x int) *Fmt {
	f.wid_present = true;
	f.wid = x;
	return f;
}

// append s to buf, padded on left (w > 0) or right (w < 0 or f.minus)
// padding is in bytes, not characters (agrees with ANSIC C, not Plan 9 C)
func (f *Fmt) pad(s string) {
	if f.wid_present && f.wid != 0 {
		left := !f.minus;
		w := f.wid;
		if w < 0 {
			left = false;
			w = -w;
		}
		w -= len(s);
		padchar := byte(' ');
		if left && f.zero {
			padchar = '0';
		}
		if w > 0 {
			if w > nByte {
				w = nByte;
			}
			buf := make([]byte, w);
			for i := 0; i < w; i++ {
				buf[i] = padchar;
			}
			if left {
				s = string(buf) + s;
			} else {
				s = s + string(buf);
			}
		}
	}
	f.buf += s;
}

// format val into buf, ending at buf[i].  (printing is easier right-to-left;
// that's why the bidi languages are right-to-left except for numbers. wait,
// never mind.)  val is known to be unsigned.  we could make things maybe
// marginally faster by splitting the 32-bit case out into a separate function
// but it's not worth the duplication, so val has 64 bits.
func putint(buf *[nByte]byte, i int, base, val uint64, digits *string) int {
	for val >= base {
		buf[i] = digits[val%base];
		i--;
		val /= base;
	}
	buf[i] = digits[val];
	return i-1;
}

// Fmt_boolean formats a boolean.
func (f *Fmt) Fmt_boolean(v bool) *Fmt {
	if v {
		f.pad("true");
	} else {
		f.pad("false");
	}
	f.clearflags();
	return f;
}

// integer; interprets prec but not wid.
func (f *Fmt) integer(a int64, base uint, is_signed bool, digits *string) string {
	var buf [nByte]byte;
	negative := is_signed && a < 0;
	if negative {
		a = -a;
	}

	// two ways to ask for extra leading zero digits: %.3d or %03d.
	// apparently the first cancels the second.
	prec := 0;
	if f.prec_present {
		prec = f.prec;
		f.zero = false;
	} else if f.zero && f.wid_present && !f.minus && f.wid > 0{
		prec = f.wid;
		if negative || f.plus || f.space {
			prec--;  // leave room for sign
		}
	}

	i := putint(&buf, nByte-1, uint64(base), uint64(a), digits);
	for i > 0 && prec > (nByte-1-i) {
		buf[i] = '0';
		i--;
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
	return string(buf[i+1:nByte]);
}

// Fmt_d64 formats an int64 in decimal.
func (f *Fmt) Fmt_d64(v int64) *Fmt {
	f.pad(f.integer(v, 10, true, &ldigits));
	f.clearflags();
	return f;
}

// Fmt_d32 formats an int32 in decimal.
func (f *Fmt) Fmt_d32(v int32) *Fmt {
	return f.Fmt_d64(int64(v));
}

// Fmt_d formats an int in decimal.
func (f *Fmt) Fmt_d(v int) *Fmt {
	return f.Fmt_d64(int64(v));
}

// Fmt_ud64 formats a uint64 in decimal.
func (f *Fmt) Fmt_ud64(v uint64) *Fmt {
	f.pad(f.integer(int64(v), 10, false, &ldigits));
	f.clearflags();
	return f;
}

// Fmt_ud32 formats a uint32 in decimal.
func (f *Fmt) Fmt_ud32(v uint32) *Fmt {
	return f.Fmt_ud64(uint64(v));
}

// Fmt_ud formats a uint in decimal.
func (f *Fmt) Fmt_ud(v uint) *Fmt {
	return f.Fmt_ud64(uint64(v));
}

// Fmt_x64 formats an int64 in hexadecimal.
func (f *Fmt) Fmt_x64(v int64) *Fmt {
	f.pad(f.integer(v, 16, true, &ldigits));
	f.clearflags();
	return f;
}

// Fmt_x32 formats an int32 in hexadecimal.
func (f *Fmt) Fmt_x32(v int32) *Fmt {
	return f.Fmt_x64(int64(v));
}

// Fmt_x formats an int in hexadecimal.
func (f *Fmt) Fmt_x(v int) *Fmt {
	return f.Fmt_x64(int64(v));
}

// Fmt_ux64 formats a uint64 in hexadecimal.
func (f *Fmt) Fmt_ux64(v uint64) *Fmt {
	f.pad(f.integer(int64(v), 16, false, &ldigits));
	f.clearflags();
	return f;
}

// Fmt_ux32 formats a uint32 in hexadecimal.
func (f *Fmt) Fmt_ux32(v uint32) *Fmt {
	return f.Fmt_ux64(uint64(v));
}

// Fmt_ux formats a uint in hexadecimal.
func (f *Fmt) Fmt_ux(v uint) *Fmt {
	return f.Fmt_ux64(uint64(v));
}

// Fmt_X64 formats an int64 in upper case hexadecimal.
func (f *Fmt) Fmt_X64(v int64) *Fmt {
	f.pad(f.integer(v, 16, true, &udigits));
	f.clearflags();
	return f;
}

// Fmt_X32 formats an int32 in upper case hexadecimal.
func (f *Fmt) Fmt_X32(v int32) *Fmt {
	return f.Fmt_X64(int64(v));
}

// Fmt_X formats an int in upper case hexadecimal.
func (f *Fmt) Fmt_X(v int) *Fmt {
	return f.Fmt_X64(int64(v));
}

// Fmt_uX64 formats a uint64 in upper case hexadecimal.
func (f *Fmt) Fmt_uX64(v uint64) *Fmt {
	f.pad(f.integer(int64(v), 16, false, &udigits));
	f.clearflags();
	return f;
}

// Fmt_uX32 formats a uint32 in upper case hexadecimal.
func (f *Fmt) Fmt_uX32(v uint32) *Fmt {
	return f.Fmt_uX64(uint64(v));
}

// Fmt_uX formats a uint in upper case hexadecimal.
func (f *Fmt) Fmt_uX(v uint) *Fmt {
	return f.Fmt_uX64(uint64(v));
}

// Fmt_o64 formats an int64 in octal.
func (f *Fmt) Fmt_o64(v int64) *Fmt {
	f.pad(f.integer(v, 8, true, &ldigits));
	f.clearflags();
	return f;
}

// Fmt_o32 formats an int32 in octal.
func (f *Fmt) Fmt_o32(v int32) *Fmt {
	return f.Fmt_o64(int64(v));
}

// Fmt_o formats an int in octal.
func (f *Fmt) Fmt_o(v int) *Fmt {
	return f.Fmt_o64(int64(v));
}

// Fmt_uo64 formats a uint64 in octal.
func (f *Fmt) Fmt_uo64(v uint64) *Fmt {
	f.pad(f.integer(int64(v), 8, false, &ldigits));
	f.clearflags();
	return f;
}

// Fmt_uo32 formats a uint32 in octal.
func (f *Fmt) Fmt_uo32(v uint32) *Fmt {
	return f.Fmt_uo64(uint64(v));
}

// Fmt_uo formats a uint in octal.
func (f *Fmt) Fmt_uo(v uint) *Fmt {
	return f.Fmt_uo64(uint64(v));
}

// Fmt_b64 formats a uint64 in binary.
func (f *Fmt) Fmt_b64(v uint64) *Fmt {
	f.pad(f.integer(int64(v), 2, false, &ldigits));
	f.clearflags();
	return f;
}

// Fmt_b32 formats a uint32 in binary.
func (f *Fmt) Fmt_b32(v uint32) *Fmt {
	return f.Fmt_b64(uint64(v));
}

// Fmt_b formats a uint in binary.
func (f *Fmt) Fmt_b(v uint) *Fmt {
	return f.Fmt_b64(uint64(v));
}

// Fmt_c formats a Unicode character.
func (f *Fmt) Fmt_c(v int) *Fmt {
	f.pad(string(v));
	f.clearflags();
	return f;
}

// Fmt_s formats a string.
func (f *Fmt) Fmt_s(s string) *Fmt {
	if f.prec_present {
		if f.prec < len(s) {
			s = s[0:f.prec];
		}
	}
	f.pad(s);
	f.clearflags();
	return f;
}

// Fmt_sx formats a string as a hexadecimal encoding of its bytes.
func (f *Fmt) Fmt_sx(s string) *Fmt {
	t := "";
	for i := 0; i < len(s); i++ {
		if i > 0 && f.space {
			t += " ";
		}
		v := s[i];
		t += string(ldigits[v>>4]);
		t += string(ldigits[v&0xF]);
	}
	f.pad(t);
	f.clearflags();
	return f;
}

// Fmt_sX formats a string as an uppercase hexadecimal encoding of its bytes.
func (f *Fmt) Fmt_sX(s string) *Fmt {
	t := "";
	for i := 0; i < len(s); i++ {
		v := s[i];
		t += string(udigits[v>>4]);
		t += string(udigits[v&0xF]);
	}
	f.pad(t);
	f.clearflags();
	return f;
}

// Fmt_q formats a string as a double-quoted, escaped Go string constant.
func (f *Fmt) Fmt_q(s string) *Fmt {
	var quoted string;
	if f.sharp && strconv.CanBackquote(s) {
		quoted = "`"+s+"`";
	} else {
		quoted = strconv.Quote(s);
	}
	f.pad(quoted);
	f.clearflags();
	return f;
}

// floating-point

func doPrec(f *Fmt, def int) int {
	if f.prec_present {
		return f.prec;
	}
	return def;
}

func fmtString(f *Fmt, s string) *Fmt {
	f.pad(s);
	f.clearflags();
	return f;
}

// Fmt_e64 formats a float64 in the form -1.23e+12.
func (f *Fmt) Fmt_e64(v float64) *Fmt {
	return fmtString(f, strconv.Ftoa64(v, 'e', doPrec(f, 6)));
}

// Fmt_f64 formats a float64 in the form -1.23.
func (f *Fmt) Fmt_f64(v float64) *Fmt {
	return fmtString(f, strconv.Ftoa64(v, 'f', doPrec(f, 6)));
}

// Fmt_g64 formats a float64 in the 'f' or 'e' form according to size.
func (f *Fmt) Fmt_g64(v float64) *Fmt {
	return fmtString(f, strconv.Ftoa64(v, 'g', doPrec(f, -1)));
}

// Fmt_fb64 formats a float64 in the form -123p3 (exponent is power of 2).
func (f *Fmt) Fmt_fb64(v float64) *Fmt {
	return fmtString(f, strconv.Ftoa64(v, 'b', 0));
}

// float32
// cannot defer to float64 versions
// because it will get rounding wrong in corner cases.

// Fmt_e32 formats a float32 in the form -1.23e+12.
func (f *Fmt) Fmt_e32(v float32) *Fmt {
	return fmtString(f, strconv.Ftoa32(v, 'e', doPrec(f, 6)));
}

// Fmt_f32 formats a float32 in the form -1.23.
func (f *Fmt) Fmt_f32(v float32) *Fmt {
	return fmtString(f, strconv.Ftoa32(v, 'f', doPrec(f, 6)));
}

// Fmt_g32 formats a float32 in the 'f' or 'e' form according to size.
func (f *Fmt) Fmt_g32(v float32) *Fmt {
	return fmtString(f, strconv.Ftoa32(v, 'g', doPrec(f, -1)));
}

// Fmt_fb32 formats a float32 in the form -123p3 (exponent is power of 2).
func (f *Fmt) Fmt_fb32(v float32) *Fmt {
	return fmtString(f, strconv.Ftoa32(v, 'b', 0));
}

// float
func (x *Fmt) f(a float) *Fmt {
	if strconv.FloatSize == 32 {
		return x.Fmt_f32(float32(a))
	}
	return x.Fmt_f64(float64(a))
}

func (x *Fmt) e(a float) *Fmt {
	if strconv.FloatSize == 32 {
		return x.Fmt_e32(float32(a))
	}
	return x.Fmt_e64(float64(a))
}

func (x *Fmt) g(a float) *Fmt {
	if strconv.FloatSize == 32 {
		return x.Fmt_g32(float32(a))
	}
	return x.Fmt_g64(float64(a))
}

func (x *Fmt) fb(a float) *Fmt {
	if strconv.FloatSize == 32 {
		return x.Fmt_fb32(float32(a))
	}
	return x.Fmt_fb64(float64(a))
}
