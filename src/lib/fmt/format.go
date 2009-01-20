// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fmt

import (
	"strconv";
)

/*
	Raw formatter. See print.go for a more palatable interface.

	f := fmt.New();
	print f.Fmt_d(1234).Fmt_s("\n").Str();  // create string, print it
	f.Fmt_d(-1234).Fmt_s("\n").put();  // print string
	f.Fmt_ud(1<<63).Putnl();  // print string with automatic newline
*/

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

func New() *Fmt {
	f := new(Fmt);
	f.init();
	return f;
}

func (f *Fmt) Str() string {
	s := f.buf;
	f.clearbuf();
	f.clearflags();
	f.buf = "";
	return s;
}

func (f *Fmt) Put() {
	print(f.buf);
	f.clearbuf();
	f.clearflags();
}

func (f *Fmt) Putnl() {
	print(f.buf, "\n");
	f.clearbuf();
	f.clearflags();
}

func (f *Fmt) Wp(w, p int) *Fmt {
	f.wid_present = true;
	f.wid = w;
	f.prec_present = true;
	f.prec = p;
	return f;
}

func (f *Fmt) P(p int) *Fmt {
	f.prec_present = true;
	f.prec = p;
	return f;
}

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

// boolean
func (f *Fmt) Fmt_boolean(a bool) *Fmt {
	if a {
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
	return string(buf)[i+1:nByte];
}

// decimal
func (f *Fmt) Fmt_d64(a int64) *Fmt {
	f.pad(f.integer(a, 10, true, &ldigits));
	f.clearflags();
	return f;
}

func (f *Fmt) Fmt_d32(a int32) *Fmt {
	return f.Fmt_d64(int64(a));
}

func (f *Fmt) Fmt_d(a int) *Fmt {
	return f.Fmt_d64(int64(a));
}

// unsigned Fmt_decimal
func (f *Fmt) Fmt_ud64(a uint64) *Fmt {
	f.pad(f.integer(int64(a), 10, false, &ldigits));
	f.clearflags();
	return f;
}

func (f *Fmt) Fmt_ud32(a uint32) *Fmt {
	return f.Fmt_ud64(uint64(a));
}

func (f *Fmt) Fmt_ud(a uint) *Fmt {
	return f.Fmt_ud64(uint64(a));
}

// hexdecimal
func (f *Fmt) Fmt_x64(a int64) *Fmt {
	f.pad(f.integer(a, 16, true, &ldigits));
	f.clearflags();
	return f;
}

func (f *Fmt) Fmt_x32(a int32) *Fmt {
	return f.Fmt_x64(int64(a));
}

func (f *Fmt) Fmt_x(a int) *Fmt {
	return f.Fmt_x64(int64(a));
}

// unsigned hexdecimal
func (f *Fmt) Fmt_ux64(a uint64) *Fmt {
	f.pad(f.integer(int64(a), 16, false, &ldigits));
	f.clearflags();
	return f;
}

func (f *Fmt) Fmt_ux32(a uint32) *Fmt {
	return f.Fmt_ux64(uint64(a));
}

func (f *Fmt) Fmt_ux(a uint) *Fmt {
	return f.Fmt_ux64(uint64(a));
}

// HEXADECIMAL
func (f *Fmt) Fmt_X64(a int64) *Fmt {
	f.pad(f.integer(a, 16, true, &udigits));
	f.clearflags();
	return f;
}

func (f *Fmt) Fmt_X32(a int32) *Fmt {
	return f.Fmt_X64(int64(a));
}

func (f *Fmt) Fmt_X(a int) *Fmt {
	return f.Fmt_X64(int64(a));
}

// unsigned HEXADECIMAL
func (f *Fmt) Fmt_uX64(a uint64) *Fmt {
	f.pad(f.integer(int64(a), 16, false, &udigits));
	f.clearflags();
	return f;
}

func (f *Fmt) Fmt_uX32(a uint32) *Fmt {
	return f.Fmt_uX64(uint64(a));
}

func (f *Fmt) Fmt_uX(a uint) *Fmt {
	return f.Fmt_uX64(uint64(a));
}

// octal
func (f *Fmt) Fmt_o64(a int64) *Fmt {
	f.pad(f.integer(a, 8, true, &ldigits));
	f.clearflags();
	return f;
}

func (f *Fmt) Fmt_o32(a int32) *Fmt {
	return f.Fmt_o64(int64(a));
}

func (f *Fmt) Fmt_o(a int) *Fmt {
	return f.Fmt_o64(int64(a));
}


// unsigned octal
func (f *Fmt) Fmt_uo64(a uint64) *Fmt {
	f.pad(f.integer(int64(a), 8, false, &ldigits));
	f.clearflags();
	return f;
}

func (f *Fmt) Fmt_uo32(a uint32) *Fmt {
	return f.Fmt_uo64(uint64(a));
}

func (f *Fmt) Fmt_uo(a uint) *Fmt {
	return f.Fmt_uo64(uint64(a));
}


// unsigned binary
func (f *Fmt) Fmt_b64(a uint64) *Fmt {
	f.pad(f.integer(int64(a), 2, false, &ldigits));
	f.clearflags();
	return f;
}

func (f *Fmt) Fmt_b32(a uint32) *Fmt {
	return f.Fmt_b64(uint64(a));
}

func (f *Fmt) Fmt_b(a uint) *Fmt {
	return f.Fmt_b64(uint64(a));
}


// character
func (f *Fmt) Fmt_c(a int) *Fmt {
	f.pad(string(a));
	f.clearflags();
	return f;
}

// string
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

// hexadecimal string
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

// quoted string
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

// float64
func (f *Fmt) Fmt_e64(a float64) *Fmt {
	return fmtString(f, strconv.Ftoa64(a, 'e', doPrec(f, 6)));
}

func (f *Fmt) Fmt_f64(a float64) *Fmt {
	return fmtString(f, strconv.Ftoa64(a, 'f', doPrec(f, 6)));
}

func (f *Fmt) Fmt_g64(a float64) *Fmt {
	return fmtString(f, strconv.Ftoa64(a, 'g', doPrec(f, -1)));
}

func (f *Fmt) Fmt_fb64(a float64) *Fmt {
	return fmtString(f, strconv.Ftoa64(a, 'b', 0));
}

// float32
// cannot defer to float64 versions
// because it will get rounding wrong in corner cases.
func (f *Fmt) Fmt_e32(a float32) *Fmt {
	return fmtString(f, strconv.Ftoa32(a, 'e', doPrec(f, 6)));
}

func (f *Fmt) Fmt_f32(a float32) *Fmt {
	return fmtString(f, strconv.Ftoa32(a, 'f', doPrec(f, 6)));
}

func (f *Fmt) Fmt_g32(a float32) *Fmt {
	return fmtString(f, strconv.Ftoa32(a, 'g', doPrec(f, -1)));
}

func (f *Fmt) Fmt_fb32(a float32) *Fmt {
	return fmtString(f, strconv.Ftoa32(a, 'b', 0));
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
