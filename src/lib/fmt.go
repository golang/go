// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fmt

/*
	f := fmt.New();
	print f.d(1234).s("\n").str();  // create string, print it
	f.d(-1234).s("\n").put();  // print string
	f.ud(^0).putnl();  // print string with automatic newline
*/

// export Fmt, New;

const NByte = 64;
const NPows10 = 160;

var ldigits string = "0123456789abcdef"  // var not const because we take its address
var udigits string = "0123456789ABCDEF"
var pows10 [NPows10] double;

func init() {
	pows10[0] = 1.0e0;
	pows10[1] = 1.0e1;
	for i:=2; i<NPows10; i++ {
		m := i/2;
		pows10[i] = pows10[m] * pows10[i-m];
	}
}

export type Fmt struct {
	buf string;
	wid int;
	wid_present bool;
	prec int;
	prec_present bool;
}

func (f *Fmt) clearflags() {
	f.wid_present = false;
	f.prec_present = false;
}

func (f *Fmt) clearbuf() {
	f.buf = "";
}

func (f *Fmt) init() {
	f.clearbuf();
	f.clearflags();
}

export func New() *Fmt {
	f := new(Fmt);
	f.init();
	return f;
}

func (f *Fmt) str() string {
	s := f.buf;
	f.clearbuf();
	f.clearflags();
	f.buf = "";
	return s;
}

func (f *Fmt) put() {
	print(f.buf);
	f.clearbuf();
	f.clearflags();
}

func (f *Fmt) putnl() {
	print(f.buf, "\n");
	f.clearbuf();
	f.clearflags();
}

func (f *Fmt) wp(w, p int) *Fmt {
	f.wid_present = true;
	f.wid = w;
	f.prec_present = true;
	f.prec = p;
	return f;
}

func (f *Fmt) p(p int) *Fmt {
	f.prec_present = true;
	f.prec = p;
	return f;
}

func (f *Fmt) w(x int) *Fmt {
	f.wid_present = true;
	f.wid = x;
	return f;
}

// append s to buf, padded on left (w > 0) or right (w < 0)
// padding is in bytes, not characters (agrees with ANSIC C, not Plan 9 C)
func (f *Fmt) pad(s string) {
	if f.wid_present && f.wid != 0 {
		left := true;
		w := f.wid;
		if w < 0 {
			left = false;
			w = -w;
		}
		w -= len(s);
		if w > 0 {
			if w > NByte {
				w = NByte;
			}
			buf := new([]byte, w);
			for i := 0; i < w; i++ {
				buf[i] = ' ';
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
func putint(buf *[NByte]byte, i int, base, val uint64, digits *string) int {
	for val >= base {
		buf[i] = digits[val%base];
		i--;
		val /= base;
	}
	buf[i] = digits[val];
	return i-1;
}

// boolean
func (f *Fmt) boolean(a bool) *Fmt {
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
	var buf [NByte]byte;
	negative := is_signed && a < 0;
	if negative {
		a = -a;
	}
	i := putint(&buf, NByte-1, uint64(base), uint64(a), digits);
	if f.prec_present {
		for i > 0 && f.prec > (NByte-1-i) {
			buf[i] = '0';
			i--;
		}
	}
	if negative {
		buf[i] = '-';
		i--;
	}
	return string(buf)[i+1:NByte];
}

// decimal
func (f *Fmt) d(a int32) *Fmt {
	f.pad(f.integer(int64(a), 10, true, &ldigits));
	f.clearflags();
	return f;
}

func (f *Fmt) D(a int64) *Fmt {
	f.pad(f.integer(a, 10, true, &ldigits));
	f.clearflags();
	return f;
}

// unsigned decimal
func (f *Fmt) ud(a int32) *Fmt {
	f.pad(f.integer(int64(uint32(a)), 10, false, &ldigits));
	f.clearflags();
	return f;
}

func (f *Fmt) uD(a int64) *Fmt {
	f.pad(f.integer(a, 10, false, &ldigits));
	f.clearflags();
	return f;
}

// hexdecimal
func (f *Fmt) x(a int32) *Fmt {
	f.pad(f.integer(int64(a), 16, true, &ldigits));
	f.clearflags();
	return f;
}

func (f *Fmt) X(a int64) *Fmt {
	f.pad(f.integer(a, 16, true, &ldigits));
	f.clearflags();
	return f;
}

// unsigned hexdecimal
func (f *Fmt) ux(a int32) *Fmt {
	f.pad(f.integer(int64(uint32(a)), 16, false, &ldigits));
	f.clearflags();
	return f;
}

func (f *Fmt) uX(a int64) *Fmt {
	f.pad(f.integer(a, 16, false, &ldigits));
	f.clearflags();
	return f;
}

// HEXADECIMAL
func (f *Fmt) Ux(a int32) *Fmt {
	f.pad(f.integer(int64(a), 16, true, &udigits));
	f.clearflags();
	return f;
}

func (f *Fmt) UX(a int64) *Fmt {
	f.pad(f.integer(a, 16, true, &udigits));
	f.clearflags();
	return f;
}

// unsigned HEXADECIMAL
func (f *Fmt) uUx(a int32) *Fmt {
	f.pad(f.integer(int64(uint32(a)), 16, false, &udigits));
	f.clearflags();
	return f;
}

func (f *Fmt) uUX(a int64) *Fmt {
	f.pad(f.integer(a, 16, false, &udigits));
	f.clearflags();
	return f;
}

// octal
func (f *Fmt) o(a int32) *Fmt {
	f.pad(f.integer(int64(a), 8, true, &ldigits));
	f.clearflags();
	return f;
}

func (f *Fmt) O(a int64) *Fmt {
	f.pad(f.integer(a, 8, true, &ldigits));
	f.clearflags();
	return f;
}

// unsigned octal
func (f *Fmt) uo(a int32) *Fmt {
	f.pad(f.integer(int64(uint32(a)), 8, false, &ldigits));
	f.clearflags();
	return f;
}

func (f *Fmt) uO(a int64) *Fmt {
	f.pad(f.integer(a, 8, false, &ldigits));
	f.clearflags();
	return f;
}

// binary
func (f *Fmt) b(a int32) *Fmt {
	f.pad(f.integer(int64(uint32(a)), 2, false, &ldigits));
	f.clearflags();
	return f;
}

func (f *Fmt) B(a int64) *Fmt {
	f.pad(f.integer(a, 2, false, &ldigits));
	f.clearflags();
	return f;
}

// character
func (f *Fmt) c(a int) *Fmt {
	f.pad(string(a));
	f.clearflags();
	return f;
}

// string
func (f *Fmt) s(s string) *Fmt {
	if f.prec_present {
		if f.prec < len(s) {
			s = s[0:f.prec];
		}
	}
	f.pad(s);
	f.clearflags();
	return f;
}

func pow10(n int) double {
	var d double;

	neg := false;
	if n < 0 {
		if n < -307 {  // DBL_MIN_10_EXP
			return 0.;
		}
		neg = true;
		n = -n;
	}else if n > 308 { // DBL_MAX_10_EXP
		return 1.79769e+308; // HUGE_VAL
	}

	if n < NPows10 {
		d = pows10[n];
	} else {
		d = pows10[NPows10-1];
		for {
			n -= NPows10 - 1;
			if n < NPows10 {
				d *= pows10[n];
				break;
			}
			d *= pows10[NPows10 - 1];
		}
	}
	if neg {
		return 1/d;
	}
	return d;
}

func unpack(a double) (negative bool, exp int, num double) {
	if a == 0 {
		return false, 0, 0.0
	}
	neg := a < 0;
	if neg {
		a = -a;
	}
	// find g,e such that a = g*10^e.
	// guess 10-exponent using 2-exponent, then fine tune.
	g, e2 := sys.frexp(a);
	e := int(double(e2) * .301029995663981);
	g = a * pow10(-e);
	for g < 1 {
		e--;
		g = a * pow10(-e);
	}
	for g >= 10 {
		e++;
		g = a * pow10(-e);
	}
	return neg, e, g;
}

// check for Inf, NaN
func(f *Fmt) InfOrNan(a double) bool {
	if sys.isInf(a, 0) {
		if sys.isInf(a, 1) {
			f.pad("Inf");
		} else {
			f.pad("-Inf");
		}
		f.clearflags();
		return true;
	}
	if sys.isNaN(a) {
		f.pad("NaN");
		f.clearflags();
		return true;
	}
	return false;
}

// double
func (f *Fmt) E(a double) *Fmt {
	var negative bool;
	var g double;
	var exp int;
	if f.InfOrNan(a) {
		return f;
	}
	negative, exp, g = unpack(a);
	prec := 6;
	if f.prec_present {
		prec = f.prec;
	}
	prec++;  // one digit left of decimal
	var s string;
	// multiply by 10^prec to get decimal places; put decimal after first digit
	if g == 0 {
		// doesn't work for zero - fake it
		s = "000000000000000000000000000000000000000000000000000000000000";
		if prec < len(s) {
			s = s[0:prec];
		} else {
			prec = len(s);
		}
	} else {
		g *= pow10(prec);
		s = f.integer(int64(g + .5), 10, true, &ldigits);  // get the digits into a string
	}
	s = s[0:1] + "." + s[1:prec];  // insert a decimal point
	// print exponent with leading 0 if appropriate.
	es := New().p(2).integer(int64(exp), 10, true, &ldigits);
	if exp >= 0 {
		es = "+" + es;  // TODO: should do this with a fmt flag
	}
	s = s + "e" + es;
	if negative {
		s = "-" + s;
	}
	f.pad(s);
	f.clearflags();
	return f;
}

// double
func (f *Fmt) F(a double) *Fmt {
	var negative bool;
	var g double;
	var exp int;
	if f.InfOrNan(a) {
		return f;
	}
	negative, exp, g = unpack(a);
	if exp > 19 || exp < -19 {  // too big for this sloppy code
		return f.E(a);
	}
	prec := 6;
	if f.prec_present {
		prec = f.prec;
	}
	// prec is number of digits after decimal point
	s := "NO";
	if exp >= 0 {
		g *= pow10(exp);
		gi := int64(g);
		s = New().integer(gi, 10, true, &ldigits);
		s = s + ".";
		g -= double(gi);
		s = s + New().p(prec).integer(int64(g*pow10(prec) + .5), 10, true, &ldigits);
	} else {
		g *= pow10(prec + exp);
		s = "0." + New().p(prec).integer(int64(g + .5), 10, true, &ldigits);
	}
	if negative {
		s = "-" + s;
	}
	f.pad(s);
	f.clearflags();
	return f;
}

// double
func (f *Fmt) G(a double) *Fmt {
	if f.InfOrNan(a) {
		return f;
	}
	f1 := New();
	f2 := New();
	if f.wid_present {
		f1.w(f.wid);
		f2.w(f.wid);
	}
	if f.prec_present {
		f1.p(f.prec);
		f2.p(f.prec);
	}
	efmt := f1.E(a).str();
	ffmt := f2.F(a).str();
	// ffmt can return e in my bogus world; don't trim trailing 0s if so.
	f_is_e := false;
	for i := 0; i < len(ffmt); i++ {
		if ffmt[i] == 'e' {
			f_is_e = true;
			break;
		}
	}
	if !f_is_e {
		// strip trailing zeros
		l := len(ffmt);
		for ffmt[l-1]=='0' {
			l--;
		}
		ffmt = ffmt[0:l];
	}
	if len(efmt) < len(ffmt) {
		f.pad(efmt);
	} else {
		f.pad(ffmt);
	}
	f.clearflags();
	return f;
}

// float
func (x *Fmt) f(a float) *Fmt {
	return x.F(double(a))
}

// float
func (x *Fmt) e(a float) *Fmt {
	return x.E(double(a))
}

// float
func (x *Fmt) g(a float) *Fmt {
	return x.G(double(a))
}
