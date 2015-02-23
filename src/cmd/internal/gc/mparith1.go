// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

import (
	"cmd/internal/obj"
	"fmt"
	"math"
)

/// uses arithmetic

func mpcmpfixflt(a *Mpint, b *Mpflt) int {
	var c Mpflt

	buf := fmt.Sprintf("%v", Bconv(a, 0))
	mpatoflt(&c, buf)
	return mpcmpfltflt(&c, b)
}

func mpcmpfltfix(a *Mpflt, b *Mpint) int {
	var c Mpflt

	buf := fmt.Sprintf("%v", Bconv(b, 0))
	mpatoflt(&c, buf)
	return mpcmpfltflt(a, &c)
}

func Mpcmpfixfix(a *Mpint, b *Mpint) int {
	var c Mpint

	mpmovefixfix(&c, a)
	mpsubfixfix(&c, b)
	return mptestfix(&c)
}

func mpcmpfixc(b *Mpint, c int64) int {
	var c1 Mpint

	Mpmovecfix(&c1, c)
	return Mpcmpfixfix(b, &c1)
}

func mpcmpfltflt(a *Mpflt, b *Mpflt) int {
	var c Mpflt

	mpmovefltflt(&c, a)
	mpsubfltflt(&c, b)
	return mptestflt(&c)
}

func mpcmpfltc(b *Mpflt, c float64) int {
	var a Mpflt

	Mpmovecflt(&a, c)
	return mpcmpfltflt(b, &a)
}

func mpsubfixfix(a *Mpint, b *Mpint) {
	mpnegfix(a)
	mpaddfixfix(a, b, 0)
	mpnegfix(a)
}

func mpsubfltflt(a *Mpflt, b *Mpflt) {
	mpnegflt(a)
	mpaddfltflt(a, b)
	mpnegflt(a)
}

func mpaddcfix(a *Mpint, c int64) {
	var b Mpint

	Mpmovecfix(&b, c)
	mpaddfixfix(a, &b, 0)
}

func mpaddcflt(a *Mpflt, c float64) {
	var b Mpflt

	Mpmovecflt(&b, c)
	mpaddfltflt(a, &b)
}

func mpmulcfix(a *Mpint, c int64) {
	var b Mpint

	Mpmovecfix(&b, c)
	mpmulfixfix(a, &b)
}

func mpmulcflt(a *Mpflt, c float64) {
	var b Mpflt

	Mpmovecflt(&b, c)
	mpmulfltflt(a, &b)
}

func mpdivfixfix(a *Mpint, b *Mpint) {
	var q Mpint
	var r Mpint

	mpdivmodfixfix(&q, &r, a, b)
	mpmovefixfix(a, &q)
}

func mpmodfixfix(a *Mpint, b *Mpint) {
	var q Mpint
	var r Mpint

	mpdivmodfixfix(&q, &r, a, b)
	mpmovefixfix(a, &r)
}

func mpcomfix(a *Mpint) {
	var b Mpint

	Mpmovecfix(&b, 1)
	mpnegfix(a)
	mpsubfixfix(a, &b)
}

func Mpmovefixflt(a *Mpflt, b *Mpint) {
	a.Val = *b
	a.Exp = 0
	mpnorm(a)
}

// convert (truncate) b to a.
// return -1 (but still convert) if b was non-integer.
func mpexactfltfix(a *Mpint, b *Mpflt) int {
	*a = b.Val
	Mpshiftfix(a, int(b.Exp))
	if b.Exp < 0 {
		var f Mpflt
		f.Val = *a
		f.Exp = 0
		mpnorm(&f)
		if mpcmpfltflt(b, &f) != 0 {
			return -1
		}
	}

	return 0
}

func mpmovefltfix(a *Mpint, b *Mpflt) int {
	if mpexactfltfix(a, b) == 0 {
		return 0
	}

	// try rounding down a little
	f := *b

	f.Val.A[0] = 0
	if mpexactfltfix(a, &f) == 0 {
		return 0
	}

	// try rounding up a little
	for i := 1; i < Mpprec; i++ {
		f.Val.A[i]++
		if f.Val.A[i] != Mpbase {
			break
		}
		f.Val.A[i] = 0
	}

	mpnorm(&f)
	if mpexactfltfix(a, &f) == 0 {
		return 0
	}

	return -1
}

func mpmovefixfix(a *Mpint, b *Mpint) {
	*a = *b
}

func mpmovefltflt(a *Mpflt, b *Mpflt) {
	*a = *b
}

var tab = []float64{1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7}

func mppow10flt(a *Mpflt, p int) {
	if p < 0 {
		panic("abort")
	}
	if p < len(tab) {
		Mpmovecflt(a, tab[p])
		return
	}

	mppow10flt(a, p>>1)
	mpmulfltflt(a, a)
	if p&1 != 0 {
		mpmulcflt(a, 10)
	}
}

func mphextofix(a *Mpint, s string) {
	for s != "" && s[0] == '0' {
		s = s[1:]
	}

	// overflow
	if 4*len(s) > Mpscale*Mpprec {
		a.Ovf = 1
		return
	}

	end := len(s) - 1
	var c int8
	var d int
	var bit int
	for hexdigitp := end; hexdigitp >= 0; hexdigitp-- {
		c = int8(s[hexdigitp])
		if c >= '0' && c <= '9' {
			d = int(c) - '0'
		} else if c >= 'A' && c <= 'F' {
			d = int(c) - 'A' + 10
		} else {
			d = int(c) - 'a' + 10
		}

		bit = 4 * (end - hexdigitp)
		for d > 0 {
			if d&1 != 0 {
				a.A[bit/Mpscale] |= int(1) << uint(bit%Mpscale)
			}
			bit++
			d = d >> 1
		}
	}
}

//
// floating point input
// required syntax is [+-]d*[.]d*[e[+-]d*] or [+-]0xH*[e[+-]d*]
//
func mpatoflt(a *Mpflt, as string) {
	for as[0] == ' ' || as[0] == '\t' {
		as = as[1:]
	}

	/* determine base */
	s := as

	base := -1
	for base == -1 {
		if s == "" {
			base = 10
			break
		}
		c := s[0]
		s = s[1:]
		switch c {
		case '-',
			'+':
			break

		case '0':
			if s != "" && s[0] == 'x' {
				base = 16
			} else {
				base = 10
			}

		default:
			base = 10
		}
	}

	s = as
	dp := 0 /* digits after decimal point */
	f := 0  /* sign */
	ex := 0 /* exponent */
	eb := 0 /* binary point */

	Mpmovecflt(a, 0.0)
	var ef int
	var c int
	if base == 16 {
		start := ""
		var c int
		for {
			c, _ = intstarstringplusplus(s)
			if c == '-' {
				f = 1
				s = s[1:]
			} else if c == '+' {
				s = s[1:]
			} else if c == '0' && s[1] == 'x' {
				s = s[2:]
				start = s
			} else if (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F') {
				s = s[1:]
			} else {
				break
			}
		}

		if start == "" {
			Yyerror("malformed hex constant: %s", as)
			goto bad
		}

		mphextofix(&a.Val, start[:len(start)-len(s)])
		if a.Val.Ovf != 0 {
			Yyerror("constant too large: %s", as)
			goto bad
		}

		a.Exp = 0
		mpnorm(a)
	}

	for {
		c, s = intstarstringplusplus(s)
		switch c {
		default:
			Yyerror("malformed constant: %s (at %c)", as, c)
			goto bad

		case '-':
			f = 1
			fallthrough

		case ' ',
			'\t',
			'+':
			continue

		case '.':
			if base == 16 {
				Yyerror("decimal point in hex constant: %s", as)
				goto bad
			}

			dp = 1
			continue

		case '1',
			'2',
			'3',
			'4',
			'5',
			'6',
			'7',
			'8',
			'9',
			'0':
			mpmulcflt(a, 10)
			mpaddcflt(a, float64(c)-'0')
			if dp != 0 {
				dp++
			}
			continue

		case 'P',
			'p':
			eb = 1
			fallthrough

		case 'E',
			'e':
			ex = 0
			ef = 0
			for {
				c, s = intstarstringplusplus(s)
				if c == '+' || c == ' ' || c == '\t' {
					continue
				}
				if c == '-' {
					ef = 1
					continue
				}

				if c >= '0' && c <= '9' {
					ex = ex*10 + (c - '0')
					if ex > 1e8 {
						Yyerror("constant exponent out of range: %s", as)
						errorexit()
					}

					continue
				}

				break
			}

			if ef != 0 {
				ex = -ex
			}
			fallthrough

		case 0:
			break
		}

		break
	}

	if eb != 0 {
		if dp != 0 {
			Yyerror("decimal point and binary point in constant: %s", as)
			goto bad
		}

		mpsetexp(a, int(a.Exp)+ex)
		goto out
	}

	if dp != 0 {
		dp--
	}
	if mpcmpfltc(a, 0.0) != 0 {
		if ex >= dp {
			var b Mpflt
			mppow10flt(&b, ex-dp)
			mpmulfltflt(a, &b)
		} else {
			// 4 approximates least_upper_bound(log2(10)).
			if dp-ex >= 1<<(32-3) || int(int16(4*(dp-ex))) != 4*(dp-ex) {
				Mpmovecflt(a, 0.0)
			} else {
				var b Mpflt
				mppow10flt(&b, dp-ex)
				mpdivfltflt(a, &b)
			}
		}
	}

out:
	if f != 0 {
		mpnegflt(a)
	}
	return

bad:
	Mpmovecflt(a, 0.0)
}

//
// fixed point input
// required syntax is [+-][0[x]]d*
//
func mpatofix(a *Mpint, as string) {
	var c int
	var s0 string

	s := as
	f := 0
	Mpmovecfix(a, 0)

	c, s = intstarstringplusplus(s)
	switch c {
	case '-':
		f = 1
		fallthrough

	case '+':
		c, s = intstarstringplusplus(s)
		if c != '0' {
			break
		}
		fallthrough

	case '0':
		goto oct
	}

	for c != 0 {
		if c >= '0' && c <= '9' {
			mpmulcfix(a, 10)
			mpaddcfix(a, int64(c)-'0')
			c, s = intstarstringplusplus(s)
			continue
		}

		Yyerror("malformed decimal constant: %s", as)
		goto bad
	}

	goto out

oct:
	c, s = intstarstringplusplus(s)
	if c == 'x' || c == 'X' {
		goto hex
	}
	for c != 0 {
		if c >= '0' && c <= '7' {
			mpmulcfix(a, 8)
			mpaddcfix(a, int64(c)-'0')
			c, s = intstarstringplusplus(s)
			continue
		}

		Yyerror("malformed octal constant: %s", as)
		goto bad
	}

	goto out

hex:
	s0 = s
	c, _ = intstarstringplusplus(s)
	for c != 0 {
		if (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F') {
			s = s[1:]
			c, _ = intstarstringplusplus(s)
			continue
		}

		Yyerror("malformed hex constant: %s", as)
		goto bad
	}

	mphextofix(a, s0)
	if a.Ovf != 0 {
		Yyerror("constant too large: %s", as)
		goto bad
	}

out:
	if f != 0 {
		mpnegfix(a)
	}
	return

bad:
	Mpmovecfix(a, 0)
}

func Bconv(xval *Mpint, flag int) string {
	var q Mpint

	mpmovefixfix(&q, xval)
	f := 0
	if mptestfix(&q) < 0 {
		f = 1
		mpnegfix(&q)
	}

	var buf [500]byte
	p := len(buf)
	var r Mpint
	if flag&obj.FmtSharp != 0 /*untyped*/ {
		// Hexadecimal
		var sixteen Mpint
		Mpmovecfix(&sixteen, 16)

		var digit int
		for {
			mpdivmodfixfix(&q, &r, &q, &sixteen)
			digit = int(Mpgetfix(&r))
			if digit < 10 {
				p--
				buf[p] = byte(digit + '0')
			} else {
				p--
				buf[p] = byte(digit - 10 + 'A')
			}
			if mptestfix(&q) <= 0 {
				break
			}
		}

		p--
		buf[p] = 'x'
		p--
		buf[p] = '0'
	} else {
		// Decimal
		var ten Mpint
		Mpmovecfix(&ten, 10)

		for {
			mpdivmodfixfix(&q, &r, &q, &ten)
			p--
			buf[p] = byte(Mpgetfix(&r) + '0')
			if mptestfix(&q) <= 0 {
				break
			}
		}
	}

	if f != 0 {
		p--
		buf[p] = '-'
	}
	var fp string
	fp += string(buf[p:])
	return fp
}

func Fconv(fvp *Mpflt, flag int) string {
	if flag&obj.FmtSharp != 0 /*untyped*/ {
		// alternate form - decimal for error messages.
		// for well in range, convert to double and use print's %g
		exp := int(fvp.Exp) + sigfig(fvp)*Mpscale

		var fp string
		if -900 < exp && exp < 900 {
			d := mpgetflt(fvp)
			if d >= 0 && (flag&obj.FmtSign != 0 /*untyped*/) {
				fp += fmt.Sprintf("+")
			}
			fp += fmt.Sprintf("%.6g", d)
			return fp
		}

		// very out of range. compute decimal approximation by hand.
		// decimal exponent
		dexp := float64(fvp.Exp) * 0.301029995663981195 // log_10(2)
		exp = int(dexp)

		// decimal mantissa
		fv := *fvp

		fv.Val.Neg = 0
		fv.Exp = 0
		d := mpgetflt(&fv)
		d *= math.Pow(10, dexp-float64(exp))
		for d >= 9.99995 {
			d /= 10
			exp++
		}

		if fvp.Val.Neg != 0 {
			fp += fmt.Sprintf("-")
		} else if flag&obj.FmtSign != 0 /*untyped*/ {
			fp += fmt.Sprintf("+")
		}
		fp += fmt.Sprintf("%.5fe+%d", d, exp)
		return fp
	}

	var fv Mpflt
	var buf string
	if sigfig(fvp) == 0 {
		buf = fmt.Sprintf("0p+0")
		goto out
	}

	fv = *fvp

	for fv.Val.A[0] == 0 {
		Mpshiftfix(&fv.Val, -Mpscale)
		fv.Exp += Mpscale
	}

	for fv.Val.A[0]&1 == 0 {
		Mpshiftfix(&fv.Val, -1)
		fv.Exp += 1
	}

	if fv.Exp >= 0 {
		buf = fmt.Sprintf("%vp+%d", Bconv(&fv.Val, obj.FmtSharp), fv.Exp)
		goto out
	}

	buf = fmt.Sprintf("%vp-%d", Bconv(&fv.Val, obj.FmtSharp), -fv.Exp)

out:
	var fp string
	fp += buf
	return fp
}
