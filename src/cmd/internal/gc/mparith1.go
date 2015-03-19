// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

import (
	"cmd/internal/obj"
	"fmt"
	"math"
	"math/big"
)

/// uses arithmetic

func mpcmpfixflt(a *Mpfix, b *Mpflt) int {
	var c Mpflt

	buf := _Bconv(a, 0)
	mpatoflt(&c, buf)
	return mpcmpfltflt(&c, b)
}

func mpcmpfltfix(a *Mpflt, b *Mpfix) int {
	var c Mpflt

	buf := _Bconv(b, 0)
	mpatoflt(&c, buf)
	return mpcmpfltflt(a, &c)
}

func Mpcmpfixfix(a, b *Mpint) int {
	return a.Val.Cmp(&b.Val)
}

func mpcmpfixc(b *Mpint, c int64) int {
	return b.Val.Cmp(big.NewInt(c))
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

func mpsubfixfix(a, b *Mpint) {
	a.Val.Sub(&a.Val, &b.Val)
}

func _mpsubfixfix(a *Mpfix, b *Mpfix) {
	_mpnegfix(a)
	_mpaddfixfix(a, b, 0)
	_mpnegfix(a)
}

func mpsubfltflt(a *Mpflt, b *Mpflt) {
	mpnegflt(a)
	mpaddfltflt(a, b)
	mpnegflt(a)
}

func mpaddcfix(a *Mpfix, c int64) {
	var b Mpfix

	_Mpmovecfix(&b, c)
	_mpaddfixfix(a, &b, 0)
}

func mpaddcflt(a *Mpflt, c float64) {
	var b Mpflt

	Mpmovecflt(&b, c)
	mpaddfltflt(a, &b)
}

func mpmulcfix(a *Mpfix, c int64) {
	var b Mpfix

	_Mpmovecfix(&b, c)
	_mpmulfixfix(a, &b)
}

func mpmulcflt(a *Mpflt, c float64) {
	var b Mpflt

	Mpmovecflt(&b, c)
	mpmulfltflt(a, &b)
}

func mpdivfixfix(a, b *Mpint) {
	a.Val.Quo(&a.Val, &b.Val)
}

func _mpdivfixfix(a *Mpfix, b *Mpfix) {
	var q Mpfix
	var r Mpfix

	mpdivmodfixfix(&q, &r, a, b)
	_mpmovefixfix(a, &q)
}

func mpmodfixfix(a, b *Mpint) {
	a.Val.Rem(&a.Val, &b.Val)
}

func _mpmodfixfix(a *Mpfix, b *Mpfix) {
	var q Mpfix
	var r Mpfix

	mpdivmodfixfix(&q, &r, a, b)
	_mpmovefixfix(a, &r)
}

func mpcomfix(a *Mpfix) {
	var b Mpfix

	_Mpmovecfix(&b, 1)
	_mpnegfix(a)
	_mpsubfixfix(a, &b)
}

// *a = Mpfix(*b)
func mpmoveintfix(a *Mpfix, b *Mpint) {
	if b.Ovf {
		_Mpmovecfix(a, 0)
		a.Ovf = 1
		return
	}

	var bb big.Int
	bb.Abs(&b.Val)
	i := 0
	for ; i < Mpprec && bb.Sign() != 0; i++ {
		// depends on (unspecified) behavior of Int.Uint64
		a.A[i] = int(bb.Uint64() & Mpmask)
		bb.Rsh(&bb, Mpscale)
	}

	if bb.Sign() != 0 {
		// MPint overflows
		_Mpmovecfix(a, 0)
		a.Ovf = 1
		return
	}

	for ; i < Mpprec; i++ {
		a.A[i] = 0
	}

	a.Neg = 0
	if b.Val.Sign() < 0 {
		a.Neg = 1
	}
	a.Ovf = 0

	// leave for debugging
	// println("mpmoveintfix:", b.Val.String(), "->", _Bconv(a, 0))
}

// *a = big.Int(*b)
func mpmovefixint(a *Mpint, b *Mpfix) {
	if b.Ovf != 0 {
		mpsetovf(a)
		return
	}

	i := Mpprec - 1
	for ; i >= 0 && b.A[i] == 0; i-- {
	}

	a.Val.SetUint64(0)
	var x big.Int
	for ; i >= 0; i-- {
		a.Val.Lsh(&a.Val, Mpscale)
		a.Val.Or(&a.Val, x.SetUint64(uint64(b.A[i]&Mpmask)))
	}

	if b.Neg != 0 {
		a.Val.Neg(&a.Val)
	}
	a.Ovf = false

	// leave for debugging
	// println("mpmovefixint:", _Bconv(b, 0), "->", a.Val.String())
}

func Mpmovefixflt(a *Mpflt, b *Mpint) {
	mpmoveintfix(&a.Val, b) // a.Val = *b
	a.Exp = 0
	mpnorm(a)
}

func _Mpmovefixflt(a *Mpflt, b *Mpfix) {
	a.Val = *b
	a.Exp = 0
	mpnorm(a)
}

// convert (truncate) b to a.
// return -1 (but still convert) if b was non-integer.
func mpexactfltfix(a *Mpint, b *Mpflt) int {
	mpmovefixint(a, &b.Val) // *a = b.Val
	Mpshiftfix(a, int(b.Exp))
	if b.Exp < 0 {
		var f Mpflt
		mpmoveintfix(&f.Val, a) // f.Val = *a
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

func mpmovefixfix(a, b *Mpint) {
	a.Val.Set(&b.Val)
}

func _mpmovefixfix(a *Mpfix, b *Mpfix) {
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

func mphextofix(a *Mpfix, s string) {
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

func mpatofix(a *Mpint, as string) {
	_, ok := a.Val.SetString(as, 0)
	if !ok {
		// required syntax is [+-][0[x]]d*
		// At the moment we lose precise error cause;
		// the old code distinguished between:
		// - malformed hex constant
		// - malformed octal constant
		// - malformed decimal constant
		// TODO(gri) use different conversion function
		Yyerror("malformed integer constant: %s", as)
		a.Val.SetUint64(0)
		return
	}
	if mptestovf(a, 0) {
		Yyerror("constant too large: %s", as)
	}
}

func Bconv(xval *Mpint, flag int) string {
	if flag&obj.FmtSharp != 0 {
		return fmt.Sprintf("%#x", &xval.Val)
	}
	return xval.Val.String()
}

func _Bconv(xval *Mpfix, flag int) string {
	var q Mpfix

	_mpmovefixfix(&q, xval)
	f := 0
	if mptestfix(&q) < 0 {
		f = 1
		_mpnegfix(&q)
	}

	var buf [500]byte
	p := len(buf)
	var r Mpfix
	if flag&obj.FmtSharp != 0 /*untyped*/ {
		// Hexadecimal
		var sixteen Mpfix
		_Mpmovecfix(&sixteen, 16)

		var digit int
		for {
			mpdivmodfixfix(&q, &r, &q, &sixteen)
			digit = int(_Mpgetfix(&r))
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
		var ten Mpfix
		_Mpmovecfix(&ten, 10)

		for {
			mpdivmodfixfix(&q, &r, &q, &ten)
			p--
			buf[p] = byte(_Mpgetfix(&r) + '0')
			if mptestfix(&q) <= 0 {
				break
			}
		}
	}

	if f != 0 {
		p--
		buf[p] = '-'
	}

	return string(buf[p:])
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
				fp += "+"
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
			fp += "-"
		} else if flag&obj.FmtSign != 0 /*untyped*/ {
			fp += "+"
		}
		fp += fmt.Sprintf("%.5fe+%d", d, exp)
		return fp
	}

	var fv Mpflt
	var buf string
	if sigfig(fvp) == 0 {
		buf = "0p+0"
		goto out
	}

	fv = *fvp

	for fv.Val.A[0] == 0 {
		_Mpshiftfix(&fv.Val, -Mpscale)
		fv.Exp += Mpscale
	}

	for fv.Val.A[0]&1 == 0 {
		_Mpshiftfix(&fv.Val, -1)
		fv.Exp += 1
	}

	if fv.Exp >= 0 {
		buf = fmt.Sprintf("%vp+%d", _Bconv(&fv.Val, obj.FmtSharp), fv.Exp)
		goto out
	}

	buf = fmt.Sprintf("%vp-%d", _Bconv(&fv.Val, obj.FmtSharp), -fv.Exp)

out:
	var fp string
	fp += buf
	return fp
}
