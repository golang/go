// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

import (
	"cmd/compile/internal/big"
	"cmd/internal/obj"
	"fmt"
	"math"
)

/// implements float arihmetic

func newMpflt() *Mpflt {
	var a Mpflt
	a.Val.SetPrec(Mpprec)
	return &a
}

func Mpmovefixflt(a *Mpflt, b *Mpint) {
	if b.Ovf {
		// sign doesn't really matter but copy anyway
		a.Val.SetInf(b.Val.Sign() < 0)
		return
	}
	a.Val.SetInt(&b.Val)
}

func mpmovefltflt(a *Mpflt, b *Mpflt) {
	a.Val.Set(&b.Val)
}

func mpaddfltflt(a *Mpflt, b *Mpflt) {
	if Mpdebug {
		fmt.Printf("\n%v + %v", a, b)
	}

	a.Val.Add(&a.Val, &b.Val)

	if Mpdebug {
		fmt.Printf(" = %v\n\n", a)
	}
}

func mpaddcflt(a *Mpflt, c float64) {
	var b Mpflt

	Mpmovecflt(&b, c)
	mpaddfltflt(a, &b)
}

func mpsubfltflt(a *Mpflt, b *Mpflt) {
	if Mpdebug {
		fmt.Printf("\n%v - %v", a, b)
	}

	a.Val.Sub(&a.Val, &b.Val)

	if Mpdebug {
		fmt.Printf(" = %v\n\n", a)
	}
}

func mpmulfltflt(a *Mpflt, b *Mpflt) {
	if Mpdebug {
		fmt.Printf("%v\n * %v\n", a, b)
	}

	a.Val.Mul(&a.Val, &b.Val)

	if Mpdebug {
		fmt.Printf(" = %v\n\n", a)
	}
}

func mpmulcflt(a *Mpflt, c float64) {
	var b Mpflt

	Mpmovecflt(&b, c)
	mpmulfltflt(a, &b)
}

func mpdivfltflt(a *Mpflt, b *Mpflt) {
	if Mpdebug {
		fmt.Printf("%v\n / %v\n", a, b)
	}

	a.Val.Quo(&a.Val, &b.Val)

	if Mpdebug {
		fmt.Printf(" = %v\n\n", a)
	}
}

func mpcmpfltflt(a *Mpflt, b *Mpflt) int {
	return a.Val.Cmp(&b.Val)
}

func mpcmpfltc(b *Mpflt, c float64) int {
	var a Mpflt

	Mpmovecflt(&a, c)
	return mpcmpfltflt(b, &a)
}

func mpgetflt(a *Mpflt) float64 {
	x, _ := a.Val.Float64()

	// check for overflow
	if math.IsInf(x, 0) && nsavederrors+nerrors == 0 {
		Yyerror("mpgetflt ovf")
	}

	return x + 0 // avoid -0 (should not be needed, but be conservative)
}

func mpgetflt32(a *Mpflt) float64 {
	x32, _ := a.Val.Float32()
	x := float64(x32)

	// check for overflow
	if math.IsInf(x, 0) && nsavederrors+nerrors == 0 {
		Yyerror("mpgetflt32 ovf")
	}

	return x + 0 // avoid -0 (should not be needed, but be conservative)
}

func Mpmovecflt(a *Mpflt, c float64) {
	if Mpdebug {
		fmt.Printf("\nconst %g", c)
	}

	// convert -0 to 0
	if c == 0 {
		c = 0
	}
	a.Val.SetFloat64(c)

	if Mpdebug {
		fmt.Printf(" = %v\n", a)
	}
}

func mpnegflt(a *Mpflt) {
	// avoid -0
	if a.Val.Sign() != 0 {
		a.Val.Neg(&a.Val)
	}
}

//
// floating point input
// required syntax is [+-]d*[.]d*[e[+-]d*] or [+-]0xH*[e[+-]d*]
//
func mpatoflt(a *Mpflt, as string) {
	for len(as) > 0 && (as[0] == ' ' || as[0] == '\t') {
		as = as[1:]
	}

	f, ok := a.Val.SetString(as)
	if !ok {
		// At the moment we lose precise error cause;
		// the old code additionally distinguished between:
		// - malformed hex constant
		// - decimal point in hex constant
		// - constant exponent out of range
		// - decimal point and binary point in constant
		// TODO(gri) use different conversion function or check separately
		Yyerror("malformed constant: %s", as)
		a.Val.SetFloat64(0)
		return
	}

	if f.IsInf() {
		Yyerror("constant too large: %s", as)
		a.Val.SetFloat64(0)
		return
	}

	// -0 becomes 0
	if f.Sign() == 0 && f.Signbit() {
		a.Val.SetFloat64(0)
	}
}

func (f *Mpflt) String() string {
	return Fconv(f, 0)
}

func Fconv(fvp *Mpflt, flag int) string {
	if flag&obj.FmtSharp == 0 {
		return fvp.Val.Text('b', 0)
	}

	// use decimal format for error messages

	// determine sign
	f := &fvp.Val
	var sign string
	if f.Sign() < 0 {
		sign = "-"
		f = new(big.Float).Abs(f)
	} else if flag&obj.FmtSign != 0 {
		sign = "+"
	}

	// Don't try to convert infinities (will not terminate).
	if f.IsInf() {
		return sign + "Inf"
	}

	// Use exact fmt formatting if in float64 range (common case):
	// proceed if f doesn't underflow to 0 or overflow to inf.
	if x, _ := f.Float64(); f.Sign() == 0 == (x == 0) && !math.IsInf(x, 0) {
		return fmt.Sprintf("%s%.6g", sign, x)
	}

	// Out of float64 range. Do approximate manual to decimal
	// conversion to avoid precise but possibly slow Float
	// formatting.
	// f = mant * 2**exp
	var mant big.Float
	exp := f.MantExp(&mant) // 0.5 <= mant < 1.0

	// approximate float64 mantissa m and decimal exponent d
	// f ~ m * 10**d
	m, _ := mant.Float64()                     // 0.5 <= m < 1.0
	d := float64(exp) * (math.Ln2 / math.Ln10) // log_10(2)

	// adjust m for truncated (integer) decimal exponent e
	e := int64(d)
	m *= math.Pow(10, d-float64(e))

	// ensure 1 <= m < 10
	switch {
	case m < 1-0.5e-6:
		// The %.6g format below rounds m to 5 digits after the
		// decimal point. Make sure that m*10 < 10 even after
		// rounding up: m*10 + 0.5e-5 < 10 => m < 1 - 0.5e6.
		m *= 10
		e--
	case m >= 10:
		m /= 10
		e++
	}

	return fmt.Sprintf("%s%.6ge%+d", sign, m, e)
}
