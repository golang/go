// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

import (
	"fmt"
	"math"
	"math/big"
)

// implements float arithmetic

const (
	// Maximum size in bits for Mpints before signalling
	// overflow and also mantissa precision for Mpflts.
	Mpprec = 512
	// Turn on for constant arithmetic debugging output.
	Mpdebug = false
)

// Mpflt represents a floating-point constant.
type Mpflt struct {
	Val big.Float
}

// Mpcplx represents a complex constant.
type Mpcplx struct {
	Real Mpflt
	Imag Mpflt
}

// Use newMpflt (not new(Mpflt)!) to get the correct default precision.
func newMpflt() *Mpflt {
	var a Mpflt
	a.Val.SetPrec(Mpprec)
	return &a
}

// Use newMpcmplx (not new(Mpcplx)!) to get the correct default precision.
func newMpcmplx() *Mpcplx {
	var a Mpcplx
	a.Real = *newMpflt()
	a.Imag = *newMpflt()
	return &a
}

func (a *Mpflt) SetInt(b *Mpint) {
	if b.checkOverflow(0) {
		// sign doesn't really matter but copy anyway
		a.Val.SetInf(b.Val.Sign() < 0)
		return
	}
	a.Val.SetInt(&b.Val)
}

func (a *Mpflt) Set(b *Mpflt) {
	a.Val.Set(&b.Val)
}

func (a *Mpflt) Add(b *Mpflt) {
	if Mpdebug {
		fmt.Printf("\n%v + %v", a, b)
	}

	a.Val.Add(&a.Val, &b.Val)

	if Mpdebug {
		fmt.Printf(" = %v\n\n", a)
	}
}

func (a *Mpflt) AddFloat64(c float64) {
	var b Mpflt

	b.SetFloat64(c)
	a.Add(&b)
}

func (a *Mpflt) Sub(b *Mpflt) {
	if Mpdebug {
		fmt.Printf("\n%v - %v", a, b)
	}

	a.Val.Sub(&a.Val, &b.Val)

	if Mpdebug {
		fmt.Printf(" = %v\n\n", a)
	}
}

func (a *Mpflt) Mul(b *Mpflt) {
	if Mpdebug {
		fmt.Printf("%v\n * %v\n", a, b)
	}

	a.Val.Mul(&a.Val, &b.Val)

	if Mpdebug {
		fmt.Printf(" = %v\n\n", a)
	}
}

func (a *Mpflt) MulFloat64(c float64) {
	var b Mpflt

	b.SetFloat64(c)
	a.Mul(&b)
}

func (a *Mpflt) Quo(b *Mpflt) {
	if Mpdebug {
		fmt.Printf("%v\n / %v\n", a, b)
	}

	a.Val.Quo(&a.Val, &b.Val)

	if Mpdebug {
		fmt.Printf(" = %v\n\n", a)
	}
}

func (a *Mpflt) Cmp(b *Mpflt) int {
	return a.Val.Cmp(&b.Val)
}

func (a *Mpflt) CmpFloat64(c float64) int {
	if c == 0 {
		return a.Val.Sign() // common case shortcut
	}
	return a.Val.Cmp(big.NewFloat(c))
}

func (a *Mpflt) Float64() float64 {
	x, _ := a.Val.Float64()

	// check for overflow
	if math.IsInf(x, 0) && nsavederrors+nerrors == 0 {
		Fatalf("ovf in Mpflt Float64")
	}

	return x + 0 // avoid -0 (should not be needed, but be conservative)
}

func (a *Mpflt) Float32() float64 {
	x32, _ := a.Val.Float32()
	x := float64(x32)

	// check for overflow
	if math.IsInf(x, 0) && nsavederrors+nerrors == 0 {
		Fatalf("ovf in Mpflt Float32")
	}

	return x + 0 // avoid -0 (should not be needed, but be conservative)
}

func (a *Mpflt) SetFloat64(c float64) {
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

func (a *Mpflt) Neg() {
	// avoid -0
	if a.Val.Sign() != 0 {
		a.Val.Neg(&a.Val)
	}
}

func (a *Mpflt) SetString(as string) {
	f, _, err := a.Val.Parse(as, 0)
	if err != nil {
		yyerror("malformed constant: %s (%v)", as, err)
		a.Val.SetFloat64(0)
		return
	}

	if f.IsInf() {
		yyerror("constant too large: %s", as)
		a.Val.SetFloat64(0)
		return
	}

	// -0 becomes 0
	if f.Sign() == 0 && f.Signbit() {
		a.Val.SetFloat64(0)
	}
}

func (f *Mpflt) String() string {
	return f.Val.Text('b', 0)
}

func (fvp *Mpflt) GoString() string {
	// determine sign
	sign := ""
	f := &fvp.Val
	if f.Sign() < 0 {
		sign = "-"
		f = new(big.Float).Abs(f)
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

// complex multiply v *= rv
//	(a, b) * (c, d) = (a*c - b*d, b*c + a*d)
func (v *Mpcplx) Mul(rv *Mpcplx) {
	var ac, ad, bc, bd Mpflt

	ac.Set(&v.Real)
	ac.Mul(&rv.Real) // ac

	bd.Set(&v.Imag)
	bd.Mul(&rv.Imag) // bd

	bc.Set(&v.Imag)
	bc.Mul(&rv.Real) // bc

	ad.Set(&v.Real)
	ad.Mul(&rv.Imag) // ad

	v.Real.Set(&ac)
	v.Real.Sub(&bd) // ac-bd

	v.Imag.Set(&bc)
	v.Imag.Add(&ad) // bc+ad
}

// complex divide v /= rv
//	(a, b) / (c, d) = ((a*c + b*d), (b*c - a*d))/(c*c + d*d)
func (v *Mpcplx) Div(rv *Mpcplx) bool {
	if rv.Real.CmpFloat64(0) == 0 && rv.Imag.CmpFloat64(0) == 0 {
		return false
	}

	var ac, ad, bc, bd, cc_plus_dd Mpflt

	cc_plus_dd.Set(&rv.Real)
	cc_plus_dd.Mul(&rv.Real) // cc

	ac.Set(&rv.Imag)
	ac.Mul(&rv.Imag)    // dd
	cc_plus_dd.Add(&ac) // cc+dd

	// We already checked that c and d are not both zero, but we can't
	// assume that c²+d² != 0 follows, because for tiny values of c
	// and/or d c²+d² can underflow to zero.  Check that c²+d² is
	// nonzero, return if it's not.
	if cc_plus_dd.CmpFloat64(0) == 0 {
		return false
	}

	ac.Set(&v.Real)
	ac.Mul(&rv.Real) // ac

	bd.Set(&v.Imag)
	bd.Mul(&rv.Imag) // bd

	bc.Set(&v.Imag)
	bc.Mul(&rv.Real) // bc

	ad.Set(&v.Real)
	ad.Mul(&rv.Imag) // ad

	v.Real.Set(&ac)
	v.Real.Add(&bd)         // ac+bd
	v.Real.Quo(&cc_plus_dd) // (ac+bd)/(cc+dd)

	v.Imag.Set(&bc)
	v.Imag.Sub(&ad)         // bc-ad
	v.Imag.Quo(&cc_plus_dd) // (bc+ad)/(cc+dd)

	return true
}

func (v *Mpcplx) String() string {
	return fmt.Sprintf("(%s+%si)", v.Real.String(), v.Imag.String())
}

func (v *Mpcplx) GoString() string {
	var re string
	sre := v.Real.CmpFloat64(0)
	if sre != 0 {
		re = v.Real.GoString()
	}

	var im string
	sim := v.Imag.CmpFloat64(0)
	if sim != 0 {
		im = v.Imag.GoString()
	}

	switch {
	case sre == 0 && sim == 0:
		return "0"
	case sre == 0:
		return im + "i"
	case sim == 0:
		return re
	case sim < 0:
		return fmt.Sprintf("(%s%si)", re, im)
	default:
		return fmt.Sprintf("(%s+%si)", re, im)
	}
}
