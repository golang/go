// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

import (
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
	if Mpdebug != 0 {
		fmt.Printf("\n%v + %v", Fconv(a, 0), Fconv(b, 0))
	}

	a.Val.Add(&a.Val, &b.Val)

	if Mpdebug != 0 {
		fmt.Printf(" = %v\n\n", Fconv(a, 0))
	}
}

func mpaddcflt(a *Mpflt, c float64) {
	var b Mpflt

	Mpmovecflt(&b, c)
	mpaddfltflt(a, &b)
}

func mpsubfltflt(a *Mpflt, b *Mpflt) {
	if Mpdebug != 0 {
		fmt.Printf("\n%v - %v", Fconv(a, 0), Fconv(b, 0))
	}

	a.Val.Sub(&a.Val, &b.Val)

	if Mpdebug != 0 {
		fmt.Printf(" = %v\n\n", Fconv(a, 0))
	}
}

func mpmulfltflt(a *Mpflt, b *Mpflt) {
	if Mpdebug != 0 {
		fmt.Printf("%v\n * %v\n", Fconv(a, 0), Fconv(b, 0))
	}

	a.Val.Mul(&a.Val, &b.Val)

	if Mpdebug != 0 {
		fmt.Printf(" = %v\n\n", Fconv(a, 0))
	}
}

func mpmulcflt(a *Mpflt, c float64) {
	var b Mpflt

	Mpmovecflt(&b, c)
	mpmulfltflt(a, &b)
}

func mpdivfltflt(a *Mpflt, b *Mpflt) {
	if Mpdebug != 0 {
		fmt.Printf("%v\n / %v\n", Fconv(a, 0), Fconv(b, 0))
	}

	a.Val.Quo(&a.Val, &b.Val)

	if Mpdebug != 0 {
		fmt.Printf(" = %v\n\n", Fconv(a, 0))
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

func mpgetfltN(a *Mpflt, prec int, bias int) float64 {
	var x float64
	switch prec {
	case 53:
		x, _ = a.Val.Float64()
	case 24:
		// We should be using a.Val.Float32() here but that seems incorrect
		// for certain denormal values (all.bash fails). The current code
		// appears to work for all existing test cases, though there ought
		// to be issues with denormal numbers that are incorrectly rounded.
		// TODO(gri) replace with a.Val.Float32() once correctly working
		// See also: https://github.com/golang/go/issues/10321
		var t Mpflt
		t.Val.SetPrec(24).Set(&a.Val)
		x, _ = t.Val.Float64()
	default:
		panic("unreachable")
	}

	// check for overflow
	if math.IsInf(x, 0) && nsavederrors+nerrors == 0 {
		Yyerror("mpgetflt ovf")
	}

	return x
}

func mpgetflt(a *Mpflt) float64 {
	return mpgetfltN(a, 53, -1023)
}

func mpgetflt32(a *Mpflt) float64 {
	return mpgetfltN(a, 24, -127)
}

func Mpmovecflt(a *Mpflt, c float64) {
	if Mpdebug != 0 {
		fmt.Printf("\nconst %g", c)
	}

	a.Val.SetFloat64(c)

	if Mpdebug != 0 {
		fmt.Printf(" = %v\n", Fconv(a, 0))
	}
}

func mpnegflt(a *Mpflt) {
	a.Val.Neg(&a.Val)
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
		a.Val.SetUint64(0)
	}

	if f.IsInf() {
		Yyerror("constant too large: %s", as)
		a.Val.SetUint64(0)
	}
}

func Fconv(fvp *Mpflt, flag int) string {
	if flag&obj.FmtSharp != 0 {
		return fvp.Val.Format('g', 6)
	}
	return fvp.Val.Format('b', 0)
}
