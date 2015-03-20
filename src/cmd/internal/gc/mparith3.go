// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

import (
	"fmt"
	"math"
)

func newMpflt() *Mpflt {
	var a Mpflt
	a.Val.SetPrec(Mpscale * Mpprec)
	return &a
}

/// implements float arihmetic

func mpaddfltflt(a *Mpflt, b *Mpflt) {
	if Mpdebug != 0 {
		fmt.Printf("\n%v + %v", Fconv(a, 0), Fconv(b, 0))
	}

	a.Val.Add(&a.Val, &b.Val)

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

func mpdivfltflt(a *Mpflt, b *Mpflt) {
	if Mpdebug != 0 {
		fmt.Printf("%v\n / %v\n", Fconv(a, 0), Fconv(b, 0))
	}

	a.Val.Quo(&a.Val, &b.Val)

	if Mpdebug != 0 {
		fmt.Printf(" = %v\n\n", Fconv(a, 0))
	}
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
