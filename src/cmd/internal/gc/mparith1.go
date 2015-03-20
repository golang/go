// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

import (
	"cmd/internal/gc/big"
	"cmd/internal/obj"
	"fmt"
)

func Mpcmpfixfix(a, b *Mpint) int {
	return a.Val.Cmp(&b.Val)
}

func mpcmpfixc(b *Mpint, c int64) int {
	return b.Val.Cmp(big.NewInt(c))
}

func mpcmpfltflt(a *Mpflt, b *Mpflt) int {
	return a.Val.Cmp(&b.Val)
}

func mpcmpfltc(b *Mpflt, c float64) int {
	var a Mpflt

	Mpmovecflt(&a, c)
	return mpcmpfltflt(b, &a)
}

func mpsubfixfix(a, b *Mpint) {
	a.Val.Sub(&a.Val, &b.Val)
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

func mpaddcflt(a *Mpflt, c float64) {
	var b Mpflt

	Mpmovecflt(&b, c)
	mpaddfltflt(a, &b)
}

func mpmulcflt(a *Mpflt, c float64) {
	var b Mpflt

	Mpmovecflt(&b, c)
	mpmulfltflt(a, &b)
}

func mpdivfixfix(a, b *Mpint) {
	a.Val.Quo(&a.Val, &b.Val)
}

func mpmodfixfix(a, b *Mpint) {
	a.Val.Rem(&a.Val, &b.Val)
}

func Mpmovefixflt(a *Mpflt, b *Mpint) {
	if b.Ovf {
		// sign doesn't really matter but copy anyway
		a.Val.SetInf(b.Val.Sign() < 0)
		return
	}
	a.Val.SetInt(&b.Val)
}

func mpmovefltfix(a *Mpint, b *Mpflt) int {
	if _, acc := b.Val.Int(&a.Val); acc == big.Exact {
		return 0
	}

	const delta = Mpscale // a reasonably small number of bits > 0
	var t big.Float
	t.SetPrec(Mpscale*Mpprec - delta)

	// try rounding down a little
	t.SetMode(big.ToZero)
	t.Set(&b.Val)
	if _, acc := t.Int(&a.Val); acc == big.Exact {
		return 0
	}

	// try rounding up a little
	t.SetMode(big.AwayFromZero)
	t.Set(&b.Val)
	if _, acc := t.Int(&a.Val); acc == big.Exact {
		return 0
	}

	return -1
}

func mpmovefixfix(a, b *Mpint) {
	a.Val.Set(&b.Val)
}

func mpmovefltflt(a *Mpflt, b *Mpflt) {
	a.Val.Set(&b.Val)
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

func Fconv(fvp *Mpflt, flag int) string {
	if flag&obj.FmtSharp != 0 {
		return fvp.Val.Format('g', 6)
	}
	return fvp.Val.Format('b', 0)
}
