// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

import (
	"cmd/compile/internal/big"
	"cmd/internal/obj"
	"fmt"
	"math"
	"strings"
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

	return x
}

func mpgetflt32(a *Mpflt) float64 {
	x32, _ := a.Val.Float32()
	x := float64(x32)

	// check for overflow
	if math.IsInf(x, 0) && nsavederrors+nerrors == 0 {
		Yyerror("mpgetflt32 ovf")
	}

	return x
}

func Mpmovecflt(a *Mpflt, c float64) {
	if Mpdebug {
		fmt.Printf("\nconst %g", c)
	}

	a.Val.SetFloat64(c)

	if Mpdebug {
		fmt.Printf(" = %v\n", a)
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

	// The spec requires accepting exponents that fit in int32.
	// Don't accept much more than that.
	// Count digits in exponent and stop early if there are too many.
	if i := strings.Index(as, "e"); i >= 0 {
		i++
		if i < len(as) && (as[i] == '-' || as[i] == '+') {
			i++
		}
		for i < len(as) && as[i] == '0' {
			i++
		}
		// TODO(rsc): This should be > 10, because we're supposed
		// to accept any signed 32-bit int as an exponent.
		// But that's not working terribly well, so we deviate from the
		// spec in order to make sure that what we accept works.
		// We can remove this restriction once those larger exponents work.
		// See golang.org/issue/11326 and test/fixedbugs/issue11326*.go.
		if len(as)-i > 8 {
			Yyerror("malformed constant: %s (exponent too large)", as)
			a.Val.SetUint64(0)
			return
		}
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
		return
	}

	if f.IsInf() {
		Yyerror("constant too large: %s", as)
		a.Val.SetUint64(0)
		return
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
	if fvp.Val.Signbit() {
		sign = "-"
		f = new(big.Float).Abs(f)
	} else if flag&obj.FmtSign != 0 {
		sign = "+"
	}

	// Use fmt formatting if in float64 range (common case).
	if x, _ := f.Float64(); !math.IsInf(x, 0) {
		return fmt.Sprintf("%s%.6g", sign, x)
	}

	// Out of float64 range. Do approximate manual to decimal
	// conversion to avoid precise but possibly slow Float
	// formatting. The exponent is > 0 since a negative out-
	// of-range exponent would have underflowed and led to 0.
	// f = mant * 2**exp
	var mant big.Float
	exp := float64(f.MantExp(&mant)) // 0.5 <= mant < 1.0, exp > 0

	// approximate float64 mantissa m and decimal exponent d
	// f ~ m * 10**d
	m, _ := mant.Float64()            // 0.5 <= m < 1.0
	d := exp * (math.Ln2 / math.Ln10) // log_10(2)

	// adjust m for truncated (integer) decimal exponent e
	e := int64(d)
	m *= math.Pow(10, d-float64(e))
	for m >= 10 {
		m /= 10
		e++
	}

	return fmt.Sprintf("%s%.5fe+%d", sign, m, e)
}
