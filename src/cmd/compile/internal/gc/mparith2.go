// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

import (
	"cmd/compile/internal/big"
	"cmd/internal/obj"
	"fmt"
)

/// implements fix arithmetic

func mpsetovf(a *Mpint) {
	a.Val.SetUint64(1) // avoid spurious div-zero errors
	a.Ovf = true
}

func mptestovf(a *Mpint, extra int) bool {
	// We don't need to be precise here, any reasonable upper limit would do.
	// For now, use existing limit so we pass all the tests unchanged.
	if a.Val.BitLen()+extra > Mpprec {
		mpsetovf(a)
	}
	return a.Ovf
}

func mpmovefixfix(a, b *Mpint) {
	a.Val.Set(&b.Val)
}

func mpmovefltfix(a *Mpint, b *Mpflt) int {
	// avoid converting huge floating-point numbers to integers
	// (2*Mpprec is large enough to permit all tests to pass)
	if b.Val.MantExp(nil) > 2*Mpprec {
		return -1
	}

	if _, acc := b.Val.Int(&a.Val); acc == big.Exact {
		return 0
	}

	const delta = 16 // a reasonably small number of bits > 0
	var t big.Float
	t.SetPrec(Mpprec - delta)

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

func mpaddfixfix(a, b *Mpint, quiet int) {
	if a.Ovf || b.Ovf {
		if nsavederrors+nerrors == 0 {
			Yyerror("ovf in mpaddfixfix")
		}
		mpsetovf(a)
		return
	}

	a.Val.Add(&a.Val, &b.Val)

	if mptestovf(a, 0) && quiet == 0 {
		Yyerror("constant addition overflow")
	}
}

func mpsubfixfix(a, b *Mpint) {
	if a.Ovf || b.Ovf {
		if nsavederrors+nerrors == 0 {
			Yyerror("ovf in mpsubfixfix")
		}
		mpsetovf(a)
		return
	}

	a.Val.Sub(&a.Val, &b.Val)

	if mptestovf(a, 0) {
		Yyerror("constant subtraction overflow")
	}
}

func mpmulfixfix(a, b *Mpint) {
	if a.Ovf || b.Ovf {
		if nsavederrors+nerrors == 0 {
			Yyerror("ovf in mpmulfixfix")
		}
		mpsetovf(a)
		return
	}

	a.Val.Mul(&a.Val, &b.Val)

	if mptestovf(a, 0) {
		Yyerror("constant multiplication overflow")
	}
}

func mpdivfixfix(a, b *Mpint) {
	if a.Ovf || b.Ovf {
		if nsavederrors+nerrors == 0 {
			Yyerror("ovf in mpdivfixfix")
		}
		mpsetovf(a)
		return
	}

	a.Val.Quo(&a.Val, &b.Val)

	if mptestovf(a, 0) {
		// can only happen for div-0 which should be checked elsewhere
		Yyerror("constant division overflow")
	}
}

func mpmodfixfix(a, b *Mpint) {
	if a.Ovf || b.Ovf {
		if nsavederrors+nerrors == 0 {
			Yyerror("ovf in mpmodfixfix")
		}
		mpsetovf(a)
		return
	}

	a.Val.Rem(&a.Val, &b.Val)

	if mptestovf(a, 0) {
		// should never happen
		Yyerror("constant modulo overflow")
	}
}

func mporfixfix(a, b *Mpint) {
	if a.Ovf || b.Ovf {
		if nsavederrors+nerrors == 0 {
			Yyerror("ovf in mporfixfix")
		}
		mpsetovf(a)
		return
	}

	a.Val.Or(&a.Val, &b.Val)
}

func mpandfixfix(a, b *Mpint) {
	if a.Ovf || b.Ovf {
		if nsavederrors+nerrors == 0 {
			Yyerror("ovf in mpandfixfix")
		}
		mpsetovf(a)
		return
	}

	a.Val.And(&a.Val, &b.Val)
}

func mpandnotfixfix(a, b *Mpint) {
	if a.Ovf || b.Ovf {
		if nsavederrors+nerrors == 0 {
			Yyerror("ovf in mpandnotfixfix")
		}
		mpsetovf(a)
		return
	}

	a.Val.AndNot(&a.Val, &b.Val)
}

func mpxorfixfix(a, b *Mpint) {
	if a.Ovf || b.Ovf {
		if nsavederrors+nerrors == 0 {
			Yyerror("ovf in mpxorfixfix")
		}
		mpsetovf(a)
		return
	}

	a.Val.Xor(&a.Val, &b.Val)
}

// shift left by s (or right by -s)
func Mpshiftfix(a *Mpint, s int) {
	switch {
	case s > 0:
		if mptestovf(a, s) {
			Yyerror("constant shift overflow")
			return
		}
		a.Val.Lsh(&a.Val, uint(s))
	case s < 0:
		a.Val.Rsh(&a.Val, uint(-s))
	}
}

func mplshfixfix(a, b *Mpint) {
	if a.Ovf || b.Ovf {
		if nsavederrors+nerrors == 0 {
			Yyerror("ovf in mplshfixfix")
		}
		mpsetovf(a)
		return
	}

	s := Mpgetfix(b)
	if s < 0 || s >= Mpprec {
		Yyerror("stupid shift: %d", s)
		Mpmovecfix(a, 0)
		return
	}

	Mpshiftfix(a, int(s))
}

func mprshfixfix(a, b *Mpint) {
	if a.Ovf || b.Ovf {
		if nsavederrors+nerrors == 0 {
			Yyerror("ovf in mprshfixfix")
		}
		mpsetovf(a)
		return
	}

	s := Mpgetfix(b)
	if s < 0 {
		Yyerror("stupid shift: %d", s)
		if a.Val.Sign() < 0 {
			Mpmovecfix(a, -1)
		} else {
			Mpmovecfix(a, 0)
		}
		return
	}

	Mpshiftfix(a, int(-s))
}

func Mpcmpfixfix(a, b *Mpint) int {
	return a.Val.Cmp(&b.Val)
}

func mpcmpfixc(b *Mpint, c int64) int {
	return b.Val.Cmp(big.NewInt(c))
}

func mpnegfix(a *Mpint) {
	a.Val.Neg(&a.Val)
}

func Mpgetfix(a *Mpint) int64 {
	if a.Ovf {
		if nsavederrors+nerrors == 0 {
			Yyerror("constant overflow")
		}
		return 0
	}

	return a.Val.Int64()
}

func Mpmovecfix(a *Mpint, c int64) {
	a.Val.SetInt64(c)
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

func (x *Mpint) String() string {
	return Bconv(x, 0)
}

func Bconv(xval *Mpint, flag int) string {
	if flag&obj.FmtSharp != 0 {
		return fmt.Sprintf("%#x", &xval.Val)
	}
	return xval.Val.String()
}
