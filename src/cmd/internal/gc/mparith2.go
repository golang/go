// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

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

/// implements fix arithmetic

func mpsetovf(a *Mpint) {
	a.Val.SetUint64(0)
	a.Ovf = true
}

func mptestovf(a *Mpint, extra int) bool {
	// We don't need to be precise here, any reasonable upper limit would do.
	// For now, use existing limit so we pass all the tests unchanged.
	const limit = Mpscale * Mpprec
	if a.Val.BitLen()+extra > limit {
		mpsetovf(a)
	}
	return a.Ovf
}

func mpaddfixfix(a, b *Mpint, quiet int) {
	if a.Ovf || b.Ovf {
		if nsavederrors+nerrors == 0 {
			Yyerror("ovf in mpaddxx")
		}
		mpsetovf(a)
		return
	}

	a.Val.Add(&a.Val, &b.Val)

	if mptestovf(a, 0) && quiet == 0 {
		Yyerror("constant addition overflow")
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

func mplshfixfix(a, b *Mpint) {
	if a.Ovf || b.Ovf {
		if nsavederrors+nerrors == 0 {
			Yyerror("ovf in mplshfixfix")
		}
		mpsetovf(a)
		return
	}

	s := Mpgetfix(b)
	if s < 0 || s >= Mpprec*Mpscale {
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
	if s < 0 || s >= Mpprec*Mpscale {
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
