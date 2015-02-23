// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

import (
	"fmt"
	"math"
)

/*
 * returns the leading non-zero
 * word of the number
 */
func sigfig(a *Mpflt) int {
	var i int

	for i = Mpprec - 1; i >= 0; i-- {
		if a.Val.A[i] != 0 {
			break
		}
	}

	//print("sigfig %d %d\n", i-z+1, z);
	return i + 1
}

/*
 * sets the exponent.
 * a too large exponent is an error.
 * a too small exponent rounds the number to zero.
 */
func mpsetexp(a *Mpflt, exp int) {
	if int(int16(exp)) != exp {
		if exp > 0 {
			Yyerror("float constant is too large")
			a.Exp = 0x7fff
		} else {
			Mpmovecflt(a, 0)
		}
	} else {
		a.Exp = int16(exp)
	}
}

/*
 * shifts the leading non-zero
 * word of the number to Mpnorm
 */
func mpnorm(a *Mpflt) {
	os := sigfig(a)
	if os == 0 {
		// zero
		a.Exp = 0

		a.Val.Neg = 0
		return
	}

	// this will normalize to the nearest word
	x := a.Val.A[os-1]

	s := (Mpnorm - os) * Mpscale

	// further normalize to the nearest bit
	for {
		x <<= 1
		if x&Mpbase != 0 {
			break
		}
		s++
		if x == 0 {
			// this error comes from trying to
			// convert an Inf or something
			// where the initial x=0x80000000
			s = (Mpnorm - os) * Mpscale

			break
		}
	}

	Mpshiftfix(&a.Val, s)
	mpsetexp(a, int(a.Exp)-s)
}

/// implements float arihmetic

func mpaddfltflt(a *Mpflt, b *Mpflt) {
	if Mpdebug != 0 /*TypeKind(100016)*/ {
		fmt.Printf("\n%v + %v", Fconv(a, 0), Fconv(b, 0))
	}

	sa := sigfig(a)
	var s int
	var sb int
	if sa == 0 {
		mpmovefltflt(a, b)
		goto out
	}

	sb = sigfig(b)
	if sb == 0 {
		goto out
	}

	s = int(a.Exp) - int(b.Exp)
	if s > 0 {
		// a is larger, shift b right
		var c Mpflt
		mpmovefltflt(&c, b)

		Mpshiftfix(&c.Val, -s)
		mpaddfixfix(&a.Val, &c.Val, 0)
		goto out
	}

	if s < 0 {
		// b is larger, shift a right
		Mpshiftfix(&a.Val, s)

		mpsetexp(a, int(a.Exp)-s)
		mpaddfixfix(&a.Val, &b.Val, 0)
		goto out
	}

	mpaddfixfix(&a.Val, &b.Val, 0)

out:
	mpnorm(a)
	if Mpdebug != 0 /*TypeKind(100016)*/ {
		fmt.Printf(" = %v\n\n", Fconv(a, 0))
	}
}

func mpmulfltflt(a *Mpflt, b *Mpflt) {
	if Mpdebug != 0 /*TypeKind(100016)*/ {
		fmt.Printf("%v\n * %v\n", Fconv(a, 0), Fconv(b, 0))
	}

	sa := sigfig(a)
	if sa == 0 {
		// zero
		a.Exp = 0

		a.Val.Neg = 0
		return
	}

	sb := sigfig(b)
	if sb == 0 {
		// zero
		mpmovefltflt(a, b)

		return
	}

	mpmulfract(&a.Val, &b.Val)
	mpsetexp(a, (int(a.Exp)+int(b.Exp))+Mpscale*Mpprec-Mpscale-1)

	mpnorm(a)
	if Mpdebug != 0 /*TypeKind(100016)*/ {
		fmt.Printf(" = %v\n\n", Fconv(a, 0))
	}
}

func mpdivfltflt(a *Mpflt, b *Mpflt) {
	if Mpdebug != 0 /*TypeKind(100016)*/ {
		fmt.Printf("%v\n / %v\n", Fconv(a, 0), Fconv(b, 0))
	}

	sb := sigfig(b)
	if sb == 0 {
		// zero and ovfl
		a.Exp = 0

		a.Val.Neg = 0
		a.Val.Ovf = 1
		Yyerror("constant division by zero")
		return
	}

	sa := sigfig(a)
	if sa == 0 {
		// zero
		a.Exp = 0

		a.Val.Neg = 0
		return
	}

	// adjust b to top
	var c Mpflt
	mpmovefltflt(&c, b)

	Mpshiftfix(&c.Val, Mpscale)

	// divide
	mpdivfract(&a.Val, &c.Val)

	mpsetexp(a, (int(a.Exp)-int(c.Exp))-Mpscale*(Mpprec-1)+1)

	mpnorm(a)
	if Mpdebug != 0 /*TypeKind(100016)*/ {
		fmt.Printf(" = %v\n\n", Fconv(a, 0))
	}
}

func mpgetfltN(a *Mpflt, prec int, bias int) float64 {
	if a.Val.Ovf != 0 && nsavederrors+nerrors == 0 {
		Yyerror("mpgetflt ovf")
	}

	s := sigfig(a)
	if s == 0 {
		return 0
	}

	if s != Mpnorm {
		Yyerror("mpgetflt norm")
		mpnorm(a)
	}

	for a.Val.A[Mpnorm-1]&Mpsign == 0 {
		Mpshiftfix(&a.Val, 1)
		mpsetexp(a, int(a.Exp)-1) // can set 'a' to zero
		s = sigfig(a)
		if s == 0 {
			return 0
		}
	}

	// pick up the mantissa, a rounding bit, and a tie-breaking bit in a uvlong
	s = prec + 2

	v := uint64(0)
	var i int
	for i = Mpnorm - 1; s >= Mpscale; i-- {
		v = v<<Mpscale | uint64(a.Val.A[i])
		s -= Mpscale
	}

	if s > 0 {
		v = v<<uint(s) | uint64(a.Val.A[i])>>uint(Mpscale-s)
		if a.Val.A[i]&((1<<uint(Mpscale-s))-1) != 0 {
			v |= 1
		}
		i--
	}

	for ; i >= 0; i-- {
		if a.Val.A[i] != 0 {
			v |= 1
		}
	}

	// gradual underflow
	e := Mpnorm*Mpscale + int(a.Exp) - prec

	minexp := bias + 1 - prec + 1
	if e < minexp {
		s := minexp - e
		if s > prec+1 {
			s = prec + 1
		}
		if v&((1<<uint(s))-1) != 0 {
			v |= 1 << uint(s)
		}
		v >>= uint(s)
		e = minexp
	}

	// round to even
	v |= (v & 4) >> 2

	v += v & 1
	v >>= 2

	f := float64(v)
	f = math.Ldexp(f, e)

	if a.Val.Neg != 0 {
		f = -f
	}

	return f
}

func mpgetflt(a *Mpflt) float64 {
	return mpgetfltN(a, 53, -1023)
}

func mpgetflt32(a *Mpflt) float64 {
	return mpgetfltN(a, 24, -127)
}

func Mpmovecflt(a *Mpflt, c float64) {
	if Mpdebug != 0 /*TypeKind(100016)*/ {
		fmt.Printf("\nconst %g", c)
	}
	Mpmovecfix(&a.Val, 0)
	a.Exp = 0
	var f float64
	var l int
	var i int
	if c == 0 {
		goto out
	}
	if c < 0 {
		a.Val.Neg = 1
		c = -c
	}

	f, i = math.Frexp(c)
	a.Exp = int16(i)

	for i := 0; i < 10; i++ {
		f = f * Mpbase
		l = int(math.Floor(f))
		f = f - float64(l)
		a.Exp -= Mpscale
		a.Val.A[0] = l
		if f == 0 {
			break
		}
		Mpshiftfix(&a.Val, Mpscale)
	}

out:
	mpnorm(a)
	if Mpdebug != 0 /*TypeKind(100016)*/ {
		fmt.Printf(" = %v\n", Fconv(a, 0))
	}
}

func mpnegflt(a *Mpflt) {
	a.Val.Neg ^= 1
}

func mptestflt(a *Mpflt) int {
	if Mpdebug != 0 /*TypeKind(100016)*/ {
		fmt.Printf("\n%v?", Fconv(a, 0))
	}
	s := sigfig(a)
	if s != 0 {
		s = +1
		if a.Val.Neg != 0 {
			s = -1
		}
	}

	if Mpdebug != 0 /*TypeKind(100016)*/ {
		fmt.Printf(" = %d\n", s)
	}
	return s
}
