// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

//
// return the significant
// words of the argument
//
func mplen(a *Mpint) int {
	n := -1
	for i := 0; i < Mpprec; i++ {
		if a.A[i] != 0 {
			n = i
		}
	}

	return n + 1
}

//
// left shift mpint by one
// ignores sign
//
func mplsh(a *Mpint, quiet int) {
	var x int

	c := 0
	for i := 0; i < Mpprec; i++ {
		x = (a.A[i] << 1) + c
		c = 0
		if x >= Mpbase {
			x -= Mpbase
			c = 1
		}

		a.A[i] = x
	}

	a.Ovf = uint8(c)
	if a.Ovf != 0 && quiet == 0 {
		Yyerror("constant shift overflow")
	}
}

//
// left shift mpint by Mpscale
// ignores sign
//
func mplshw(a *Mpint, quiet int) {
	i := Mpprec - 1
	if a.A[i] != 0 {
		a.Ovf = 1
		if quiet == 0 {
			Yyerror("constant shift overflow")
		}
	}

	for ; i > 0; i-- {
		a.A[i] = a.A[i-1]
	}
	a.A[i] = 0
}

//
// right shift mpint by one
// ignores sign and overflow
//
func mprsh(a *Mpint) {
	var x int

	c := 0
	lo := a.A[0] & 1
	for i := Mpprec - 1; i >= 0; i-- {
		x = a.A[i]
		a.A[i] = (x + c) >> 1
		c = 0
		if x&1 != 0 {
			c = Mpbase
		}
	}

	if a.Neg != 0 && lo != 0 {
		mpaddcfix(a, -1)
	}
}

//
// right shift mpint by Mpscale
// ignores sign and overflow
//
func mprshw(a *Mpint) {
	var i int

	lo := a.A[0]
	for i = 0; i < Mpprec-1; i++ {
		a.A[i] = a.A[i+1]
	}

	a.A[i] = 0
	if a.Neg != 0 && lo != 0 {
		mpaddcfix(a, -1)
	}
}

//
// return the sign of (abs(a)-abs(b))
//
func mpcmp(a *Mpint, b *Mpint) int {
	if a.Ovf != 0 || b.Ovf != 0 {
		if nsavederrors+nerrors == 0 {
			Yyerror("ovf in cmp")
		}
		return 0
	}

	var x int
	for i := Mpprec - 1; i >= 0; i-- {
		x = a.A[i] - b.A[i]
		if x > 0 {
			return +1
		}
		if x < 0 {
			return -1
		}
	}

	return 0
}

//
// negate a
// ignore sign and ovf
//
func mpneg(a *Mpint) {
	var x int

	c := 0
	for i := 0; i < Mpprec; i++ {
		x = -a.A[i] - c
		c = 0
		if x < 0 {
			x += Mpbase
			c = 1
		}

		a.A[i] = x
	}
}

// shift left by s (or right by -s)
func Mpshiftfix(a *Mpint, s int) {
	if s >= 0 {
		for s >= Mpscale {
			mplshw(a, 0)
			s -= Mpscale
		}

		for s > 0 {
			mplsh(a, 0)
			s--
		}
	} else {
		s = -s
		for s >= Mpscale {
			mprshw(a)
			s -= Mpscale
		}

		for s > 0 {
			mprsh(a)
			s--
		}
	}
}

/// implements fix arihmetic

func mpaddfixfix(a *Mpint, b *Mpint, quiet int) {
	if a.Ovf != 0 || b.Ovf != 0 {
		if nsavederrors+nerrors == 0 {
			Yyerror("ovf in mpaddxx")
		}
		a.Ovf = 1
		return
	}

	c := 0
	var x int
	if a.Neg != b.Neg {
		goto sub
	}

	// perform a+b
	for i := 0; i < Mpprec; i++ {
		x = a.A[i] + b.A[i] + c
		c = 0
		if x >= Mpbase {
			x -= Mpbase
			c = 1
		}

		a.A[i] = x
	}

	a.Ovf = uint8(c)
	if a.Ovf != 0 && quiet == 0 {
		Yyerror("constant addition overflow")
	}

	return

	// perform a-b
sub:
	switch mpcmp(a, b) {
	case 0:
		Mpmovecfix(a, 0)

	case 1:
		var x int
		for i := 0; i < Mpprec; i++ {
			x = a.A[i] - b.A[i] - c
			c = 0
			if x < 0 {
				x += Mpbase
				c = 1
			}

			a.A[i] = x
		}

	case -1:
		a.Neg ^= 1
		var x int
		for i := 0; i < Mpprec; i++ {
			x = b.A[i] - a.A[i] - c
			c = 0
			if x < 0 {
				x += Mpbase
				c = 1
			}

			a.A[i] = x
		}
	}
}

func mpmulfixfix(a *Mpint, b *Mpint) {
	if a.Ovf != 0 || b.Ovf != 0 {
		if nsavederrors+nerrors == 0 {
			Yyerror("ovf in mpmulfixfix")
		}
		a.Ovf = 1
		return
	}

	// pick the smaller
	// to test for bits
	na := mplen(a)

	nb := mplen(b)
	var s Mpint
	var c *Mpint
	if na > nb {
		mpmovefixfix(&s, a)
		c = b
		na = nb
	} else {
		mpmovefixfix(&s, b)
		c = a
	}

	s.Neg = 0

	var q Mpint
	Mpmovecfix(&q, 0)
	var j int
	var x int
	for i := 0; i < na; i++ {
		x = c.A[i]
		for j = 0; j < Mpscale; j++ {
			if x&1 != 0 {
				if s.Ovf != 0 {
					q.Ovf = 1
					goto out
				}

				mpaddfixfix(&q, &s, 1)
				if q.Ovf != 0 {
					goto out
				}
			}

			mplsh(&s, 1)
			x >>= 1
		}
	}

out:
	q.Neg = a.Neg ^ b.Neg
	mpmovefixfix(a, &q)
	if a.Ovf != 0 {
		Yyerror("constant multiplication overflow")
	}
}

func mpmulfract(a *Mpint, b *Mpint) {
	if a.Ovf != 0 || b.Ovf != 0 {
		if nsavederrors+nerrors == 0 {
			Yyerror("ovf in mpmulflt")
		}
		a.Ovf = 1
		return
	}

	var s Mpint
	mpmovefixfix(&s, b)
	s.Neg = 0
	var q Mpint
	Mpmovecfix(&q, 0)

	i := Mpprec - 1
	x := a.A[i]
	if x != 0 {
		Yyerror("mpmulfract not normal")
	}

	var j int
	for i--; i >= 0; i-- {
		x = a.A[i]
		if x == 0 {
			mprshw(&s)
			continue
		}

		for j = 0; j < Mpscale; j++ {
			x <<= 1
			if x&Mpbase != 0 {
				mpaddfixfix(&q, &s, 1)
			}
			mprsh(&s)
		}
	}

	q.Neg = a.Neg ^ b.Neg
	mpmovefixfix(a, &q)
	if a.Ovf != 0 {
		Yyerror("constant multiplication overflow")
	}
}

func mporfixfix(a *Mpint, b *Mpint) {
	x := 0
	if a.Ovf != 0 || b.Ovf != 0 {
		if nsavederrors+nerrors == 0 {
			Yyerror("ovf in mporfixfix")
		}
		Mpmovecfix(a, 0)
		a.Ovf = 1
		return
	}

	if a.Neg != 0 {
		a.Neg = 0
		mpneg(a)
	}

	if b.Neg != 0 {
		mpneg(b)
	}

	for i := 0; i < Mpprec; i++ {
		x = a.A[i] | b.A[i]
		a.A[i] = x
	}

	if b.Neg != 0 {
		mpneg(b)
	}
	if x&Mpsign != 0 {
		a.Neg = 1
		mpneg(a)
	}
}

func mpandfixfix(a *Mpint, b *Mpint) {
	x := 0
	if a.Ovf != 0 || b.Ovf != 0 {
		if nsavederrors+nerrors == 0 {
			Yyerror("ovf in mpandfixfix")
		}
		Mpmovecfix(a, 0)
		a.Ovf = 1
		return
	}

	if a.Neg != 0 {
		a.Neg = 0
		mpneg(a)
	}

	if b.Neg != 0 {
		mpneg(b)
	}

	for i := 0; i < Mpprec; i++ {
		x = a.A[i] & b.A[i]
		a.A[i] = x
	}

	if b.Neg != 0 {
		mpneg(b)
	}
	if x&Mpsign != 0 {
		a.Neg = 1
		mpneg(a)
	}
}

func mpandnotfixfix(a *Mpint, b *Mpint) {
	x := 0
	if a.Ovf != 0 || b.Ovf != 0 {
		if nsavederrors+nerrors == 0 {
			Yyerror("ovf in mpandnotfixfix")
		}
		Mpmovecfix(a, 0)
		a.Ovf = 1
		return
	}

	if a.Neg != 0 {
		a.Neg = 0
		mpneg(a)
	}

	if b.Neg != 0 {
		mpneg(b)
	}

	for i := 0; i < Mpprec; i++ {
		x = a.A[i] &^ b.A[i]
		a.A[i] = x
	}

	if b.Neg != 0 {
		mpneg(b)
	}
	if x&Mpsign != 0 {
		a.Neg = 1
		mpneg(a)
	}
}

func mpxorfixfix(a *Mpint, b *Mpint) {
	x := 0
	if a.Ovf != 0 || b.Ovf != 0 {
		if nsavederrors+nerrors == 0 {
			Yyerror("ovf in mporfixfix")
		}
		Mpmovecfix(a, 0)
		a.Ovf = 1
		return
	}

	if a.Neg != 0 {
		a.Neg = 0
		mpneg(a)
	}

	if b.Neg != 0 {
		mpneg(b)
	}

	for i := 0; i < Mpprec; i++ {
		x = a.A[i] ^ b.A[i]
		a.A[i] = x
	}

	if b.Neg != 0 {
		mpneg(b)
	}
	if x&Mpsign != 0 {
		a.Neg = 1
		mpneg(a)
	}
}

func mplshfixfix(a *Mpint, b *Mpint) {
	if a.Ovf != 0 || b.Ovf != 0 {
		if nsavederrors+nerrors == 0 {
			Yyerror("ovf in mporfixfix")
		}
		Mpmovecfix(a, 0)
		a.Ovf = 1
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

func mprshfixfix(a *Mpint, b *Mpint) {
	if a.Ovf != 0 || b.Ovf != 0 {
		if nsavederrors+nerrors == 0 {
			Yyerror("ovf in mprshfixfix")
		}
		Mpmovecfix(a, 0)
		a.Ovf = 1
		return
	}

	s := Mpgetfix(b)
	if s < 0 || s >= Mpprec*Mpscale {
		Yyerror("stupid shift: %d", s)
		if a.Neg != 0 {
			Mpmovecfix(a, -1)
		} else {
			Mpmovecfix(a, 0)
		}
		return
	}

	Mpshiftfix(a, int(-s))
}

func mpnegfix(a *Mpint) {
	a.Neg ^= 1
}

func Mpgetfix(a *Mpint) int64 {
	if a.Ovf != 0 {
		if nsavederrors+nerrors == 0 {
			Yyerror("constant overflow")
		}
		return 0
	}

	v := int64(uint64(a.A[0]))
	v |= int64(uint64(a.A[1]) << Mpscale)
	v |= int64(uint64(a.A[2]) << (Mpscale + Mpscale))
	if a.Neg != 0 {
		v = int64(-uint64(v))
	}
	return v
}

func Mpmovecfix(a *Mpint, c int64) {
	a.Neg = 0
	a.Ovf = 0

	x := c
	if x < 0 {
		a.Neg = 1
		x = int64(-uint64(x))
	}

	for i := 0; i < Mpprec; i++ {
		a.A[i] = int(x & Mpmask)
		x >>= Mpscale
	}
}

func mpdivmodfixfix(q *Mpint, r *Mpint, n *Mpint, d *Mpint) {
	var i int

	ns := int(n.Neg)
	ds := int(d.Neg)
	n.Neg = 0
	d.Neg = 0

	mpmovefixfix(r, n)
	Mpmovecfix(q, 0)

	// shift denominator until it
	// is larger than numerator
	for i = 0; i < Mpprec*Mpscale; i++ {
		if mpcmp(d, r) > 0 {
			break
		}
		mplsh(d, 1)
	}

	// if it never happens
	// denominator is probably zero
	if i >= Mpprec*Mpscale {
		q.Ovf = 1
		r.Ovf = 1
		n.Neg = uint8(ns)
		d.Neg = uint8(ds)
		Yyerror("constant division overflow")
		return
	}

	// shift denominator back creating
	// quotient a bit at a time
	// when done the remaining numerator
	// will be the remainder
	for ; i > 0; i-- {
		mplsh(q, 1)
		mprsh(d)
		if mpcmp(d, r) <= 0 {
			mpaddcfix(q, 1)
			mpsubfixfix(r, d)
		}
	}

	n.Neg = uint8(ns)
	d.Neg = uint8(ds)
	r.Neg = uint8(ns)
	q.Neg = uint8(ns ^ ds)
}

func mpiszero(a *Mpint) bool {
	for i := Mpprec - 1; i >= 0; i-- {
		if a.A[i] != 0 {
			return false
		}
	}
	return true
}

func mpdivfract(a *Mpint, b *Mpint) {
	var n Mpint
	var d Mpint
	var j int
	var x int

	mpmovefixfix(&n, a) // numerator
	mpmovefixfix(&d, b) // denominator

	neg := int(n.Neg) ^ int(d.Neg)

	n.Neg = 0
	d.Neg = 0
	for i := Mpprec - 1; i >= 0; i-- {
		x = 0
		for j = 0; j < Mpscale; j++ {
			x <<= 1
			if mpcmp(&d, &n) <= 0 {
				if !mpiszero(&d) {
					x |= 1
				}
				mpsubfixfix(&n, &d)
			}

			mprsh(&d)
		}

		a.A[i] = x
	}

	a.Neg = uint8(neg)
}

func mptestfix(a *Mpint) int {
	var b Mpint

	Mpmovecfix(&b, 0)
	r := mpcmp(a, &b)
	if a.Neg != 0 {
		if r > 0 {
			return -1
		}
		if r < 0 {
			return +1
		}
	}

	return r
}
