// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

//
// return the significant
// words of the argument
//
func mplen(a *Mpfix) int {
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
func mplsh(a *Mpfix, quiet int) {
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
func mplshw(a *Mpfix, quiet int) {
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
func mprsh(a *Mpfix) {
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
func mprshw(a *Mpfix) {
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
func mpcmp(a *Mpfix, b *Mpfix) int {
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
func mpneg(a *Mpfix) {
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

// shift left by s (or right by -s)
func _Mpshiftfix(a *Mpfix, s int) {
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

func _mpaddfixfix(a *Mpfix, b *Mpfix, quiet int) {
	if a.Ovf != 0 || b.Ovf != 0 {
		if nsavederrors+nerrors == 0 {
			Yyerror("ovf in mpaddxx")
		}
		a.Ovf = 1
		return
	}

	c := 0
	if a.Neg != b.Neg {
		// perform a-b
		switch mpcmp(a, b) {
		case 0:
			_Mpmovecfix(a, 0)

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
		return
	}

	// perform a+b
	var x int
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

func _mpmulfixfix(a *Mpfix, b *Mpfix) {
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
	var s Mpfix
	var c *Mpfix
	if na > nb {
		_mpmovefixfix(&s, a)
		c = b
		na = nb
	} else {
		_mpmovefixfix(&s, b)
		c = a
	}

	s.Neg = 0

	var q Mpfix
	_Mpmovecfix(&q, 0)
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

				_mpaddfixfix(&q, &s, 1)
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
	_mpmovefixfix(a, &q)
	if a.Ovf != 0 {
		Yyerror("constant multiplication overflow")
	}
}

func mpmulfract(a *Mpfix, b *Mpfix) {
	if a.Ovf != 0 || b.Ovf != 0 {
		if nsavederrors+nerrors == 0 {
			Yyerror("ovf in mpmulflt")
		}
		a.Ovf = 1
		return
	}

	var s Mpfix
	_mpmovefixfix(&s, b)
	s.Neg = 0
	var q Mpfix
	_Mpmovecfix(&q, 0)

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
				_mpaddfixfix(&q, &s, 1)
			}
			mprsh(&s)
		}
	}

	q.Neg = a.Neg ^ b.Neg
	_mpmovefixfix(a, &q)
	if a.Ovf != 0 {
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

func _mpnegfix(a *Mpfix) {
	a.Neg ^= 1
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

func _Mpgetfix(a *Mpfix) int64 {
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
	a.Val.SetInt64(c)
}

func _Mpmovecfix(a *Mpfix, c int64) {
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

func mpdivmodfixfix(q *Mpfix, r *Mpfix, n *Mpfix, d *Mpfix) {
	var i int

	ns := int(n.Neg)
	ds := int(d.Neg)
	n.Neg = 0
	d.Neg = 0

	_mpmovefixfix(r, n)
	_Mpmovecfix(q, 0)

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
			_mpsubfixfix(r, d)
		}
	}

	n.Neg = uint8(ns)
	d.Neg = uint8(ds)
	r.Neg = uint8(ns)
	q.Neg = uint8(ns ^ ds)
}

func mpiszero(a *Mpfix) bool {
	for i := Mpprec - 1; i >= 0; i-- {
		if a.A[i] != 0 {
			return false
		}
	}
	return true
}

func mpdivfract(a *Mpfix, b *Mpfix) {
	var n Mpfix
	var d Mpfix
	var j int
	var x int

	_mpmovefixfix(&n, a) // numerator
	_mpmovefixfix(&d, b) // denominator

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
				_mpsubfixfix(&n, &d)
			}

			mprsh(&d)
		}

		a.A[i] = x
	}

	a.Neg = uint8(neg)
}

func mptestfix(a *Mpfix) int {
	var b Mpfix

	_Mpmovecfix(&b, 0)
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
