// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package big

import "math/bits"

func (z nat) div(z2, u, v nat) (q, r nat) {
	if len(v) == 0 {
		panic("division by zero")
	}

	if u.cmp(v) < 0 {
		q = z[:0]
		r = z2.set(u)
		return
	}

	if len(v) == 1 {
		var r2 Word
		q, r2 = z.divW(u, v[0])
		r = z2.setWord(r2)
		return
	}

	q, r = z.divLarge(z2, u, v)
	return
}

// q = (x-r)/y, with 0 <= r < y
func (z nat) divW(x nat, y Word) (q nat, r Word) {
	m := len(x)
	switch {
	case y == 0:
		panic("division by zero")
	case y == 1:
		q = z.set(x) // result is x
		return
	case m == 0:
		q = z[:0] // result is 0
		return
	}
	// m > 0
	z = z.make(m)
	r = divWVW(z, 0, x, y)
	q = z.norm()
	return
}

// modW returns x % d.
func (x nat) modW(d Word) (r Word) {
	// TODO(agl): we don't actually need to store the q value.
	var q nat
	q = q.make(len(x))
	return divWVW(q, 0, x, d)
}

func divWVW(z []Word, xn Word, x []Word, y Word) (r Word) {
	r = xn
	if len(x) == 1 {
		qq, rr := bits.Div(uint(r), uint(x[0]), uint(y))
		z[0] = Word(qq)
		return Word(rr)
	}
	rec := reciprocalWord(y)
	for i := len(z) - 1; i >= 0; i-- {
		z[i], r = divWW(r, x[i], y, rec)
	}
	return r
}

// q = (uIn-r)/vIn, with 0 <= r < vIn
// Uses z as storage for q, and u as storage for r if possible.
// See Knuth, Volume 2, section 4.3.1, Algorithm D.
// Preconditions:
//    len(vIn) >= 2
//    len(uIn) >= len(vIn)
//    u must not alias z
func (z nat) divLarge(u, uIn, vIn nat) (q, r nat) {
	n := len(vIn)
	m := len(uIn) - n

	// D1.
	shift := nlz(vIn[n-1])
	// do not modify vIn, it may be used by another goroutine simultaneously
	vp := getNat(n)
	v := *vp
	shlVU(v, vIn, shift)

	// u may safely alias uIn or vIn, the value of uIn is used to set u and vIn was already used
	u = u.make(len(uIn) + 1)
	u[len(uIn)] = shlVU(u[0:len(uIn)], uIn, shift)

	// z may safely alias uIn or vIn, both values were used already
	if alias(z, u) {
		z = nil // z is an alias for u - cannot reuse
	}
	q = z.make(m + 1)

	if n < divRecursiveThreshold {
		q.divBasic(u, v)
	} else {
		q.divRecursive(u, v)
	}
	putNat(vp)

	q = q.norm()
	shrVU(u, u, shift)
	r = u.norm()

	return q, r
}

// divBasic performs word-by-word division of u by v.
// The quotient is written in pre-allocated q.
// The remainder overwrites input u.
//
// Precondition:
// - q is large enough to hold the quotient u / v
//   which has a maximum length of len(u)-len(v)+1.
func (q nat) divBasic(u, v nat) {
	n := len(v)
	m := len(u) - n

	qhatvp := getNat(n + 1)
	qhatv := *qhatvp

	// D2.
	vn1 := v[n-1]
	rec := reciprocalWord(vn1)
	for j := m; j >= 0; j-- {
		// D3.
		qhat := Word(_M)
		var ujn Word
		if j+n < len(u) {
			ujn = u[j+n]
		}
		if ujn != vn1 {
			var rhat Word
			qhat, rhat = divWW(ujn, u[j+n-1], vn1, rec)

			// x1 | x2 = q̂v_{n-2}
			vn2 := v[n-2]
			x1, x2 := mulWW(qhat, vn2)
			// test if q̂v_{n-2} > br̂ + u_{j+n-2}
			ujn2 := u[j+n-2]
			for greaterThan(x1, x2, rhat, ujn2) {
				qhat--
				prevRhat := rhat
				rhat += vn1
				// v[n-1] >= 0, so this tests for overflow.
				if rhat < prevRhat {
					break
				}
				x1, x2 = mulWW(qhat, vn2)
			}
		}

		// D4.
		// Compute the remainder u - (q̂*v) << (_W*j).
		// The subtraction may overflow if q̂ estimate was off by one.
		qhatv[n] = mulAddVWW(qhatv[0:n], v, qhat, 0)
		qhl := len(qhatv)
		if j+qhl > len(u) && qhatv[n] == 0 {
			qhl--
		}
		c := subVV(u[j:j+qhl], u[j:], qhatv)
		if c != 0 {
			c := addVV(u[j:j+n], u[j:], v)
			// If n == qhl, the carry from subVV and the carry from addVV
			// cancel out and don't affect u[j+n].
			if n < qhl {
				u[j+n] += c
			}
			qhat--
		}

		if j == m && m == len(q) && qhat == 0 {
			continue
		}
		q[j] = qhat
	}

	putNat(qhatvp)
}

// greaterThan reports whether (x1<<_W + x2) > (y1<<_W + y2)
func greaterThan(x1, x2, y1, y2 Word) bool {
	return x1 > y1 || x1 == y1 && x2 > y2
}

const divRecursiveThreshold = 100

// divRecursive performs word-by-word division of u by v.
// The quotient is written in pre-allocated z.
// The remainder overwrites input u.
//
// Precondition:
// - len(z) >= len(u)-len(v)
//
// See Burnikel, Ziegler, "Fast Recursive Division", Algorithm 1 and 2.
func (z nat) divRecursive(u, v nat) {
	// Recursion depth is less than 2 log2(len(v))
	// Allocate a slice of temporaries to be reused across recursion.
	recDepth := 2 * bits.Len(uint(len(v)))
	// large enough to perform Karatsuba on operands as large as v
	tmp := getNat(3 * len(v))
	temps := make([]*nat, recDepth)
	z.clear()
	z.divRecursiveStep(u, v, 0, tmp, temps)
	for _, n := range temps {
		if n != nil {
			putNat(n)
		}
	}
	putNat(tmp)
}

// divRecursiveStep computes the division of u by v.
// - z must be large enough to hold the quotient
// - the quotient will overwrite z
// - the remainder will overwrite u
func (z nat) divRecursiveStep(u, v nat, depth int, tmp *nat, temps []*nat) {
	u = u.norm()
	v = v.norm()

	if len(u) == 0 {
		z.clear()
		return
	}
	n := len(v)
	if n < divRecursiveThreshold {
		z.divBasic(u, v)
		return
	}
	m := len(u) - n
	if m < 0 {
		return
	}

	// Produce the quotient by blocks of B words.
	// Division by v (length n) is done using a length n/2 division
	// and a length n/2 multiplication for each block. The final
	// complexity is driven by multiplication complexity.
	B := n / 2

	// Allocate a nat for qhat below.
	if temps[depth] == nil {
		temps[depth] = getNat(n)
	} else {
		*temps[depth] = temps[depth].make(B + 1)
	}

	j := m
	for j > B {
		// Divide u[j-B:j+n] by vIn. Keep remainder in u
		// for next block.
		//
		// The following property will be used (Lemma 2):
		// if u = u1 << s + u0
		//    v = v1 << s + v0
		// then floor(u1/v1) >= floor(u/v)
		//
		// Moreover, the difference is at most 2 if len(v1) >= len(u/v)
		// We choose s = B-1 since len(v)-s >= B+1 >= len(u/v)
		s := (B - 1)
		// Except for the first step, the top bits are always
		// a division remainder, so the quotient length is <= n.
		uu := u[j-B:]

		qhat := *temps[depth]
		qhat.clear()
		qhat.divRecursiveStep(uu[s:B+n], v[s:], depth+1, tmp, temps)
		qhat = qhat.norm()
		// Adjust the quotient:
		//    u = u_h << s + u_l
		//    v = v_h << s + v_l
		//  u_h = q̂ v_h + rh
		//    u = q̂ (v - v_l) + rh << s + u_l
		// After the above step, u contains a remainder:
		//    u = rh << s + u_l
		// and we need to subtract q̂ v_l
		//
		// But it may be a bit too large, in which case q̂ needs to be smaller.
		qhatv := tmp.make(3 * n)
		qhatv.clear()
		qhatv = qhatv.mul(qhat, v[:s])
		for i := 0; i < 2; i++ {
			e := qhatv.cmp(uu.norm())
			if e <= 0 {
				break
			}
			subVW(qhat, qhat, 1)
			c := subVV(qhatv[:s], qhatv[:s], v[:s])
			if len(qhatv) > s {
				subVW(qhatv[s:], qhatv[s:], c)
			}
			addAt(uu[s:], v[s:], 0)
		}
		if qhatv.cmp(uu.norm()) > 0 {
			panic("impossible")
		}
		c := subVV(uu[:len(qhatv)], uu[:len(qhatv)], qhatv)
		if c > 0 {
			subVW(uu[len(qhatv):], uu[len(qhatv):], c)
		}
		addAt(z, qhat, j-B)
		j -= B
	}

	// Now u < (v<<B), compute lower bits in the same way.
	// Choose shift = B-1 again.
	s := B - 1
	qhat := *temps[depth]
	qhat.clear()
	qhat.divRecursiveStep(u[s:].norm(), v[s:], depth+1, tmp, temps)
	qhat = qhat.norm()
	qhatv := tmp.make(3 * n)
	qhatv.clear()
	qhatv = qhatv.mul(qhat, v[:s])
	// Set the correct remainder as before.
	for i := 0; i < 2; i++ {
		if e := qhatv.cmp(u.norm()); e > 0 {
			subVW(qhat, qhat, 1)
			c := subVV(qhatv[:s], qhatv[:s], v[:s])
			if len(qhatv) > s {
				subVW(qhatv[s:], qhatv[s:], c)
			}
			addAt(u[s:], v[s:], 0)
		}
	}
	if qhatv.cmp(u.norm()) > 0 {
		panic("impossible")
	}
	c := subVV(u[0:len(qhatv)], u[0:len(qhatv)], qhatv)
	if c > 0 {
		c = subVW(u[len(qhatv):], u[len(qhatv):], c)
	}
	if c > 0 {
		panic("impossible")
	}

	// Done!
	addAt(z, qhat.norm(), 0)
}
