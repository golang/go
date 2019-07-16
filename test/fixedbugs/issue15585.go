// compile

// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bug

func example(n int) (rc int) {
	var cc, ll, pp, rr [27]int
	for q0 := 0; q0 < n-2; q0++ {
		for q1 := q0 + 2; q1 < n; q1++ {
			var c, d, l, p, r int
			b0 := 1 << uint(q0)
			b1 := 1 << uint(q1)
			l = ((b0 << 1) | b1) << 1
			c = b0 | b1 | (-1 << uint(n))
			r = ((b0 >> 1) | b1) >> 1
		E:
			if c != -1 {
				p = ^(l | c | r)
			} else {
				rc++
				goto R
			}
		L:
			if p != 0 {
				lsb := p & -p
				p &^= lsb
				ll[d], cc[d], rr[d], pp[d] = l, c, r, p
				l, c, r = (l|lsb)<<1, c|lsb, (r|lsb)>>1
				d++
				goto E
			}
		R:
			d--
			if d >= 0 {
				l, c, r, p = ll[d], cc[d], rr[d], pp[d]
				goto L
			}
		}
	}
	rc <<= 1
	return
}
