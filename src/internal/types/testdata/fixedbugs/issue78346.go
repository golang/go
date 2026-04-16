// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

const (
	a = "x"
	b = a + a
	c = b + b
	d = c + c
	e = d + d
	f = e + e
	g = f + f
	h = g + g
	i = h + h
	j = i + i
	k = j + j
	l = k + k
	m = l + l
	n = m + m
	o = n + n
	p = o + o
	q = p + p
	r = q + q
	s = r + r
	t = s + s
	u = t + t
	v = u + u
	w = v + v
	x = w + w
	y = x + /* ERROR "constant string too long" */ x
	z = y + y
)
