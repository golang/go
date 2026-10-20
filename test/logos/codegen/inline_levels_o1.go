// errorcheck -O=1 -m=2 -0

package main

// M13-S1 -O levels (SPEC T-105), O1 side of inline_levels.go: at O1 the
// upstream budget of 80 applies, so neither the 167-node mediumCost nor
// the recursive pair inlines. main stays empty so call-site and escape
// notes never appear.

func mediumCost(a, b, c, d int) int { // ERROR "cannot inline mediumCost"
	x1 := a + b
	x2 := c + d
	x3 := a * c
	x4 := b * d
	y1 := x1 * x2
	y2 := x3 * x4
	z1 := y1 - y2
	z2 := y1 + y2
	w1 := z1 * x1
	w2 := z2 * x2
	w3 := w1 + w2
	w4 := w3 - x3
	q1 := w4 * x4
	q2 := q1 + y1
	q3 := q2 - y2
	q4 := q3 * z1
	r1 := q4 + z2
	r2 := r1 - w1
	r3 := r2 * w2
	r4 := r3 + w3
	s1 := r4 - w4
	s2 := s1 * q1
	s3 := s2 + q2
	return s3 + r1 + x1
}

func recF(n int) int { // ERROR "cannot inline recF"
	if n <= 0 {
		return 0
	}
	a := n * 3
	b := a - n
	c := b * a
	d := c - b
	e := d * c
	f := e - d
	g := f * e
	h := g - f
	i := h * g
	j := i - h
	k := j * i
	l := k - j
	m := l * k
	o := m - l
	p := o * m
	return recG(n-1) + p
}

func recG(n int) int { // ERROR "cannot inline recG"
	if n <= 0 {
		return 1
	}
	a := n * 5
	b := a - n
	c := b * a
	d := c - b
	e := d * c
	f := e - d
	g := f * e
	h := g - f
	i := h * g
	j := i - h
	k := j * i
	l := k - j
	m := l * k
	o := m - l
	p := o * m
	return recF(n-1) + p
}

func main() {} // ERROR "can inline main"
