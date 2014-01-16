package main

// Tests of field promotion logic.

type A struct {
	x int
	y *int
}

type B struct {
	p int
	q *int
}

type C struct {
	A
	*B
}

type D struct {
	a int
	C
}

func assert(cond bool) {
	if !cond {
		panic("failed")
	}
}

func f1(c C) {
	assert(c.x == c.A.x)
	assert(c.y == c.A.y)
	assert(&c.x == &c.A.x)
	assert(&c.y == &c.A.y)

	assert(c.p == c.B.p)
	assert(c.q == c.B.q)
	assert(&c.p == &c.B.p)
	assert(&c.q == &c.B.q)

	c.x = 1
	*c.y = 1
	c.p = 1
	*c.q = 1
}

func f2(c *C) {
	assert(c.x == c.A.x)
	assert(c.y == c.A.y)
	assert(&c.x == &c.A.x)
	assert(&c.y == &c.A.y)

	assert(c.p == c.B.p)
	assert(c.q == c.B.q)
	assert(&c.p == &c.B.p)
	assert(&c.q == &c.B.q)

	c.x = 1
	*c.y = 1
	c.p = 1
	*c.q = 1
}

func f3(d D) {
	assert(d.x == d.C.A.x)
	assert(d.y == d.C.A.y)
	assert(&d.x == &d.C.A.x)
	assert(&d.y == &d.C.A.y)

	assert(d.p == d.C.B.p)
	assert(d.q == d.C.B.q)
	assert(&d.p == &d.C.B.p)
	assert(&d.q == &d.C.B.q)

	d.x = 1
	*d.y = 1
	d.p = 1
	*d.q = 1
}

func f4(d *D) {
	assert(d.x == d.C.A.x)
	assert(d.y == d.C.A.y)
	assert(&d.x == &d.C.A.x)
	assert(&d.y == &d.C.A.y)

	assert(d.p == d.C.B.p)
	assert(d.q == d.C.B.q)
	assert(&d.p == &d.C.B.p)
	assert(&d.q == &d.C.B.q)

	d.x = 1
	*d.y = 1
	d.p = 1
	*d.q = 1
}

func main() {
	y := 123
	c := C{
		A{x: 42, y: &y},
		&B{p: 42, q: &y},
	}

	assert(&c.x == &c.A.x)

	f1(c)
	f2(&c)

	d := D{C: c}
	f3(d)
	f4(&d)
}
