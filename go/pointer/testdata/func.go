//go:build ignore
// +build ignore

package main

var a, b, c int

var unknown bool // defeat dead-code elimination

func func1() {
	var h int // @line f1h
	f := func(x *int) *int {
		if unknown {
			return &b
		}
		return x
	}

	// FV(g) = {f, h}
	g := func(x *int) *int {
		if unknown {
			return &h
		}
		return f(x)
	}

	print(g(&a)) // @pointsto command-line-arguments.a | command-line-arguments.b | h@f1h:6
	print(f(&a)) // @pointsto command-line-arguments.a | command-line-arguments.b
	print(&a)    // @pointsto command-line-arguments.a
}

// @calls command-line-arguments.func1 -> command-line-arguments.func1$2
// @calls command-line-arguments.func1 -> command-line-arguments.func1$1
// @calls command-line-arguments.func1$2 ->  command-line-arguments.func1$1

func func2() {
	var x, y *int
	defer func() {
		x = &a
	}()
	go func() {
		y = &b
	}()
	print(x) // @pointsto command-line-arguments.a
	print(y) // @pointsto command-line-arguments.b
}

func func3() {
	x, y := func() (x, y *int) {
		x = &a
		y = &b
		if unknown {
			return nil, &c
		}
		return
	}()
	print(x) // @pointsto command-line-arguments.a
	print(y) // @pointsto command-line-arguments.b | command-line-arguments.c
}

func swap(x, y *int) (*int, *int) { // @line swap
	print(&x) // @pointsto x@swap:11
	print(x)  // @pointsto makeslice[*]@func4make:11
	print(&y) // @pointsto y@swap:14
	print(y)  // @pointsto j@f4j:5
	return y, x
}

func func4() {
	a := make([]int, 10) // @line func4make
	i, j := 123, 456     // @line f4j
	_ = i
	p, q := swap(&a[3], &j)
	print(p) // @pointsto j@f4j:5
	print(q) // @pointsto makeslice[*]@func4make:11

	f := &b
	print(f) // @pointsto command-line-arguments.b
}

type T int

func (t *T) f(x *int) *int {
	print(t) // @pointsto command-line-arguments.a
	print(x) // @pointsto command-line-arguments.c
	return &b
}

func (t *T) g(x *int) *int {
	print(t) // @pointsto command-line-arguments.a
	print(x) // @pointsto command-line-arguments.b
	return &c
}

func (t *T) h(x *int) *int {
	print(t) // @pointsto command-line-arguments.a
	print(x) // @pointsto command-line-arguments.b
	return &c
}

var h func(*T, *int) *int

func func5() {
	// Static call of method.
	t := (*T)(&a)
	print(t.f(&c)) // @pointsto command-line-arguments.b

	// Static call of method as function
	print((*T).g(t, &b)) // @pointsto command-line-arguments.c

	// Dynamic call (not invoke) of method.
	h = (*T).h
	print(h(t, &b)) // @pointsto command-line-arguments.c
}

// @calls command-line-arguments.func5 -> (*command-line-arguments.T).f
// @calls command-line-arguments.func5 -> (*command-line-arguments.T).g$thunk
// @calls command-line-arguments.func5 -> (*command-line-arguments.T).h$thunk

func func6() {
	A := &a
	f := func() *int {
		return A // (free variable)
	}
	print(f()) // @pointsto command-line-arguments.a
}

// @calls command-line-arguments.func6 -> command-line-arguments.func6$1

type I interface {
	f()
}

type D struct{}

func (D) f() {}

func func7() {
	var i I = D{}
	imethodClosure := i.f
	imethodClosure()
	// @calls command-line-arguments.func7 -> (command-line-arguments.I).f$bound
	// @calls (command-line-arguments.I).f$bound -> (command-line-arguments.D).f

	var d D
	cmethodClosure := d.f
	cmethodClosure()
	// @calls command-line-arguments.func7 -> (command-line-arguments.D).f$bound
	// @calls (command-line-arguments.D).f$bound ->(command-line-arguments.D).f

	methodExpr := D.f
	methodExpr(d)
	// @calls command-line-arguments.func7 -> (command-line-arguments.D).f$thunk
}

func func8(x ...int) {
	print(&x[0]) // @pointsto varargs[*]@varargs:15
}

type E struct {
	x1, x2, x3, x4, x5 *int
}

func (e E) f() {}

func func9() {
	// Regression test for bug reported by Jon Valdes on golang-dev, Jun 19 2014.
	// The receiver of a bound method closure may be of a multi-node type, E.
	// valueNode was reserving only a single node for it, so the
	// nodes used by the immediately following constraints
	// (e.g. param 'i') would get clobbered.

	var e E
	e.x1 = &a
	e.x2 = &a
	e.x3 = &a
	e.x4 = &a
	e.x5 = &a

	_ = e.f // form a closure---must reserve sizeof(E) nodes

	func(i I) {
		i.f() // must not crash the solver
	}(new(D))

	print(e.x1) // @pointsto command-line-arguments.a
	print(e.x2) // @pointsto command-line-arguments.a
	print(e.x3) // @pointsto command-line-arguments.a
	print(e.x4) // @pointsto command-line-arguments.a
	print(e.x5) // @pointsto command-line-arguments.a
}

func main() {
	func1()
	func2()
	func3()
	func4()
	func5()
	func6()
	func7()
	func8(1, 2, 3) // @line varargs
	func9()
}

// @calls <root> -> command-line-arguments.main
// @calls <root> -> command-line-arguments.init
