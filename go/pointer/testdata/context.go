//go:build ignore
// +build ignore

package main

// Test of context-sensitive treatment of certain function calls,
// e.g. static calls to simple accessor methods.

var a, b int

type T struct{ x *int }

func (t *T) SetX(x *int) { t.x = x }
func (t *T) GetX() *int  { return t.x }

func context1() {
	var t1, t2 T
	t1.SetX(&a)
	t2.SetX(&b)
	print(t1.GetX()) // @pointsto command-line-arguments.a
	print(t2.GetX()) // @pointsto command-line-arguments.b
}

func context2() {
	id := func(x *int) *int {
		print(x) // @pointsto command-line-arguments.a | command-line-arguments.b
		return x
	}
	print(id(&a)) // @pointsto command-line-arguments.a
	print(id(&b)) // @pointsto command-line-arguments.b

	// Same again, but anon func has free vars.
	var c int // @line context2c
	id2 := func(x *int) (*int, *int) {
		print(x) // @pointsto command-line-arguments.a | command-line-arguments.b
		return x, &c
	}
	p, q := id2(&a)
	print(p) // @pointsto command-line-arguments.a
	print(q) // @pointsto c@context2c:6
	r, s := id2(&b)
	print(r) // @pointsto command-line-arguments.b
	print(s) // @pointsto c@context2c:6
}

func main() {
	context1()
	context2()
}
