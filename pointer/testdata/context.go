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
	print(t1.GetX()) // @pointsto main.a
	print(t2.GetX()) // @pointsto main.b
}

func context2() {
	id := func(x *int) *int {
		print(x) // @pointsto main.a | main.b
		return x
	}
	print(id(&a)) // @pointsto main.a
	print(id(&b)) // @pointsto main.b

	// Same again, but anon func has free vars.
	var c int // @line context2c
	id2 := func(x *int) (*int, *int) {
		print(x) // @pointsto main.a | main.b
		return x, &c
	}
	p, q := id2(&a)
	print(p) // @pointsto main.a
	print(q) // @pointsto c@context2c:6
	r, s := id2(&b)
	print(r) // @pointsto main.b
	print(s) // @pointsto c@context2c:6
}

func main() {
	context1()
	context2()
}
