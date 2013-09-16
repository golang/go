// +build ignore

package main

// Demonstration of directionality of flow edges.

func f1() {}
func f2() {}

var somepred bool

// Tracking functions.
func flow1() {
	s := f1
	p := f2
	q := p
	r := q
	if somepred {
		r = s
	}
	print(s) // @pointsto main.f1
	print(p) // @pointsto main.f2
	print(q) // @pointsto main.f2
	print(r) // @pointsto main.f1 | main.f2
}

// Tracking concrete types in interfaces.
func flow2() {
	var s interface{} = 1
	var p interface{} = "foo"
	q := p
	r := q
	if somepred {
		r = s
	}
	print(s) // @types int
	print(p) // @types string
	print(q) // @types string
	print(r) // @types int | string
}

var g1, g2 int

// Tracking addresses of globals.
func flow3() {
	s := &g1
	p := &g2
	q := p
	r := q
	if somepred {
		r = s
	}
	print(s) // @pointsto main.g1
	print(p) // @pointsto main.g2
	print(q) // @pointsto main.g2
	print(r) // @pointsto main.g2 | main.g1
}

func main() {
	flow1()
	flow2()
	flow3()
}
