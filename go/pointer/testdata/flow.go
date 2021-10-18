//go:build ignore
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
	print(s) // @pointsto command-line-arguments.f1
	print(p) // @pointsto command-line-arguments.f2
	print(q) // @pointsto command-line-arguments.f2
	print(r) // @pointsto command-line-arguments.f1 | command-line-arguments.f2
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
	print(s) // @pointsto command-line-arguments.g1
	print(p) // @pointsto command-line-arguments.g2
	print(q) // @pointsto command-line-arguments.g2
	print(r) // @pointsto command-line-arguments.g2 | command-line-arguments.g1
}

func main() {
	flow1()
	flow2()
	flow3()
}
