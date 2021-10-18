//go:build ignore
// +build ignore

// This is a slice of the fmt package.

package main

type pp struct {
	field interface{}
}

func newPrinter() *pp {
	return new(pp)
}

func Fprintln(a ...interface{}) {
	p := newPrinter()
	p.doPrint(a, true, true)
}

func Println(a ...interface{}) {
	Fprintln(a...)
}

func (p *pp) doPrint(a []interface{}, addspace, addnewline bool) {
	print(a[0]) // @types S | string
	stringer := a[0].(interface {
		String() string
	})

	stringer.String()
	print(stringer) // @types S
}

type S int

func (S) String() string { return "" }

func main() {
	Println("Hello, World!", S(0))
}

// @calls (*command-line-arguments.pp).doPrint -> (command-line-arguments.S).String
