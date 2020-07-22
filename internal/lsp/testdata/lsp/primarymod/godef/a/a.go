// Package a is a package for testing go to definition.
package a //@mark(aPackage, "a "),hover("a ", aPackage)

import (
	"fmt"
	"go/types"
	"sync"
)

var (
	// x is a variable.
	x string //@x,hover("x", x)
)

// Constant block. When I hover on h, I should see this comment.
const (
	// When I hover on g, I should see this comment.
	g = 1 //@g,hover("g", g)

	h = 2 //@h,hover("h", h)
)

// z is a variable too.
var z string //@z,hover("z", z)

type A string //@mark(AString, "A")

func AStuff() { //@AStuff
	x := 5
	Random2(x) //@godef("dom2", Random2)
	Random()   //@godef("()", Random)

	var err error         //@err
	fmt.Printf("%v", err) //@godef("err", err)

	var y string       //@string,hover("string", string)
	_ = make([]int, 0) //@make,hover("make", make)

	var mu sync.Mutex
	mu.Lock() //@Lock,hover("Lock", Lock)

	var typ *types.Named //@mark(typesImport, "types"),hover("types", typesImport)
	typ.Obj().Name()     //@Name,hover("Name", Name)
}

type A struct {
}

func (_ A) Hi() {} //@mark(AHi, "Hi")

type S struct {
	Field int //@mark(AField, "Field")
	R         // embed a struct
	H         // embed an interface
}

type R struct {
	Field2 int //@mark(AField2, "Field2")
}

func (_ R) Hey() {} //@mark(AHey, "Hey")

type H interface {
	Goodbye() //@mark(AGoodbye, "Goodbye")
}

type I interface {
	B() //@mark(AB, "B")
	J
}

type J interface {
	Hello() //@mark(AHello, "Hello")
}
