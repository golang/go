package main

import (
	"io"
)

var x = 42 //@mark(symbolsx, "x"), symbol("x", "x", "Variable", "", "x")

const y = 43 //@symbol("y", "y", "Constant", "", "y")

type Number int //@symbol("Number", "Number", "Number", "", "Number")

type Alias = string //@symbol("Alias", "Alias", "String", "", "Alias")

type NumberAlias = Number //@symbol("NumberAlias", "NumberAlias", "Number", "", "NumberAlias")

type (
	Boolean   bool   //@symbol("Boolean", "Boolean", "Boolean", "", "Boolean")
	BoolAlias = bool //@symbol("BoolAlias", "BoolAlias", "Boolean", "", "BoolAlias")
)

type Foo struct { //@mark(symbolsFoo, "Foo"), symbol("Foo", "Foo", "Struct", "", "Foo")
	Quux           //@mark(fQuux, "Quux"), symbol("Quux", "Quux", "Field", "Foo", "Quux")
	W    io.Writer //@symbol("W" , "W", "Field", "Foo", "W")
	Bar  int       //@mark(fBar, "Bar"), symbol("Bar", "Bar", "Field", "Foo", "Bar")
	baz  string    //@symbol("baz", "baz", "Field", "Foo", "baz")
}

type Quux struct { //@symbol("Quux", "Quux", "Struct", "", "Quux")
	X, Y float64 //@mark(qX, "X"), symbol("X", "X", "Field", "Quux", "X"), symbol("Y", "Y", "Field", "Quux", "Y")
}

func (f Foo) Baz() string { //@symbol("(Foo).Baz", "Baz", "Method", "", "Baz")
	return f.baz
}

func (q *Quux) Do() {} //@mark(qDo, "Do"), symbol("(*Quux).Do", "Do", "Method", "", "Do")

func main() { //@symbol("main", "main", "Function", "", "main")

}

type Stringer interface { //@symbol("Stringer", "Stringer", "Interface", "", "Stringer")
	String() string //@symbol("String", "String", "Method", "Stringer", "String")
}

type ABer interface { //@mark(ABerInterface, "ABer"), symbol("ABer", "ABer", "Interface", "", "ABer")
	B()        //@symbol("B", "B", "Method", "ABer", "B")
	A() string //@mark(ABerA, "A"), symbol("A", "A", "Method", "ABer", "A")
}

type WithEmbeddeds interface { //@symbol("WithEmbeddeds", "WithEmbeddeds", "Interface", "", "WithEmbeddeds")
	Do()      //@symbol("Do", "Do", "Method", "WithEmbeddeds", "Do")
	ABer      //@symbol("ABer", "ABer", "Interface", "WithEmbeddeds", "ABer")
	io.Writer //@mark(ioWriter, "io.Writer"), symbol("io.Writer", "io.Writer", "Interface", "WithEmbeddeds", "Writer")
}

func Dunk() int { return 0 } //@symbol("Dunk", "Dunk", "Function", "", "Dunk")

func dunk() {} //@symbol("dunk", "dunk", "Function", "", "dunk")
