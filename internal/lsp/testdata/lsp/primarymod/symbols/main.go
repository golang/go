package main

import (
	"io"
)

var x = 42 //@mark(symbolsx, "x"), symbol("x", "x", "Variable", "")

const y = 43 //@symbol("y", "y", "Constant", "")

type Number int //@symbol("Number", "Number", "Number", "")

type Alias = string //@symbol("Alias", "Alias", "String", "")

type NumberAlias = Number //@symbol("NumberAlias", "NumberAlias", "Number", "")

type (
	Boolean   bool   //@symbol("Boolean", "Boolean", "Boolean", "")
	BoolAlias = bool //@symbol("BoolAlias", "BoolAlias", "Boolean", "")
)

type Foo struct { //@mark(symbolsFoo, "Foo"), symbol("Foo", "Foo", "Struct", "")
	Quux           //@mark(fQuux, "Quux"), symbol("Quux", "Quux", "Field", "Foo")
	W    io.Writer //@symbol("W" , "W", "Field", "Foo")
	Bar  int       //@mark(fBar, "Bar"), symbol("Bar", "Bar", "Field", "Foo")
	baz  string    //@symbol("baz", "baz", "Field", "Foo")
}

type Quux struct { //@symbol("Quux", "Quux", "Struct", "")
	X, Y float64 //@mark(qX, "X"), symbol("X", "X", "Field", "Quux"), symbol("Y", "Y", "Field", "Quux")
}

func (f Foo) Baz() string { //@symbol("Baz", "Baz", "Method", "Foo")
	return f.baz
}

func (q *Quux) Do() {} //@mark(qDo, "Do"), symbol("Do", "Do", "Method", "Quux")

func main() { //@symbol("main", "main", "Function", "")

}

type Stringer interface { //@symbol("Stringer", "Stringer", "Interface", "")
	String() string //@symbol("String", "String", "Method", "Stringer")
}

type ABer interface { //@mark(ABerInterface, "ABer"), symbol("ABer", "ABer", "Interface", "")
	B()        //@symbol("B", "B", "Method", "ABer")
	A() string //@mark(ABerA, "A"), symbol("A", "A", "Method", "ABer")
}

type WithEmbeddeds interface { //@symbol("WithEmbeddeds", "WithEmbeddeds", "Interface", "")
	Do()      //@symbol("Do", "Do", "Method", "WithEmbeddeds")
	ABer      //@symbol("ABer", "ABer", "Interface", "WithEmbeddeds")
	io.Writer //@mark(ioWriter, "io.Writer"), symbol("io.Writer", "io.Writer", "Interface", "WithEmbeddeds")
}
