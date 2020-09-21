package main

import (
	"io"
)

var x = 42 //@mark(symbolsx, "x"), symbol("x", "x", "Variable", "", "main.x")

const y = 43 //@symbol("y", "y", "Constant", "", "main.y")

type Number int //@symbol("Number", "Number", "Number", "", "main.Number")

type Alias = string //@symbol("Alias", "Alias", "String", "", "main.Alias")

type NumberAlias = Number //@symbol("NumberAlias", "NumberAlias", "Number", "", "main.NumberAlias")

type (
	Boolean   bool   //@symbol("Boolean", "Boolean", "Boolean", "", "main.Boolean")
	BoolAlias = bool //@symbol("BoolAlias", "BoolAlias", "Boolean", "", "main.BoolAlias")
)

type Foo struct { //@mark(symbolsFoo, "Foo"), symbol("Foo", "Foo", "Struct", "", "main.Foo")
	Quux           //@mark(fQuux, "Quux"), symbol("Quux", "Quux", "Field", "Foo", "main.Foo.Quux")
	W    io.Writer //@symbol("W" , "W", "Field", "Foo", "main.Foo.W")
	Bar  int       //@mark(fBar, "Bar"), symbol("Bar", "Bar", "Field", "Foo", "main.Foo.Bar")
	baz  string    //@symbol("baz", "baz", "Field", "Foo", "main.Foo.baz")
}

type Quux struct { //@symbol("Quux", "Quux", "Struct", "", "main.Quux")
	X, Y float64 //@mark(qX, "X"), symbol("X", "X", "Field", "Quux", "main.X"), symbol("Y", "Y", "Field", "Quux", "main.Y")
}

func (f Foo) Baz() string { //@symbol("(Foo).Baz", "Baz", "Method", "", "main.Foo.Baz")
	return f.baz
}

func (q *Quux) Do() {} //@mark(qDo, "Do"), symbol("(*Quux).Do", "Do", "Method", "", "main.Quux.Do")

func main() { //@symbol("main", "main", "Function", "", "main.main")

}

type Stringer interface { //@symbol("Stringer", "Stringer", "Interface", "", "main.Stringer")
	String() string //@symbol("String", "String", "Method", "Stringer", "main.Stringer.String")
}

type ABer interface { //@mark(ABerInterface, "ABer"), symbol("ABer", "ABer", "Interface", "", "main.ABer")
	B()        //@symbol("B", "B", "Method", "ABer", "main.ABer.B")
	A() string //@mark(ABerA, "A"), symbol("A", "A", "Method", "ABer", "main.ABer.A")
}

type WithEmbeddeds interface { //@symbol("WithEmbeddeds", "WithEmbeddeds", "Interface", "", "main.WithEmbeddeds")
	Do()      //@symbol("Do", "Do", "Method", "WithEmbeddeds", "main.WithEmbeddeds.Do")
	ABer      //@symbol("ABer", "ABer", "Interface", "WithEmbeddeds", "main.WithEmbeddeds.ABer")
	io.Writer //@mark(ioWriter, "io.Writer"), symbol("io.Writer", "io.Writer", "Interface", "WithEmbeddeds", "main.WithEmbeddeds.Writer")
}

func Dunk() int { return 0 } //@symbol("Dunk", "Dunk", "Function", "", "main.Dunk")

func dunk() {} //@symbol("dunk", "dunk", "Function", "", "main.dunk")
