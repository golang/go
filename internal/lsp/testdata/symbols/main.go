package main

import "io"

var x = 42 //@symbol("x", "x", 13, "")

const y = 43 //@symbol("y", "y", 14, "")

type Number int //@symbol("Number", "Number", 16, "")

type Alias = string //@symbol("Alias", "Alias", 15, "")

type NumberAlias = Number //@symbol("NumberAlias", "NumberAlias", 16, "")

type (
	Boolean   bool   //@symbol("Boolean", "Boolean", 17, "")
	BoolAlias = bool //@symbol("BoolAlias", "BoolAlias", 17, "")
)

type Foo struct { //@symbol("Foo", "Foo", 23, "")
	Quux           //@symbol("Quux", "Quux", 8, "Foo")
	W    io.Writer //@symbol("W" , "W", 8, "Foo")
	Bar  int       //@symbol("Bar", "Bar", 8, "Foo")
	baz  string    //@symbol("baz", "baz", 8, "Foo")
}

type Quux struct { //@symbol("Quux", "Quux", 23, "")
	X, Y float64 //@symbol("X", "X", 8, "Quux"), symbol("Y", "Y", 8, "Quux")
}

func (f Foo) Baz() string { //@symbol("Baz", "Baz", 6, "Foo")
	return f.baz
}

func (q *Quux) Do() {} //@symbol("Do", "Do", 6, "Quux")

func main() { //@symbol("main", "main", 12, "")

}

type Stringer interface { //@symbol("Stringer", "Stringer", 11, "")
	String() string
}
