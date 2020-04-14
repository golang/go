package main

import (
	"io"
)

var x = 42 //@mark(symbolsx, "x"), symbol("x", "x", "Variable", "", "golang.org/x/tools/internal/lsp/symbols.x")

const y = 43 //@symbol("y", "y", "Constant", "", "golang.org/x/tools/internal/lsp/symbols.y")

type Number int //@symbol("Number", "Number", "Number", "", "golang.org/x/tools/internal/lsp/symbols.Number")

type Alias = string //@symbol("Alias", "Alias", "String", "", "golang.org/x/tools/internal/lsp/symbols.Alias")

type NumberAlias = Number //@symbol("NumberAlias", "NumberAlias", "Number", "", "golang.org/x/tools/internal/lsp/symbols.NumberAlias")

type (
	Boolean   bool   //@symbol("Boolean", "Boolean", "Boolean", "", "golang.org/x/tools/internal/lsp/symbols.Boolean")
	BoolAlias = bool //@symbol("BoolAlias", "BoolAlias", "Boolean", "", "golang.org/x/tools/internal/lsp/symbols.BoolAlias")
)

type Foo struct { //@mark(symbolsFoo, "Foo"), symbol("Foo", "Foo", "Struct", "", "golang.org/x/tools/internal/lsp/symbols.Foo")
	Quux           //@mark(fQuux, "Quux"), symbol("Quux", "Quux", "Field", "Foo", "golang.org/x/tools/internal/lsp/symbols.Foo.Quux")
	W    io.Writer //@symbol("W" , "W", "Field", "Foo", "golang.org/x/tools/internal/lsp/symbols.Foo.W")
	Bar  int       //@mark(fBar, "Bar"), symbol("Bar", "Bar", "Field", "Foo", "golang.org/x/tools/internal/lsp/symbols.Foo.Bar")
	baz  string    //@symbol("baz", "baz", "Field", "Foo", "golang.org/x/tools/internal/lsp/symbols.Foo.baz")
}

type Quux struct { //@symbol("Quux", "Quux", "Struct", "", "golang.org/x/tools/internal/lsp/symbols.Quux")
	X, Y float64 //@mark(qX, "X"), symbol("X", "X", "Field", "Quux", "golang.org/x/tools/internal/lsp/symbols.X"), symbol("Y", "Y", "Field", "Quux", "golang.org/x/tools/internal/lsp/symbols.Y")
}

func (f Foo) Baz() string { //@symbol("(Foo).Baz", "Baz", "Method", "", "golang.org/x/tools/internal/lsp/symbols.Foo.Baz")
	return f.baz
}

func (q *Quux) Do() {} //@mark(qDo, "Do"), symbol("(*Quux).Do", "Do", "Method", "", "golang.org/x/tools/internal/lsp/symbols.Quux.Do")

func main() { //@symbol("main", "main", "Function", "", "golang.org/x/tools/internal/lsp/symbols.main")

}

type Stringer interface { //@symbol("Stringer", "Stringer", "Interface", "", "golang.org/x/tools/internal/lsp/symbols.Stringer")
	String() string //@symbol("String", "String", "Method", "Stringer", "golang.org/x/tools/internal/lsp/symbols.Stringer.String")
}

type ABer interface { //@mark(ABerInterface, "ABer"), symbol("ABer", "ABer", "Interface", "", "golang.org/x/tools/internal/lsp/symbols.ABer")
	B()        //@symbol("B", "B", "Method", "ABer", "golang.org/x/tools/internal/lsp/symbols.ABer.B")
	A() string //@mark(ABerA, "A"), symbol("A", "A", "Method", "ABer", "golang.org/x/tools/internal/lsp/symbols.ABer.A")
}

type WithEmbeddeds interface { //@symbol("WithEmbeddeds", "WithEmbeddeds", "Interface", "", "golang.org/x/tools/internal/lsp/symbols.WithEmbeddeds")
	Do()      //@symbol("Do", "Do", "Method", "WithEmbeddeds", "golang.org/x/tools/internal/lsp/symbols.WithEmbeddeds.Do")
	ABer      //@symbol("ABer", "ABer", "Interface", "WithEmbeddeds", "golang.org/x/tools/internal/lsp/symbols.WithEmbeddeds.ABer")
	io.Writer //@mark(ioWriter, "io.Writer"), symbol("io.Writer", "io.Writer", "Interface", "WithEmbeddeds", "golang.org/x/tools/internal/lsp/symbols.WithEmbeddeds.Writer")
}

func Dunk() int { return 0 } //@symbol("Dunk", "Dunk", "Function", "", "golang.org/x/tools/internal/lsp/symbols.Dunk")

func dunk() {} //@symbol("dunk", "dunk", "Function", "", "golang.org/x/tools/internal/lsp/symbols.dunk")
