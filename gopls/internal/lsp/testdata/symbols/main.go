package main

import (
	"io"
)

// Each symbol marker in this file defines the following information:
//  symbol(name, selectionSpan, kind, detail, id, parentID)
//    - name: DocumentSymbol.Name
//    - selectionSpan: DocumentSymbol.SelectionRange
//    - kind: DocumentSymbol.Kind
//    - detail: DocumentSymbol.Detail
//    - id: if non-empty, a unique identifier for this symbol
//    - parentID: if non-empty, the id of the parent of this symbol
//
// This data in aggregate defines a set of document symbols and their
// parent-child relationships, which is compared against the DocummentSymbols
// response from gopls for the current file.
//
// TODO(rfindley): the symbol annotations here are complicated and difficult to
// maintain. It would be simpler to just write out the full expected response
// in the golden file, perhaps as raw JSON.

var _ = 1

var x = 42 //@symbol("x", "x", "Variable", "", "", "")

var nested struct { //@symbol("nested", "nested", "Variable", "struct{...}", "nested", "")
	nestedField struct { //@symbol("nestedField", "nestedField", "Field", "struct{...}", "nestedField", "nested")
		f int //@symbol("f", "f", "Field", "int", "", "nestedField")
	}
}

const y = 43 //@symbol("y", "y", "Constant", "", "", "")

type Number int //@symbol("Number", "Number", "Class", "int", "", "")

type Alias = string //@symbol("Alias", "Alias", "Class", "string", "", "")

type NumberAlias = Number //@symbol("NumberAlias", "NumberAlias", "Class", "Number", "", "")

type (
	Boolean   bool   //@symbol("Boolean", "Boolean", "Class", "bool", "", "")
	BoolAlias = bool //@symbol("BoolAlias", "BoolAlias", "Class", "bool", "", "")
)

type Foo struct { //@symbol("Foo", "Foo", "Struct", "struct{...}", "Foo", "")
	Quux                    //@symbol("Quux", "Quux", "Field", "Quux", "", "Foo")
	W         io.Writer     //@symbol("W", "W", "Field", "io.Writer", "", "Foo")
	Bar       int           //@symbol("Bar", "Bar", "Field", "int", "", "Foo")
	baz       string        //@symbol("baz", "baz", "Field", "string", "", "Foo")
	funcField func(int) int //@symbol("funcField", "funcField", "Field", "func(int) int", "", "Foo")
}

type Quux struct { //@symbol("Quux", "Quux", "Struct", "struct{...}", "Quux", "")
	X, Y float64 //@symbol("X", "X", "Field", "float64", "", "Quux"), symbol("Y", "Y", "Field", "float64", "", "Quux")
}

type EmptyStruct struct{} //@symbol("EmptyStruct", "EmptyStruct", "Struct", "struct{}", "", "")

func (f Foo) Baz() string { //@symbol("(Foo).Baz", "Baz", "Method", "func() string", "", "")
	return f.baz
}

func _() {}

func (q *Quux) Do() {} //@symbol("(*Quux).Do", "Do", "Method", "func()", "", "")

func main() { //@symbol("main", "main", "Function", "func()", "", "")
}

type Stringer interface { //@symbol("Stringer", "Stringer", "Interface", "interface{...}", "Stringer", "")
	String() string //@symbol("String", "String", "Method", "func() string", "", "Stringer")
}

type ABer interface { //@symbol("ABer", "ABer", "Interface", "interface{...}", "ABer", "")
	B()        //@symbol("B", "B", "Method", "func()", "", "ABer")
	A() string //@symbol("A", "A", "Method", "func() string", "", "ABer")
}

type WithEmbeddeds interface { //@symbol("WithEmbeddeds", "WithEmbeddeds", "Interface", "interface{...}", "WithEmbeddeds", "")
	Do()      //@symbol("Do", "Do", "Method", "func()", "", "WithEmbeddeds")
	ABer      //@symbol("ABer", "ABer", "Field", "ABer", "", "WithEmbeddeds")
	io.Writer //@symbol("Writer", "Writer", "Field", "io.Writer", "", "WithEmbeddeds")
}

type EmptyInterface interface{} //@symbol("EmptyInterface", "EmptyInterface", "Interface", "interface{}", "", "")

func Dunk() int { return 0 } //@symbol("Dunk", "Dunk", "Function", "func() int", "", "")

func dunk() {} //@symbol("dunk", "dunk", "Function", "func()", "", "")
