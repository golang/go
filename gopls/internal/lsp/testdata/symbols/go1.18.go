//go:build go1.18
// +build go1.18

package main

type T[P any] struct { //@symbol("T", "T", "Struct", "struct{...}", "T", "")
	F P //@symbol("F", "F", "Field", "P", "", "T")
}

type Constraint interface { //@symbol("Constraint", "Constraint", "Interface", "interface{...}", "Constraint", "")
	~int | struct{ int } //@symbol("~int | struct{int}", "~int | struct{ int }", "Field", "", "", "Constraint")

	// TODO(rfindley): the selection range below is the entire interface field.
	// Can we reduce it?
	interface{ M() } //@symbol("interface{...}", "interface{ M() }", "Field", "", "iFaceField", "Constraint"), symbol("M", "M", "Method", "func()", "", "iFaceField")
}
