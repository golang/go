package importedcomplit

import (
	"golang.org/x/tools/internal/lsp/foo"
)

func _() {
	var V int //@item(icVVar, "V", "int", "var")
	_ = foo.StructFoo{V} //@complete("}", Value, icVVar)
}

func _() {
	var (
		aa string //@item(icAAVar, "aa", "string", "var")
		ab int    //@item(icABVar, "ab", "int", "var")
	)

	_ = foo.StructFoo{a} //@complete("}", abVar, aaVar)

	var s struct {
		AA string //@item(icFieldAA, "AA", "string", "field")
		AB int    //@item(icFieldAB, "AB", "int", "field")
	}

	_ = foo.StructFoo{s.} //@complete("}", icFieldAB, icFieldAA)
}
