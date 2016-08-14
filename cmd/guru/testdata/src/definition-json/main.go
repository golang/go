package definition

// Tests of 'definition' query, -json output.
// See golang.org/x/tools/cmd/guru/guru_test.go for explanation.
// See main.golden for expected query results.

// TODO(adonovan): test: selection of member of same package defined in another file.

import (
	"lib"
	lib2 "lib"
	"nosuchpkg"
)

func main() {
	var _ int // @definition builtin "int"

	var _ undef           // @definition lexical-undef "undef"
	var x lib.T           // @definition lexical-pkgname "lib"
	f()                   // @definition lexical-func "f"
	print(x)              // @definition lexical-var "x"
	if x := ""; x == "" { // @definition lexical-shadowing "x"
	}

	var _ lib.Type     // @definition qualified-type "Type"
	var _ lib.Func     // @definition qualified-func "Func"
	var _ lib.Var      // @definition qualified-var "Var"
	var _ lib.Const    // @definition qualified-const "Const"
	var _ lib2.Type    // @definition qualified-type-renaming "Type"
	var _ lib.Nonesuch // @definition qualified-nomember "Nonesuch"
	var _ nosuchpkg.T  // @definition qualified-nopkg "nosuchpkg"

	var u U
	print(u.field) // @definition select-field "field"
	u.method()     // @definition select-method "method"
}

func f()

type T struct{ field int }

func (T) method()

type U struct{ T }

type V1 struct {
	W // @definition embedded-other-file "W"
}

type V2 struct {
	*W // @definition embedded-other-file-pointer "W"
}

type V3 struct {
	int // @definition embedded-basic "int"
}

type V4 struct {
	*int // @definition embedded-basic-pointer "int"
}

type V5 struct {
	lib.Type // @definition embedded-other-pkg "Type"
}

type V6 struct {
	T // @definition embedded-same-file "T"
}
