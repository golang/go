// Package refs is a package used to test find references.
package refs

import "os" //@mark(osDecl, `"os"`),refs("os", osDecl, osUse)

type i int //@mark(typeI, "i"),refs("i", typeI, argI, returnI, embeddedI)

type X struct {
	Y int //@mark(typeXY, "Y")
}

func _(_ i) []bool { //@mark(argI, "i")
	return nil
}

func _(_ []byte) i { //@mark(returnI, "i")
	return 0
}

var q string //@mark(declQ, "q"),refs("q", declQ, assignQ, bobQ)

var Q string //@mark(declExpQ, "Q"),refs("Q", declExpQ, assignExpQ, bobExpQ)

func _() {
	q = "hello" //@mark(assignQ, "q")
	bob := func(_ string) {}
	bob(q) //@mark(bobQ, "q")
}

type e struct {
	i //@mark(embeddedI, "i"),refs("i", embeddedI, embeddedIUse)
}

func _() {
	_ = e{}.i //@mark(embeddedIUse, "i")
}

const (
	foo = iota //@refs("iota")
)

func _(x interface{}) {
	// We use the _ prefix because the markers inhabit a single
	// namespace and yDecl is already used in ../highlights/highlights.go.
	switch _y := x.(type) { //@mark(_yDecl, "_y"),refs("_y", _yDecl, _yInt, _yDefault)
	case int:
		println(_y) //@mark(_yInt, "_y"),refs("_y", _yDecl, _yInt, _yDefault)
	default:
		println(_y) //@mark(_yDefault, "_y")
	}

	os.Getwd() //@mark(osUse, "os")
}
