package describe // @describe pkgdecl "describe"

// Tests of 'describe' query.
// See go.tools/guru/guru_test.go for explanation.
// See describe.golden for expected query results.

// TODO(adonovan): more coverage of the (extensive) logic.

import (
	"lib"
	"nosuchpkg"            // @describe badimport1 "nosuchpkg"
	nosuchpkg2 "nosuchpkg" // @describe badimport2 "nosuchpkg2"
	// The unsafe package changed in Go 1.17 with the addition of
	// unsafe.Add and unsafe.Slice. While we still support older versions
	// of Go, the test case below cannot be enabled.
	// _ "unsafe"             // @describe unsafe "unsafe"
)

var _ nosuchpkg.T
var _ nosuchpkg2.T

type cake float64 // @describe type-ref-builtin "float64"

const c = iota // @describe const-ref-iota "iota"

const pi = 3.141     // @describe const-def-pi "pi"
const pie = cake(pi) // @describe const-def-pie "pie"
const _ = pi         // @describe const-ref-pi "pi"

var global = new(string) // NB: ssa.Global is indirect, i.e. **string

func main() { // @describe func-def-main "main"
	// func objects
	_ = main      // @describe func-ref-main "main"
	_ = (*C).f    // @describe func-ref-*C.f "..C..f"
	_ = D.f       // @describe func-ref-D.f "D.f"
	_ = I.f       // @describe func-ref-I.f "I.f"
	var d D       // @describe type-D "D"
	var i I       // @describe type-I "I"
	_ = d.f       // @describe func-ref-d.f "d.f"
	_ = i.f       // @describe func-ref-i.f "i.f"
	var slice []D // @describe slice-of-D "slice"

	var dptr *D // @describe ptr-with-nonptr-methods "dptr"
	_ = dptr

	// var objects
	anon := func() {
		_ = d // @describe ref-lexical-d "d"
	}
	_ = anon   // @describe ref-anon "anon"
	_ = global // @describe ref-global "global"

	// SSA affords some local flow sensitivity.
	var a, b int
	var x = &a // @describe var-def-x-1 "x"
	_ = x      // @describe var-ref-x-1 "x"
	x = &b     // @describe var-def-x-2 "x"
	_ = x      // @describe var-ref-x-2 "x"

	i = new(C) // @describe var-ref-i-C "i"
	if i != nil {
		i = D{} // @describe var-ref-i-D "i"
	}
	print(i) // @describe var-ref-i "\\bi\\b"

	// const objects
	const localpi = 3.141     // @describe const-local-pi "localpi"
	const localpie = cake(pi) // @describe const-local-pie "localpie"
	const _ = localpi         // @describe const-ref-localpi "localpi"

	// type objects
	type T int      // @describe type-def-T "T"
	var three T = 3 // @describe type-ref-T "T"
	_ = three

	print(1 + 2*3)        // @describe const-expr " 2.3"
	print(real(1+2i) - 3) // @describe const-expr2 "real.*3"

	m := map[string]*int{"a": &a}
	mapval, _ := m["a"] // @describe map-lookup,ok "m..a.."
	_ = mapval          // @describe mapval "mapval"
	_ = m               // @describe m "m"

	defer main() // @describe defer-stmt "defer"
	go main()    // @describe go-stmt "go"

	panic(3) // @describe builtin-ref-panic "panic"

	var a2 int // @describe var-decl-stmt "var a2 int"
	_ = a2
	var _ int // @describe var-decl-stmt2 "var _ int"
	var _ int // @describe var-def-blank "_"

	var _ lib.Outer // @describe lib-outer "Outer"

	var mmm map[C]D // @describe var-map-of-C-D "mmm"

	d := newD().ThirdField // @describe field-access "ThirdField"

	astCopy := ast
	unknown() // @describe call-unknown "\\("
}

type I interface { // @describe def-iface-I "I"
	f() // @describe def-imethod-I.f "f"
}

type C int
type D struct {
	Field        int
	AnotherField string
	ThirdField   C
}

func (c *C) f() {}
func (d D) f()  {}

func newD() D { return D{} }
