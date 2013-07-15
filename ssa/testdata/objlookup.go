//+build ignore

package main

// This file is the input to TestObjValueLookup in source_test.go,
// which ensures that each occurrence of an ident defining or
// referring to a func, var or const object can be mapped to its
// corresponding SSA Value.
//
// For every reference to a var object, we use annotations in comments
// to denote both the expected SSA Value kind, and whether to expect
// its value (x) or its address (&x).
//
// For const and func objects, the results don't vary by reference and
// are always values not addresses, so no annotations are needed.

import "fmt"

type J int

func (*J) method() {}

const globalConst = 0

var globalVar int // &globalVar::Global

func globalFunc() {}

type I interface {
	interfaceMethod() // TODO(adonovan): unimplemented (blacklisted in source_test)
}

type S struct {
	x int
}

func main() {
	var v0 int = 1 // v0::Literal (simple local value spec)
	if v0 > 0 {    // v0::Literal
		v0 = 2 // v0::Literal
	}
	print(v0) // v0::Phi

	// v1 is captured and thus implicitly address-taken.
	var v1 int = 1         // v1::Literal
	v1 = 2                 // v1::Literal
	fmt.Println(v1)        // v1::UnOp (load)
	f := func(param int) { // f::MakeClosure param::Parameter
		if y := 1; y > 0 { // y::Literal
			print(v1, param) // v1::UnOp (load) param::Parameter
		}
		param = 2      // param::Literal
		println(param) // param::Literal
	}

	f(0) // f::MakeClosure

	var v2 int // v2::Literal (implicitly zero-initialized local value spec)
	print(v2)  // v2::Literal

	m := make(map[string]int) // m::MakeMap

	// Local value spec with multi-valued RHS:
	var v3, v4 = m[""] // v3::Extract v4::Extract m::MakeMap
	print(v3)          // v3::Extract
	print(v4)          // v4::Extract

	v3++    // v3::BinOp (assign with op)
	v3 += 2 // v3::BinOp (assign with op)

	v5, v6 := false, "" // v5::Literal v6::Literal (defining assignment)
	print(v5)           // v5::Literal
	print(v6)           // v6::Literal

	var v7 S // v7::UnOp (load from Alloc)
	v7.x = 1 // &v7::Alloc

	var v8 [1]int // v8::UnOp (load from Alloc)
	v8[0] = 0     // &v8::Alloc
	print(v8[:])  // &v8::Alloc
	_ = v8[0]     // v8::UnOp (load from Alloc)
	_ = v8[:][0]  // &v8::Alloc
	v8ptr := &v8  // v8ptr::Alloc &v8::Alloc
	_ = v8ptr[0]  // v8ptr::Alloc
	_ = *v8ptr    // v8ptr::Alloc

	v9 := S{} // &v9::Alloc

	v10 := &v9 // v10::Alloc &v9::Alloc

	var v11 *J = nil // v11::Literal
	v11.method()     // v11::Literal

	var v12 J    // v12::UnOp (load from Alloc)
	v12.method() // &v12::Alloc (implicitly address-taken)

	// These vars are optimised away.
	if false {
		v13 := 0     // v13::nil
		println(v13) // v13::nil
	}

	switch x := 1; x { // x::Literal
	case v0: // v0::Phi
	}

	for k, v := range m { // k::Extract v::Extract m::MakeMap
		v++ // v::BinOp
	}

	if y := 0; y > 1 { // y::Literal y::Literal
	}

	var i interface{}      // i::Literal (nil interface)
	i = 1                  // i::MakeInterface
	switch i := i.(type) { // i::MakeInterface i::MakeInterface
	case int:
		println(i) // i::Extract
	}

	ch := make(chan int) // ch::MakeChan
	select {
	case x := <-ch: // x::UnOp (receive) ch::MakeChan
	}
}
