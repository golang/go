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
// are always values not addresses, so no annotations are needed.  The
// declaration is enough.

import "fmt"
import "os"

type J int

func (*J) method() {}

const globalConst = 0

var globalVar int // &globalVar::Global

func globalFunc() {}

type I interface {
	interfaceMethod()
}

type S struct {
	x int // x::nil
}

func main() {
	print(globalVar) // globalVar::UnOp
	globalVar = 1    // globalVar::Const

	var v0 int = 1 // v0::Const (simple local value spec)
	if v0 > 0 {    // v0::Const
		v0 = 2 // v0::Const
	}
	print(v0) // v0::Phi

	// v1 is captured and thus implicitly address-taken.
	var v1 int = 1         // v1::Const
	v1 = 2                 // v1::Const
	fmt.Println(v1)        // v1::UnOp (load)
	f := func(param int) { // f::MakeClosure param::Parameter
		if y := 1; y > 0 { // y::Const
			print(v1, param) // v1::UnOp (load) param::Parameter
		}
		param = 2      // param::Const
		println(param) // param::Const
	}

	f(0) // f::MakeClosure

	var v2 int // v2::Const (implicitly zero-initialized local value spec)
	print(v2)  // v2::Const

	m := make(map[string]int) // m::MakeMap

	// Local value spec with multi-valued RHS:
	var v3, v4 = m[""] // v3::Extract v4::Extract m::MakeMap
	print(v3)          // v3::Extract
	print(v4)          // v4::Extract

	v3++    // v3::BinOp (assign with op)
	v3 += 2 // v3::BinOp (assign with op)

	v5, v6 := false, "" // v5::Const v6::Const (defining assignment)
	print(v5)           // v5::Const
	print(v6)           // v6::Const

	var v7 S    // &v7::Alloc
	v7.x = 1    // &v7::Alloc x::Const
	print(v7.x) // v7::UnOp x::Field

	var v8 [1]int // &v8::Alloc
	v8[0] = 0     // &v8::Alloc
	print(v8[:])  // &v8::Alloc
	_ = v8[0]     // v8::UnOp (load from Alloc)
	_ = v8[:][0]  // &v8::Alloc
	v8ptr := &v8  // v8ptr::Alloc &v8::Alloc
	_ = v8ptr[0]  // v8ptr::Alloc
	_ = *v8ptr    // v8ptr::Alloc

	v8a := make([]int, 1) // v8a::MakeSlice
	v8a[0] = 0            // v8a::MakeSlice
	print(v8a[:])         // v8a::MakeSlice

	v9 := S{} // &v9::Alloc

	v10 := &v9 // v10::Alloc &v9::Alloc
	_ = v10    // v10::Alloc

	var v11 *J = nil // v11::Const
	v11.method()     // v11::Const

	var v12 J    // &v12::Alloc
	v12.method() // &v12::Alloc (implicitly address-taken)

	// NB, in the following, 'method' resolves to the *types.Func
	// of (*J).method, so it doesn't help us locate the specific
	// ssa.Values here: a bound-method closure and a promotion
	// wrapper.
	_ = v11.method // v11::Const
	_ = (*struct{ J }).method

	// These vars are optimised away.
	if false {
		v13 := 0     // v13::nil
		println(v13) // v13::nil
	}

	switch x := 1; x { // x::Const
	case v0: // v0::Phi
	}

	for k, v := range m { // k::Extract v::Extract m::MakeMap
		_ = k // k::Extract
		v++   // v::BinOp
	}

	if y := 0; y > 1 { // y::Const y::Const
	}

	var i interface{}      // i::Const (nil interface)
	i = 1                  // i::MakeInterface
	switch i := i.(type) { // i::MakeInterface i::MakeInterface
	case int:
		println(i) // i::Extract
	}

	ch := make(chan int) // ch::MakeChan
	select {
	case x := <-ch: // x::UnOp (receive) ch::MakeChan
		_ = x // x::UnOp
	}

	// .Op is an inter-package FieldVal-selection.
	var err os.PathError // &err::Alloc
	_ = err.Op           // err::UnOp Op::Field
	_ = &err.Op          // &err::Alloc &Op::FieldAddr

	// Exercise corner-cases of lvalues vs rvalues.
	// (Guessing IsAddr from the 'pointerness' won't cut it here.)
	type N *N
	var n N    // n::Const
	n1 := n    // n1::Const n::Const
	n2 := &n1  // n2::Alloc &n1::Alloc
	n3 := *n2  // n3::UnOp n2::Alloc
	n4 := **n3 // n4::UnOp n3::UnOp
	_ = n4     // n4::UnOp
}
