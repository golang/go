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

var globalVar int //@ ssa(globalVar,"&Global")

func globalFunc() {}

type I interface {
	interfaceMethod()
}

type S struct {
	x int //@ ssa(x,"nil")
}

func main() {
	print(globalVar) //@ ssa(globalVar,"UnOp")
	globalVar = 1    //@ ssa(globalVar,"Const")

	var v0 int = 1 //@ ssa(v0,"Const") // simple local value spec
	if v0 > 0 {    //@ ssa(v0,"Const")
		v0 = 2 //@ ssa(v0,"Const")
	}
	print(v0) //@ ssa(v0,"Phi")

	// v1 is captured and thus implicitly address-taken.
	var v1 int = 1         //@ ssa(v1,"Const")
	v1 = 2                 //@ ssa(v1,"Const")
	fmt.Println(v1)        //@ ssa(v1,"UnOp") // load
	f := func(param int) { //@ ssa(f,"MakeClosure"), ssa(param,"Parameter")
		if y := 1; y > 0 { //@ ssa(y,"Const")
			print(v1, param) //@ ssa(v1,"UnOp") /*load*/, ssa(param,"Parameter")
		}
		param = 2      //@ ssa(param,"Const")
		println(param) //@ ssa(param,"Const")
	}

	f(0) //@ ssa(f,"MakeClosure")

	var v2 int //@ ssa(v2,"Const") // implicitly zero-initialized local value spec
	print(v2)  //@ ssa(v2,"Const")

	m := make(map[string]int) //@ ssa(m,"MakeMap")

	// Local value spec with multi-valued RHS:
	var v3, v4 = m[""] //@ ssa(v3,"Extract"), ssa(v4,"Extract"), ssa(m,"MakeMap")
	print(v3)          //@ ssa(v3,"Extract")
	print(v4)          //@ ssa(v4,"Extract")

	v3++    //@ ssa(v3,"BinOp") // assign with op
	v3 += 2 //@ ssa(v3,"BinOp") // assign with op

	v5, v6 := false, "" //@ ssa(v5,"Const"), ssa(v6,"Const") // defining assignment
	print(v5)           //@ ssa(v5,"Const")
	print(v6)           //@ ssa(v6,"Const")

	var v7 S    //@ ssa(v7,"&Alloc")
	v7.x = 1    //@ ssa(v7,"&Alloc"), ssa(x,"&FieldAddr")
	print(v7.x) //@ ssa(v7,"&Alloc"), ssa(x,"&FieldAddr")

	var v8 [1]int //@ ssa(v8,"&Alloc")
	v8[0] = 0     //@ ssa(v8,"&Alloc")
	print(v8[:])  //@ ssa(v8,"&Alloc")
	_ = v8[0]     //@ ssa(v8,"&Alloc")
	_ = v8[:][0]  //@ ssa(v8,"&Alloc")
	v8ptr := &v8  //@ ssa(v8ptr,"Alloc"), ssa(v8,"&Alloc")
	_ = v8ptr[0]  //@ ssa(v8ptr,"Alloc")
	_ = *v8ptr    //@ ssa(v8ptr,"Alloc")

	v8a := make([]int, 1) //@ ssa(v8a,"Slice")
	v8a[0] = 0            //@ ssa(v8a,"Slice")
	print(v8a[:])         //@ ssa(v8a,"Slice")

	v9 := S{} //@ ssa(v9,"&Alloc")

	v10 := &v9 //@ ssa(v10,"Alloc"), ssa(v9,"&Alloc")
	_ = v10    //@ ssa(v10,"Alloc")

	var v11 *J = nil //@ ssa(v11,"Const")
	v11.method()     //@ ssa(v11,"Const")

	var v12 J    //@ ssa(v12,"&Alloc")
	v12.method() //@ ssa(v12,"&Alloc") // implicitly address-taken

	// NB, in the following, 'method' resolves to the *types.Func
	// of (*J).method, so it doesn't help us locate the specific
	// ssa.Values here: a bound-method closure and a promotion
	// wrapper.
	_ = v11.method            //@ ssa(v11,"Const")
	_ = (*struct{ J }).method //@ ssa(J,"nil")

	// These vars are not optimised away.
	if false {
		v13 := 0     //@ ssa(v13,"Const")
		println(v13) //@ ssa(v13,"Const")
	}

	switch x := 1; x { //@ ssa(x,"Const")
	case v0: //@ ssa(v0,"Phi")
	}

	for k, v := range m { //@ ssa(k,"Extract"), ssa(v,"Extract"), ssa(m,"MakeMap")
		_ = k //@ ssa(k,"Extract")
		v++   //@ ssa(v,"BinOp")
	}

	if y := 0; y > 1 { //@ ssa(y,"Const"), ssa(y,"Const")
	}

	var i interface{}      //@ ssa(i,"Const") // nil interface
	i = 1                  //@ ssa(i,"MakeInterface")
	switch i := i.(type) { //@ ssa(i,"MakeInterface"), ssa(i,"MakeInterface")
	case int:
		println(i) //@ ssa(i,"Extract")
	}

	ch := make(chan int) //@ ssa(ch,"MakeChan")
	select {
	case x := <-ch: //@ ssa(x,"UnOp") /*receive*/, ssa(ch,"MakeChan")
		_ = x //@ ssa(x,"UnOp")
	}

	// .Op is an inter-package FieldVal-selection.
	var err os.PathError //@ ssa(err,"&Alloc")
	_ = err.Op           //@ ssa(err,"&Alloc"), ssa(Op,"&FieldAddr")
	_ = &err.Op          //@ ssa(err,"&Alloc"), ssa(Op,"&FieldAddr")

	// Exercise corner-cases of lvalues vs rvalues.
	// (Guessing IsAddr from the 'pointerness' won't cut it here.)
	type N *N
	var n N    //@ ssa(n,"Const")
	n1 := n    //@ ssa(n1,"Const"), ssa(n,"Const")
	n2 := &n1  //@ ssa(n2,"Alloc"), ssa(n1,"&Alloc")
	n3 := *n2  //@ ssa(n3,"UnOp"), ssa(n2,"Alloc")
	n4 := **n3 //@ ssa(n4,"UnOp"), ssa(n3,"UnOp")
	_ = n4     //@ ssa(n4,"UnOp")
}
