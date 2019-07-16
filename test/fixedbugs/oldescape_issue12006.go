// errorcheck -0 -m -l -newescape=false

// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test escape analysis through ... parameters.

package foo

func FooN(vals ...*int) (s int) { // ERROR "FooN vals does not escape"
	for _, v := range vals {
		s += *v
	}
	return s
}

// Append forces heap allocation and copies entries in vals to heap, therefore they escape to heap.
func FooNx(x *int, vals ...*int) (s int) { // ERROR "leaking param: x" "leaking param content: vals"
	vals = append(vals, x)
	return FooN(vals...)
}

var sink []*int

func FooNy(x *int, vals ...*int) (s int) { // ERROR "leaking param: x" "leaking param: vals" "leaking param content: vals"
	vals = append(vals, x)
	sink = vals
	return FooN(vals...)
}

func FooNz(vals ...*int) (s int) { // ERROR "leaking param: vals"
	sink = vals
	return FooN(vals...)
}

func TFooN() {
	for i := 0; i < 1000; i++ {
		var i, j int
		FooN(&i, &j) // ERROR "TFooN ... argument does not escape"
	}
}

func TFooNx() {
	for i := 0; i < 1000; i++ {
		var i, j, k int   // ERROR "moved to heap: i" "moved to heap: j" "moved to heap: k"
		FooNx(&k, &i, &j) // ERROR "TFooNx ... argument does not escape"
	}
}

func TFooNy() {
	for i := 0; i < 1000; i++ {
		var i, j, k int   // ERROR "moved to heap: i" "moved to heap: j" "moved to heap: k"
		FooNy(&k, &i, &j) // ERROR "... argument escapes to heap"
	}
}

func TFooNz() {
	for i := 0; i < 1000; i++ {
		var i, j int  // ERROR "moved to heap: i" "moved to heap: j"
		FooNz(&i, &j) // ERROR "... argument escapes to heap"
	}
}

var isink *int32

func FooI(args ...interface{}) { // ERROR "leaking param content: args"
	for i := 0; i < len(args); i++ {
		switch x := args[i].(type) {
		case nil:
			println("is nil")
		case int32:
			println("is int32")
		case *int32:
			println("is *int32")
			isink = x
		case string:
			println("is string")
		}
	}
}

func TFooI() {
	a := int32(1) // ERROR "moved to heap: a"
	b := "cat"
	c := &a
	FooI(a, b, c) // ERROR "a escapes to heap" "b escapes to heap" "c escapes to heap" "TFooI ... argument does not escape"
}

func FooJ(args ...interface{}) *int32 { // ERROR "leaking param: args to result ~r1 level=1"
	for i := 0; i < len(args); i++ {
		switch x := args[i].(type) {
		case nil:
			println("is nil")
		case int32:
			println("is int32")
		case *int32:
			println("is *int32")
			return x
		case string:
			println("is string")
		}
	}
	return nil
}

func TFooJ1() {
	a := int32(1)
	b := "cat"
	c := &a
	FooJ(a, b, c) // ERROR "TFooJ1 a does not escape" "TFooJ1 b does not escape" "TFooJ1 c does not escape" "TFooJ1 ... argument does not escape"
}

func TFooJ2() {
	a := int32(1) // ERROR "moved to heap: a"
	b := "cat"
	c := &a
	isink = FooJ(a, b, c) // ERROR "a escapes to heap" "b escapes to heap" "c escapes to heap" "TFooJ2 ... argument does not escape"
}

type fakeSlice struct {
	l int
	a *[4]interface{}
}

func FooK(args fakeSlice) *int32 { // ERROR "leaking param: args to result ~r1 level=1"
	for i := 0; i < args.l; i++ {
		switch x := (*args.a)[i].(type) {
		case nil:
			println("is nil")
		case int32:
			println("is int32")
		case *int32:
			println("is *int32")
			return x
		case string:
			println("is string")
		}
	}
	return nil
}

func TFooK2() {
	a := int32(1) // ERROR "moved to heap: a"
	b := "cat"
	c := &a
	fs := fakeSlice{3, &[4]interface{}{a, b, c, nil}} // ERROR "a escapes to heap" "b escapes to heap" "c escapes to heap" "TFooK2 &\[4\]interface {} literal does not escape"
	isink = FooK(fs)
}

func FooL(args []interface{}) *int32 { // ERROR "leaking param: args to result ~r1 level=1"
	for i := 0; i < len(args); i++ {
		switch x := args[i].(type) {
		case nil:
			println("is nil")
		case int32:
			println("is int32")
		case *int32:
			println("is *int32")
			return x
		case string:
			println("is string")
		}
	}
	return nil
}

func TFooL2() {
	a := int32(1) // ERROR "moved to heap: a"
	b := "cat"
	c := &a
	s := []interface{}{a, b, c} // ERROR "a escapes to heap" "b escapes to heap" "c escapes to heap" "TFooL2 \[\]interface {} literal does not escape"
	isink = FooL(s)
}
