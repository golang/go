// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file contains the code snippets included in "The Laws of Reflection."

package main

import (
	"fmt"
	"reflect"
)

func main() {
	var x float64 = 3.4
	fmt.Println("type:", reflect.TypeOf(x))
	// STOP OMIT
	// TODO(proppy): test output OMIT
}

// STOP main OMIT

func f1() {
	// START f1 OMIT
	var x float64 = 3.4
	v := reflect.ValueOf(x)
	fmt.Println("type:", v.Type())
	fmt.Println("kind is float64:", v.Kind() == reflect.Float64)
	fmt.Println("value:", v.Float())
	// STOP OMIT
}

func f2() {
	// START f2 OMIT
	var x uint8 = 'x'
	v := reflect.ValueOf(x)
	fmt.Println("type:", v.Type())                            // uint8.
	fmt.Println("kind is uint8: ", v.Kind() == reflect.Uint8) // true.
	x = uint8(v.Uint())                                       // v.Uint returns a uint64.
	// STOP OMIT
}

func f3() {
	// START f3 OMIT
	type MyInt int
	var x MyInt = 7
	v := reflect.ValueOf(x)
	// STOP OMIT
	// START f3b OMIT
	y := v.Interface().(float64) // y will have type float64.
	fmt.Println(y)
	// STOP OMIT
	// START f3c OMIT
	fmt.Println(v.Interface())
	// STOP OMIT
	// START f3d OMIT
	fmt.Printf("value is %7.1e\n", v.Interface())
	// STOP OMIT
}

func f4() {
	// START f4 OMIT
	var x float64 = 3.4
	v := reflect.ValueOf(x)
	v.SetFloat(7.1) // Error: will panic.
	// STOP OMIT
}

func f5() {
	// START f5 OMIT
	var x float64 = 3.4
	v := reflect.ValueOf(x)
	fmt.Println("settability of v:", v.CanSet())
	// STOP OMIT
}

func f6() {
	// START f6 OMIT
	var x float64 = 3.4
	v := reflect.ValueOf(x)
	// STOP OMIT
	// START f6b OMIT
	v.SetFloat(7.1)
	// STOP OMIT
}

func f7() {
	// START f7 OMIT
	var x float64 = 3.4
	p := reflect.ValueOf(&x) // Note: take the address of x.
	fmt.Println("type of p:", p.Type())
	fmt.Println("settability of p:", p.CanSet())
	// STOP OMIT
	// START f7b OMIT
	v := p.Elem()
	fmt.Println("settability of v:", v.CanSet())
	// STOP OMIT
	// START f7c OMIT
	v.SetFloat(7.1)
	fmt.Println(v.Interface())
	fmt.Println(x)
	// STOP OMIT
}

func f8() {
	// START f8 OMIT
	type T struct {
		A int
		B string
	}
	t := T{23, "skidoo"}
	s := reflect.ValueOf(&t).Elem()
	typeOfT := s.Type()
	for i := 0; i < s.NumField(); i++ {
		f := s.Field(i)
		fmt.Printf("%d: %s %s = %v\n", i,
			typeOfT.Field(i).Name, f.Type(), f.Interface())
	}
	// STOP OMIT
	// START f8b OMIT
	s.Field(0).SetInt(77)
	s.Field(1).SetString("Sunset Strip")
	fmt.Println("t is now", t)
	// STOP OMIT
}

func f9() {
	// START f9 OMIT
	var x float64 = 3.4
	fmt.Println("value:", reflect.ValueOf(x))
	// STOP OMIT
}
