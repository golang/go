// +build ignore

package main

import "reflect"

//
// This test is very sensitive to line-number perturbations!

// Test of maps with reflection.

var a int
var b bool

func mapreflect1() {
	m := make(map[*int]*bool)
	m[&a] = &b

	mrv := reflect.ValueOf(m)
	print(mrv.Interface())                  // @concrete map[*int]*bool
	print(mrv.Interface().(map[*int]*bool)) // @pointsto makemap@testdata/mapreflect.go:16:11

	for _, k := range mrv.MapKeys() {
		print(k.Interface())        // @concrete *int
		print(k.Interface().(*int)) // @pointsto main.a

		v := mrv.MapIndex(k)
		print(v.Interface())         // @concrete *bool
		print(v.Interface().(*bool)) // @pointsto main.b
	}
}

func mapreflect2() {
	m := make(map[*int]*bool)
	mrv := reflect.ValueOf(m)
	mrv.SetMapIndex(reflect.ValueOf(&a), reflect.ValueOf(&b))

	print(m[nil]) // @pointsto main.b

	for _, k := range mrv.MapKeys() {
		print(k.Interface())        // @concrete *int
		print(k.Interface().(*int)) // @pointsto main.a
	}

	print(reflect.Zero(reflect.TypeOf(m).Key()).Interface())  // @concrete *int
	print(reflect.Zero(reflect.TypeOf(m).Elem()).Interface()) // @concrete *bool
}

func main() {
	mapreflect1()
	mapreflect2()
}
