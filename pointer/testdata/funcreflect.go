// +build ignore

package main

import "reflect"

var zero, a, b int

// func f(p *int) *int {
// 	print(p) // #@pointsto
// 	return &b
// }

// func g(p *bool) {
// }

// func reflectValueCall() {
// 	rvf := reflect.ValueOf(f)
// 	res := rvf.Call([]reflect.Value{reflect.ValueOf(&a)})
// 	print(res[0].Interface())        // #@types
// 	print(res[0].Interface().(*int)) // #@pointsto
// }

// #@calls main.reflectValueCall -> main.f

func reflectTypeInOut() {
	var f func(float64, bool) (string, int)
	// TODO(adonovan): when the In/Out argument is a valid index constant,
	// only include a single type in the result.  Needs some work.
	print(reflect.Zero(reflect.TypeOf(f).In(0)).Interface())    // @types float64 | bool
	print(reflect.Zero(reflect.TypeOf(f).In(1)).Interface())    // @types float64 | bool
	print(reflect.Zero(reflect.TypeOf(f).In(-1)).Interface())   // @types float64 | bool
	print(reflect.Zero(reflect.TypeOf(f).In(zero)).Interface()) // @types float64 | bool

	print(reflect.Zero(reflect.TypeOf(f).Out(0)).Interface()) // @types string | int
	print(reflect.Zero(reflect.TypeOf(f).Out(1)).Interface()) // @types string | int
	print(reflect.Zero(reflect.TypeOf(f).Out(2)).Interface()) // @types string | int
	print(reflect.Zero(reflect.TypeOf(3).Out(0)).Interface()) // @types
}

func main() {
	//reflectValueCall()
	reflectTypeInOut()
}
