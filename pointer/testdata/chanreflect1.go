// +build ignore

package main

import "reflect"

//
// This test is very sensitive to line-number perturbations!

// Test of channels with reflection.

var a, b int

func chanreflect1() {
	ch := make(chan *int, 0)
	crv := reflect.ValueOf(ch)
	crv.Send(reflect.ValueOf(&a))
	print(crv.Interface())             // @types chan *int
	print(crv.Interface().(chan *int)) // @pointsto makechan@testdata/chanreflect.go:15:12
	print(<-ch)                        // @pointsto main.a
}

func chanreflect2() {
	ch := make(chan *int, 0)
	ch <- &b
	crv := reflect.ValueOf(ch)
	r, _ := crv.Recv()
	print(r.Interface())        // @types *int
	print(r.Interface().(*int)) // @pointsto main.b
}

func main() {
	chanreflect1()
	chanreflect2()
}
