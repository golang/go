// +build ignore

package main

//

import "reflect"

var a, b int

func f(p *int) *int {
	print(p) // @pointsto
	return &b
}

func g(p *bool) {
}

func funcreflect1() {
	rvf := reflect.ValueOf(f)
	res := rvf.Call([]reflect.Value{reflect.ValueOf(&a)})
	print(res[0].Interface())        // @concrete
	print(res[0].Interface().(*int)) // @pointsto
}

// @calls main.funcreflect1 -> main.f

func main() {
	funcreflect1()
}
