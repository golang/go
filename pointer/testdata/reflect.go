// +build ignore

package main

import "reflect"
import "unsafe"

var a, b int

func reflectIndirect() {
	ptr := &a
	// Pointer:
	print(reflect.Indirect(reflect.ValueOf(&ptr)).Interface().(*int)) // @pointsto main.a
	// Non-pointer:
	print(reflect.Indirect(reflect.ValueOf([]*int{ptr})).Interface().([]*int)[0]) // @pointsto main.a
}

func reflectNewAt() {
	var x [8]byte
	print(reflect.NewAt(reflect.TypeOf(3), unsafe.Pointer(&x)).Interface()) // @concrete *int
}

// @warning "unsound: main.reflectNewAt contains a reflect.NewAt.. call"

func reflectTypeOf() {
	t := reflect.TypeOf(3)
	if unknown {
		t = reflect.TypeOf("foo")
	}
	print(t)                             // @concrete *reflect.rtype
	print(reflect.Zero(t).Interface())   // @concrete int | string
	newint := reflect.New(t).Interface() // @line rtonew
	print(newint)                        // @concrete *int | *string
	print(newint.(*int))                 // @pointsto reflectAlloc@rtonew:23
	print(newint.(*string))              // @pointsto reflectAlloc@rtonew:23
}

func reflectTypeElem() {
	print(reflect.Zero(reflect.TypeOf(&a).Elem()).Interface())                       // @concrete int
	print(reflect.Zero(reflect.TypeOf([]string{}).Elem()).Interface())               // @concrete string
	print(reflect.Zero(reflect.TypeOf(make(chan bool)).Elem()).Interface())          // @concrete bool
	print(reflect.Zero(reflect.TypeOf(make(map[string]float64)).Elem()).Interface()) // @concrete float64
	print(reflect.Zero(reflect.TypeOf([3]complex64{}).Elem()).Interface())           // @concrete complex64
	print(reflect.Zero(reflect.TypeOf(3).Elem()).Interface())                        // @concrete
}

func reflectTypeInOut() {
	var f func(float64, bool) (string, int)
	print(reflect.Zero(reflect.TypeOf(f).In(0)).Interface())    // @concrete float64
	print(reflect.Zero(reflect.TypeOf(f).In(1)).Interface())    // @concrete bool
	print(reflect.Zero(reflect.TypeOf(f).In(-1)).Interface())   // @concrete float64 | bool
	print(reflect.Zero(reflect.TypeOf(f).In(zero)).Interface()) // @concrete float64 | bool

	print(reflect.Zero(reflect.TypeOf(f).Out(0)).Interface()) // @concrete string
	print(reflect.Zero(reflect.TypeOf(f).Out(1)).Interface()) // @concrete int
	print(reflect.Zero(reflect.TypeOf(f).Out(2)).Interface()) // @concrete string | int
	print(reflect.Zero(reflect.TypeOf(3).Out(0)).Interface()) // @concrete
}

func main() {
	reflectIndirect()
	reflectNewAt()
	reflectTypeOf()
	reflectTypeElem()
	reflectTypeInOut()
}

var unknown bool
var zero int
