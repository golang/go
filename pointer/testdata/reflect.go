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
	print(reflect.NewAt(reflect.TypeOf(3), unsafe.Pointer(&x)).Interface()) // @types *int
}

// @warning "unsound: main.reflectNewAt contains a reflect.NewAt.. call"

func reflectTypeOf() {
	t := reflect.TypeOf(3)
	if unknown {
		t = reflect.TypeOf("foo")
	}
	print(t)                             // @types *reflect.rtype
	print(reflect.Zero(t).Interface())   // @types int | string
	newint := reflect.New(t).Interface() // @line rtonew
	print(newint)                        // @types *int | *string
	print(newint.(*int))                 // @pointsto reflectAlloc@rtonew:23
	print(newint.(*string))              // @pointsto reflectAlloc@rtonew:23
}

func reflectTypeElem() {
	print(reflect.Zero(reflect.TypeOf(&a).Elem()).Interface())                       // @types int
	print(reflect.Zero(reflect.TypeOf([]string{}).Elem()).Interface())               // @types string
	print(reflect.Zero(reflect.TypeOf(make(chan bool)).Elem()).Interface())          // @types bool
	print(reflect.Zero(reflect.TypeOf(make(map[string]float64)).Elem()).Interface()) // @types float64
	print(reflect.Zero(reflect.TypeOf([3]complex64{}).Elem()).Interface())           // @types complex64
	print(reflect.Zero(reflect.TypeOf(3).Elem()).Interface())                        // @types
}

func reflectTypeInOut() {
	var f func(float64, bool) (string, int)
	print(reflect.Zero(reflect.TypeOf(f).In(0)).Interface())    // @types float64
	print(reflect.Zero(reflect.TypeOf(f).In(1)).Interface())    // @types bool
	print(reflect.Zero(reflect.TypeOf(f).In(-1)).Interface())   // @types float64 | bool
	print(reflect.Zero(reflect.TypeOf(f).In(zero)).Interface()) // @types float64 | bool

	print(reflect.Zero(reflect.TypeOf(f).Out(0)).Interface()) // @types string
	print(reflect.Zero(reflect.TypeOf(f).Out(1)).Interface()) // @types int
	print(reflect.Zero(reflect.TypeOf(f).Out(2)).Interface()) // @types string | int
	print(reflect.Zero(reflect.TypeOf(3).Out(0)).Interface()) // @types
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
