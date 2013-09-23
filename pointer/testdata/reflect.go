// +build ignore

package main

import "reflect"
import "unsafe"

var a, b int
var unknown bool

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

// TODO(adonovan): report the location of the caller, not NewAt.
// #warning "unsound: main.reflectNewAt contains a reflect.NewAt.. call"
// @warning "unsound: reflect.NewAt.. call"

func reflectTypeOf() {
	t := reflect.TypeOf(3)
	if unknown {
		t = reflect.TypeOf("foo")
	}
	// TODO(adonovan): make types.Eval let us refer to unexported types.
	print(t)                             // #@types *reflect.rtype
	print(reflect.Zero(t).Interface())   // @types int | string
	newint := reflect.New(t).Interface() // @line rtonew
	print(newint)                        // @types *int | *string
	print(newint.(*int))                 // @pointsto <alloc in reflect.New>
	print(newint.(*string))              // @pointsto <alloc in reflect.New>
}

func reflectTypeElem() {
	print(reflect.Zero(reflect.TypeOf(&a).Elem()).Interface())                       // @types int
	print(reflect.Zero(reflect.TypeOf([]string{}).Elem()).Interface())               // @types string
	print(reflect.Zero(reflect.TypeOf(make(chan bool)).Elem()).Interface())          // @types bool
	print(reflect.Zero(reflect.TypeOf(make(map[string]float64)).Elem()).Interface()) // @types float64
	print(reflect.Zero(reflect.TypeOf([3]complex64{}).Elem()).Interface())           // @types complex64
	print(reflect.Zero(reflect.TypeOf(3).Elem()).Interface())                        // @types
}

func main() {
	reflectIndirect()
	reflectNewAt()
	reflectTypeOf()
	reflectTypeElem()
}
