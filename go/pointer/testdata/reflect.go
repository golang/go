//go:build ignore
// +build ignore

package main

import (
	"reflect"
	"unsafe"
)

var a, b int
var unknown bool

func reflectIndirect() {
	ptr := &a
	// Pointer:
	print(reflect.Indirect(reflect.ValueOf(&ptr)).Interface().(*int)) // @pointsto command-line-arguments.a
	// Non-pointer:
	print(reflect.Indirect(reflect.ValueOf([]*int{ptr})).Interface().([]*int)[0]) // @pointsto command-line-arguments.a
}

func reflectNewAt() {
	var x [8]byte
	print(reflect.NewAt(reflect.TypeOf(3), unsafe.Pointer(&x)).Interface()) // @types *int
}

// @warning "unsound: command-line-arguments.reflectNewAt contains a reflect.NewAt.. call"

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
	print(reflect.Zero(reflect.TypeOf(new(interface{})).Elem()))                     // @types interface{}
	print(reflect.Zero(reflect.TypeOf(new(interface{})).Elem()).Interface())         // @types
}

// reflect.Values within reflect.Values.
func metareflection() {
	// "box" a *int twice, unbox it twice.
	v0 := reflect.ValueOf(&a)
	print(v0)                              // @types *int
	v1 := reflect.ValueOf(v0)              // box
	print(v1)                              // @types reflect.Value
	v2 := reflect.ValueOf(v1)              // box
	print(v2)                              // @types reflect.Value
	v1a := v2.Interface().(reflect.Value)  // unbox
	print(v1a)                             // @types reflect.Value
	v0a := v1a.Interface().(reflect.Value) // unbox
	print(v0a)                             // @types *int
	print(v0a.Interface().(*int))          // @pointsto command-line-arguments.a

	// "box" an interface{} lvalue twice, unbox it twice.
	var iface interface{} = 3
	x0 := reflect.ValueOf(&iface).Elem()
	print(x0)                              // @types interface{}
	x1 := reflect.ValueOf(x0)              // box
	print(x1)                              // @types reflect.Value
	x2 := reflect.ValueOf(x1)              // box
	print(x2)                              // @types reflect.Value
	x1a := x2.Interface().(reflect.Value)  // unbox
	print(x1a)                             // @types reflect.Value
	x0a := x1a.Interface().(reflect.Value) // unbox
	print(x0a)                             // @types interface{}
	print(x0a.Interface())                 // @types int
}

type T struct{}

// When the output of a type constructor flows to its input, we must
// bound the set of types created to ensure termination of the algorithm.
func typeCycle() {
	t := reflect.TypeOf(0)
	u := reflect.TypeOf("")
	v := reflect.TypeOf(T{})
	for unknown {
		t = reflect.PtrTo(t)
		t = reflect.SliceOf(t)

		u = reflect.SliceOf(u)

		if unknown {
			v = reflect.ChanOf(reflect.BothDir, v)
		} else {
			v = reflect.PtrTo(v)
		}
	}

	// Type height is bounded to about 4 map/slice/chan/pointer constructors.
	print(reflect.Zero(t).Interface()) // @types int | []*int | []*[]*int
	print(reflect.Zero(u).Interface()) // @types string | []string | [][]string | [][][]string | [][][][]string
	print(reflect.Zero(v).Interface()) // @types T | *T | **T | ***T | ****T | chan T | *chan T | **chan T | chan *T | *chan *T | chan **T | chan ***T | chan chan T | chan *chan T | chan chan *T
}

func main() {
	reflectIndirect()
	reflectNewAt()
	reflectTypeOf()
	reflectTypeElem()
	metareflection()
	typeCycle()
}
