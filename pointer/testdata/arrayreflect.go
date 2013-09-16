// +build ignore

package main

import "reflect"

// Test of arrays & slices with reflection.

var a int

func arrayreflect1() {
	sl := make([]*int, 10) // @line ar1make
	sl[0] = &a

	srv := reflect.ValueOf(sl).Slice(0, 0)
	print(srv.Interface())              // @types []*int
	print(srv.Interface().([]*int))     // @pointsto makeslice@ar1make:12
	print(srv.Interface().([]*int)[42]) // @pointsto main.a
}

func arrayreflect2() {
	var arr [10]*int
	sl := arr[:]
	sl[0] = &a

	srv := reflect.ValueOf(sl).Slice(0, 0)
	print(srv.Interface())              // @types []*int
	print(srv.Interface().([]*int))     // pointsto TODO
	print(srv.Interface().([]*int)[42]) // @pointsto main.a
}

func arrayreflect3() {
	srv := reflect.ValueOf("hi").Slice(0, 0)
	print(srv.Interface()) // @types string

	type S string
	srv2 := reflect.ValueOf(S("hi")).Slice(0, 0)
	print(srv2.Interface()) // @types main.S
}

func arrayreflect4() {
	rv1 := reflect.ValueOf("hi")
	rv2 := rv1 // backflow!
	if unknown {
		rv2 = reflect.ValueOf(123)
	}
	// We see backflow through the assignment above causing an
	// imprecise result for rv1.  This is because the SSA builder
	// doesn't yet lift structs (like reflect.Value) into
	// registers so these are all loads/stores to the stack.
	// Under Das's algorithm, the extra indirection results in
	// (undirected) unification not (directed) flow edges.
	// TODO(adonovan): precision: lift aggregates.
	print(rv1.Interface()) // @types string | int
	print(rv2.Interface()) // @types string | int
}

func arrayreflect5() {
	sl1 := make([]byte, 0)
	sl2 := make([]byte, 0)

	srv := reflect.ValueOf(sl1)

	print(srv.Interface())          // @types []byte
	print(srv.Interface().([]byte)) // @pointsto makeslice@testdata/arrayreflect.go:62:13
	print(srv.Bytes())              // @pointsto makeslice@testdata/arrayreflect.go:62:13

	srv2 := reflect.ValueOf(123)
	srv2.SetBytes(sl2)
	print(srv2.Interface())          // @types []byte | int
	print(srv2.Interface().([]byte)) // @pointsto makeslice@testdata/arrayreflect.go:63:13
	print(srv2.Bytes())              // @pointsto makeslice@testdata/arrayreflect.go:63:13
}

func arrayreflect6() {
	sl1 := []*bool{new(bool)}
	sl2 := []*int{&a}

	srv1 := reflect.ValueOf(sl1)
	print(srv1.Index(42).Interface())         // @types *bool
	print(srv1.Index(42).Interface().(*bool)) // @pointsto alloc@testdata/arrayreflect.go:79:20

	srv2 := reflect.ValueOf(sl2)
	print(srv2.Index(42).Interface())        // @types *int
	print(srv2.Index(42).Interface().(*int)) // @pointsto main.a

	p1 := &sl1[0]
	p2 := &sl2[0]

	prv1 := reflect.ValueOf(p1)
	print(prv1.Elem().Interface())         // @types *bool
	print(prv1.Elem().Interface().(*bool)) // @pointsto alloc@testdata/arrayreflect.go:79:20

	prv2 := reflect.ValueOf(p2)
	print(prv2.Elem().Interface())        // @types *int
	print(prv2.Elem().Interface().(*int)) // @pointsto main.a
}

func main() {
	arrayreflect1()
	arrayreflect2()
	arrayreflect3()
	arrayreflect4()
	arrayreflect5()
	arrayreflect6()
}

var unknown bool
