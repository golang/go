// +build ignore

package main

import "reflect"

type A struct {
	f *int
	g interface{}
	h bool
}

var dyn string

func reflectTypeFieldByName() {
	f, _ := reflect.TypeOf(A{}).FieldByName("f")
	print(f.Type) // @pointsto *int

	g, _ := reflect.TypeOf(A{}).FieldByName("g")
	print(g.Type)               // @pointsto interface{}
	print(reflect.Zero(g.Type)) // @pointsto <alloc in reflect.Zero>
	print(reflect.Zero(g.Type)) // @types interface{}

	// TODO(adonovan): fix: the following should return a zero
	// value of the empty interface (i.e. pts is empty), but that
	// requires fixing the TODO comment in
	// reflectZeroConstraint.solve, which in turn requires that we
	// add a "settable" flag to tagged objects.
	print(reflect.Zero(g.Type).Interface()) // @types interface{}

	h, _ := reflect.TypeOf(A{}).FieldByName("h")
	print(h.Type) // @pointsto bool

	missing, _ := reflect.TypeOf(A{}).FieldByName("missing")
	print(missing.Type) // @pointsto

	dyn, _ := reflect.TypeOf(A{}).FieldByName(dyn)
	print(dyn.Type) // @pointsto *int | bool | interface{}
}

func reflectTypeField() {
	fld := reflect.TypeOf(A{}).Field(0)
	print(fld.Type) // @pointsto *int | bool | interface{}
}

func main() {
	reflectTypeFieldByName()
	reflectTypeField()
}
