// +build ignore

package main

import "reflect"

var a, b int

type A struct {
	f *int
	g interface{}
	h bool
}

func structReflect1() {
	var a A
	fld, _ := reflect.TypeOf(a).FieldByName("f") // "f" is ignored
	// TODO(adonovan): what does interface{} even mean here?
	print(reflect.Zero(fld.Type).Interface()) // @types *int | bool | interface{}
	// TODO(adonovan): test promotion/embedding.
}

func main() {
	structReflect1()
}
