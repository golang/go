//go:build ignore
// +build ignore

package main

// Test of maps with reflection.

import "reflect"

var a int
var b bool

func reflectMapKeysIndex() {
	m := make(map[*int]*bool) // @line mr1make
	m[&a] = &b

	mrv := reflect.ValueOf(m)
	print(mrv.Interface())                  // @types map[*int]*bool
	print(mrv.Interface().(map[*int]*bool)) // @pointsto makemap@mr1make:11
	print(mrv)                              // @pointsto makeinterface:map[*int]*bool
	print(mrv)                              // @types map[*int]*bool

	keys := mrv.MapKeys()
	print(keys) // @pointsto <alloc in (reflect.Value).MapKeys>
	for _, k := range keys {
		print(k)                    // @pointsto <alloc in (reflect.Value).MapKeys>
		print(k)                    // @types *int
		print(k.Interface())        // @types *int
		print(k.Interface().(*int)) // @pointsto command-line-arguments.a

		v := mrv.MapIndex(k)
		print(v.Interface())         // @types *bool
		print(v.Interface().(*bool)) // @pointsto command-line-arguments.b
	}
}

func reflectSetMapIndex() {
	m := make(map[*int]*bool)
	mrv := reflect.ValueOf(m)
	mrv.SetMapIndex(reflect.ValueOf(&a), reflect.ValueOf(&b))

	print(m[nil]) // @pointsto command-line-arguments.b

	for _, k := range mrv.MapKeys() {
		print(k.Interface())        // @types *int
		print(k.Interface().(*int)) // @pointsto command-line-arguments.a
	}

	tmap := reflect.TypeOf(m)
	// types.EvalNode won't let us refer to non-exported types:
	// print(tmap) // #@types *reflect.rtype
	print(tmap) // @pointsto map[*int]*bool

	zmap := reflect.Zero(tmap)
	print(zmap)             // @pointsto <alloc in reflect.Zero>
	print(zmap.Interface()) // @pointsto <alloc in reflect.Zero>

	print(tmap.Key())                            // @pointsto *int
	print(tmap.Elem())                           // @pointsto *bool
	print(reflect.Zero(tmap.Key()))              // @pointsto <alloc in reflect.Zero>
	print(reflect.Zero(tmap.Key()).Interface())  // @pointsto <alloc in reflect.Zero>
	print(reflect.Zero(tmap.Key()).Interface())  // @types *int
	print(reflect.Zero(tmap.Elem()))             // @pointsto <alloc in reflect.Zero>
	print(reflect.Zero(tmap.Elem()).Interface()) // @pointsto <alloc in reflect.Zero>
	print(reflect.Zero(tmap.Elem()).Interface()) // @types *bool
}

func reflectSetMapIndexInterface() {
	// Exercises reflect.Value conversions to/from interfaces:
	// a different code path than for concrete types.
	m := make(map[interface{}]interface{})
	reflect.ValueOf(m).SetMapIndex(reflect.ValueOf(&a), reflect.ValueOf(&b))
	for k, v := range m {
		print(k)         // @types *int
		print(k.(*int))  // @pointsto command-line-arguments.a
		print(v)         // @types *bool
		print(v.(*bool)) // @pointsto command-line-arguments.b
	}
}

func reflectSetMapIndexAssignable() {
	// SetMapIndex performs implicit assignability conversions.
	type I *int
	type J *int

	str := reflect.ValueOf("")

	// *int is assignable to I.
	m1 := make(map[string]I)
	reflect.ValueOf(m1).SetMapIndex(str, reflect.ValueOf(new(int))) // @line int
	print(m1[""])                                                   // @pointsto new@int:58

	// I is assignable to I.
	m2 := make(map[string]I)
	reflect.ValueOf(m2).SetMapIndex(str, reflect.ValueOf(I(new(int)))) // @line I
	print(m2[""])                                                      // @pointsto new@I:60

	// J is not assignable to I.
	m3 := make(map[string]I)
	reflect.ValueOf(m3).SetMapIndex(str, reflect.ValueOf(J(new(int))))
	print(m3[""]) // @pointsto
}

func reflectMakeMap() {
	t := reflect.TypeOf(map[*int]*bool(nil))
	v := reflect.MakeMap(t)
	print(v) // @types map[*int]*bool
	print(v) // @pointsto <alloc in reflect.MakeMap>
}

func main() {
	reflectMapKeysIndex()
	reflectSetMapIndex()
	reflectSetMapIndexInterface()
	reflectSetMapIndexAssignable()
	reflectMakeMap()
	// TODO(adonovan): reflect.MapOf(Type)
}
