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
		print(k.Interface().(*int)) // @pointsto main.a

		v := mrv.MapIndex(k)
		print(v.Interface())         // @types *bool
		print(v.Interface().(*bool)) // @pointsto main.b
	}
}

func reflectSetMapIndex() {
	m := make(map[*int]*bool)
	mrv := reflect.ValueOf(m)
	mrv.SetMapIndex(reflect.ValueOf(&a), reflect.ValueOf(&b))

	print(m[nil]) // @pointsto main.b

	for _, k := range mrv.MapKeys() {
		print(k.Interface())        // @types *int
		print(k.Interface().(*int)) // @pointsto main.a
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

func reflectMakeMap() {
	t := reflect.TypeOf(map[*int]*bool(nil))
	v := reflect.MakeMap(t)
	print(v) // @types map[*int]*bool
	print(v) // @pointsto <alloc in reflect.MakeMap>
}

func main() {
	reflectMapKeysIndex()
	reflectSetMapIndex()
	reflectMakeMap()
	// TODO(adonovan): reflect.MapOf(Type)
}
