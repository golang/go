// +build ignore

package main

// Test of arrays & slices with reflection.

import "reflect"

var a, b int

type S string

func reflectValueSlice() {
	// reflect.Value contains a slice.
	slice := make([]*int, 10) // @line slice
	slice[0] = &a
	rvsl := reflect.ValueOf(slice).Slice(0, 0)
	print(rvsl.Interface())              // @types []*int
	print(rvsl.Interface().([]*int))     // @pointsto makeslice@slice:15
	print(rvsl.Interface().([]*int)[42]) // @pointsto main.a

	// reflect.Value contains an array (non-addressable).
	array := [10]*int{&a} // @line array
	rvarray := reflect.ValueOf(array).Slice(0, 0)
	print(rvarray.Interface())              // @types
	print(rvarray.Interface().([]*int))     // @pointsto
	print(rvarray.Interface().([]*int)[42]) // @pointsto

	// reflect.Value contains a pointer-to-array
	rvparray := reflect.ValueOf(&array).Slice(0, 0)
	print(rvparray.Interface())              // @types []*int
	print(rvparray.Interface().([]*int))     // @pointsto array@array:2
	print(rvparray.Interface().([]*int)[42]) // @pointsto main.a

	// reflect.Value contains a string.
	rvstring := reflect.ValueOf("hi").Slice(0, 0)
	print(rvstring.Interface()) // @types string

	// reflect.Value contains a (named) string type.
	rvS := reflect.ValueOf(S("hi")).Slice(0, 0)
	print(rvS.Interface()) // @types S

	// reflect.Value contains a non-array pointer.
	rvptr := reflect.ValueOf(new(int)).Slice(0, 0)
	print(rvptr.Interface()) // @types

	// reflect.Value contains a non-string basic type.
	rvint := reflect.ValueOf(3).Slice(0, 0)
	print(rvint.Interface()) // @types
}

func reflectValueBytes() {
	sl1 := make([]byte, 0) // @line ar5sl1
	sl2 := make([]byte, 0) // @line ar5sl2

	rvsl1 := reflect.ValueOf(sl1)
	print(rvsl1.Interface())          // @types []byte
	print(rvsl1.Interface().([]byte)) // @pointsto makeslice@ar5sl1:13
	print(rvsl1.Bytes())              // @pointsto makeslice@ar5sl1:13

	rvsl2 := reflect.ValueOf(123)
	rvsl2.SetBytes(sl2)
	print(rvsl2.Interface())          // @types int
	print(rvsl2.Interface().([]byte)) // @pointsto
	print(rvsl2.Bytes())              // @pointsto

	rvsl3 := reflect.ValueOf([]byte(nil))
	rvsl3.SetBytes(sl2)
	print(rvsl3.Interface())          // @types []byte
	print(rvsl3.Interface().([]byte)) // @pointsto makeslice@ar5sl2:13
	print(rvsl3.Bytes())              // @pointsto makeslice@ar5sl2:13
}

func reflectValueIndex() {
	slice := []*int{&a} // @line ar6slice
	rv1 := reflect.ValueOf(slice)
	print(rv1.Index(42).Interface())        // @types *int
	print(rv1.Index(42).Interface().(*int)) // @pointsto main.a

	array := [10]*int{&a}
	rv2 := reflect.ValueOf(array)
	print(rv2.Index(42).Interface())        // @types *int
	print(rv2.Index(42).Interface().(*int)) // @pointsto main.a

	rv3 := reflect.ValueOf("string")
	print(rv3.Index(42).Interface()) // @types rune

	rv4 := reflect.ValueOf(&array)
	print(rv4.Index(42).Interface()) // @types

	rv5 := reflect.ValueOf(3)
	print(rv5.Index(42).Interface()) // @types
}

func reflectValueElem() {
	// Interface.
	var iface interface{} = &a
	rv1 := reflect.ValueOf(&iface).Elem()
	print(rv1.Interface())               // @types *int
	print(rv1.Interface().(*int))        // @pointsto main.a
	print(rv1.Elem().Interface())        // @types *int
	print(rv1.Elem().Interface().(*int)) // @pointsto main.a

	print(reflect.ValueOf(new(interface{})).Elem().Elem()) // @types

	// Pointer.
	ptr := &a
	rv2 := reflect.ValueOf(&ptr)
	print(rv2.Elem().Interface())        // @types *int
	print(rv2.Elem().Interface().(*int)) // @pointsto main.a

	// No other type works with (rV).Elem, not even those that
	// work with (rT).Elem: slice, array, map, chan.

	rv3 := reflect.ValueOf([]*int{&a})
	print(rv3.Elem().Interface()) // @types

	rv4 := reflect.ValueOf([10]*int{&a})
	print(rv4.Elem().Interface()) // @types

	rv5 := reflect.ValueOf(map[*int]*int{&a: &b})
	print(rv5.Elem().Interface()) // @types

	ch := make(chan *int)
	ch <- &a
	rv6 := reflect.ValueOf(ch)
	print(rv6.Elem().Interface()) // @types

	rv7 := reflect.ValueOf(3)
	print(rv7.Elem().Interface()) // @types
}

func reflectTypeElem() {
	rt1 := reflect.TypeOf(make([]*int, 0))
	print(reflect.Zero(rt1.Elem())) // @types *int

	rt2 := reflect.TypeOf([10]*int{})
	print(reflect.Zero(rt2.Elem())) // @types *int

	rt3 := reflect.TypeOf(map[*int]*int{})
	print(reflect.Zero(rt3.Elem())) // @types *int

	rt4 := reflect.TypeOf(make(chan *int))
	print(reflect.Zero(rt4.Elem())) // @types *int

	ptr := &a
	rt5 := reflect.TypeOf(&ptr)
	print(reflect.Zero(rt5.Elem())) // @types *int

	rt6 := reflect.TypeOf(3)
	print(reflect.Zero(rt6.Elem())) // @types
}

func reflectPtrTo() {
	tInt := reflect.TypeOf(3)
	tPtrInt := reflect.PtrTo(tInt)
	print(reflect.Zero(tPtrInt)) // @types *int
	tPtrPtrInt := reflect.PtrTo(tPtrInt)
	print(reflect.Zero(tPtrPtrInt)) // @types **int
}

func reflectSliceOf() {
	tInt := reflect.TypeOf(3)
	tSliceInt := reflect.SliceOf(tInt)
	print(reflect.Zero(tSliceInt)) // @types []int
}

type T struct{ x int }

func reflectMakeSlice() {
	rt := []reflect.Type{
		reflect.TypeOf(3),
		reflect.TypeOf([]int{}),
		reflect.TypeOf([]T{}),
	}[0]
	sl := reflect.MakeSlice(rt, 0, 0)
	print(sl)                         // @types []int | []T
	print(sl)                         // @pointsto <alloc in reflect.MakeSlice> | <alloc in reflect.MakeSlice>
	print(&sl.Interface().([]T)[0].x) // @pointsto <alloc in reflect.MakeSlice>[*].x
}

func main() {
	reflectValueSlice()
	reflectValueBytes()
	reflectValueIndex()
	reflectValueElem()
	reflectTypeElem()
	reflectPtrTo()
	reflectSliceOf()
	reflectMakeSlice()
}
