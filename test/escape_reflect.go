// errorcheck -0 -m -l

// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test escape analysis for reflect Value operations.

package escape

import (
	"reflect"
	"unsafe"
)

var sink interface{}

func typ(x int) any {
	v := reflect.ValueOf(x) // ERROR "x does not escape"
	return v.Type()
}

func kind(x int) reflect.Kind {
	v := reflect.ValueOf(x) // ERROR "x does not escape"
	return v.Kind()
}

func int1(x int) int {
	v := reflect.ValueOf(x) // ERROR "x does not escape"
	return int(v.Int())
}

func ptr(x *int) *int { // ERROR "leaking param: x to result ~r0 level=0"
	v := reflect.ValueOf(x)
	return (*int)(v.UnsafePointer())
}

func bytes1(x []byte) byte { // ERROR "x does not escape"
	v := reflect.ValueOf(x) // ERROR "x does not escape"
	return v.Bytes()[0]
}

// Unfortunate: should only escape content. x (the interface storage) should not escape.
func bytes2(x []byte) []byte { // ERROR "leaking param: x$"
	v := reflect.ValueOf(x) // ERROR "x escapes to heap"
	return v.Bytes()
}

func string1(x string) string { // ERROR "leaking param: x to result ~r0 level=0"
	v := reflect.ValueOf(x) // ERROR "x does not escape"
	return v.String()
}

func string2(x int) string {
	v := reflect.ValueOf(x) // ERROR "x does not escape"
	return v.String()
}

// Unfortunate: should only escape to result.
func interface1(x any) any { // ERROR "leaking param: x$"
	v := reflect.ValueOf(x)
	return v.Interface()
}

func interface2(x int) any {
	v := reflect.ValueOf(x) // ERROR "x escapes to heap"
	return v.Interface()
}

// Unfortunate: should not escape.
func interface3(x int) int {
	v := reflect.ValueOf(x) // ERROR "x escapes to heap"
	return v.Interface().(int)
}

// Unfortunate: should only escape to result.
func interface4(x *int) any { // ERROR "leaking param: x$"
	v := reflect.ValueOf(x)
	return v.Interface()
}

func addr(x *int) reflect.Value { // ERROR "leaking param: x to result ~r0 level=0"
	v := reflect.ValueOf(x).Elem()
	return v.Addr()
}

// functions returning pointer as uintptr have to escape.
func uintptr1(x *int) uintptr { // ERROR "leaking param: x$"
	v := reflect.ValueOf(x)
	return v.Pointer()
}

func unsafeaddr(x *int) uintptr { // ERROR "leaking param: x$"
	v := reflect.ValueOf(x).Elem()
	return v.UnsafeAddr()
}

func ifacedata(x any) [2]uintptr { // ERROR "moved to heap: x"
	v := reflect.ValueOf(&x).Elem()
	return v.InterfaceData()
}

func can(x int) bool {
	v := reflect.ValueOf(x) // ERROR "x does not escape"
	return v.CanAddr() || v.CanInt() || v.CanSet() || v.CanInterface()
}

func is(x int) bool {
	v := reflect.ValueOf(x) // ERROR "x does not escape"
	return v.IsValid() || v.IsNil() || v.IsZero()
}

func is2(x [2]int) bool {
	v := reflect.ValueOf(x) // ERROR "x does not escape"
	return v.IsValid() || v.IsNil() || v.IsZero()
}

func is3(x struct{ a, b int }) bool {
	v := reflect.ValueOf(x) // ERROR "x does not escape"
	return v.IsValid() || v.IsNil() || v.IsZero()
}

func overflow(x int) bool {
	v := reflect.ValueOf(x) // ERROR "x does not escape"
	return v.OverflowInt(1 << 62)
}

func len1(x []int) int { // ERROR "x does not escape"
	v := reflect.ValueOf(x) // ERROR "x does not escape"
	return v.Len()
}

func len2(x [3]int) int {
	v := reflect.ValueOf(x) // ERROR "x does not escape"
	return v.Len()
}

func len3(x string) int { // ERROR "x does not escape"
	v := reflect.ValueOf(x) // ERROR "x does not escape"
	return v.Len()
}

func len4(x map[int]int) int { // ERROR "x does not escape"
	v := reflect.ValueOf(x)
	return v.Len()
}

func len5(x chan int) int { // ERROR "x does not escape"
	v := reflect.ValueOf(x)
	return v.Len()
}

func cap1(x []int) int { // ERROR "x does not escape"
	v := reflect.ValueOf(x) // ERROR "x does not escape"
	return v.Cap()
}

func cap2(x [3]int) int {
	v := reflect.ValueOf(x) // ERROR "x does not escape"
	return v.Cap()
}

func cap3(x chan int) int { // ERROR "x does not escape"
	v := reflect.ValueOf(x)
	return v.Cap()
}

func setlen(x *[]int, n int) { // ERROR "x does not escape"
	v := reflect.ValueOf(x).Elem()
	v.SetLen(n)
}

func setcap(x *[]int, n int) { // ERROR "x does not escape"
	v := reflect.ValueOf(x).Elem()
	v.SetCap(n)
}

// Unfortunate: x doesn't need to escape to heap, just to result.
func slice1(x []byte) []byte { // ERROR "leaking param: x$"
	v := reflect.ValueOf(x) // ERROR "x escapes to heap"
	return v.Slice(1, 2).Bytes()
}

// Unfortunate: x doesn't need to escape to heap, just to result.
func slice2(x string) string { // ERROR "leaking param: x$"
	v := reflect.ValueOf(x) // ERROR "x escapes to heap"
	return v.Slice(1, 2).String()
}

func slice3(x [10]byte) []byte {
	v := reflect.ValueOf(x) // ERROR "x escapes to heap"
	return v.Slice(1, 2).Bytes()
}

func elem1(x *int) int { // ERROR "x does not escape"
	v := reflect.ValueOf(x)
	return int(v.Elem().Int())
}

func elem2(x *string) string { // ERROR "leaking param: x to result ~r0 level=1"
	v := reflect.ValueOf(x)
	return string(v.Elem().String())
}

type S struct {
	A int
	B *int
	C string
}

func (S) M() {}

func field1(x S) int { // ERROR "x does not escape"
	v := reflect.ValueOf(x) // ERROR "x does not escape"
	return int(v.Field(0).Int())
}

func field2(x S) string { // ERROR "leaking param: x to result ~r0 level=0"
	v := reflect.ValueOf(x) // ERROR "x does not escape"
	return v.Field(2).String()
}

func numfield(x S) int { // ERROR "x does not escape"
	v := reflect.ValueOf(x) // ERROR "x does not escape"
	return v.NumField()
}

func index1(x []int) int { // ERROR "x does not escape"
	v := reflect.ValueOf(x) // ERROR "x does not escape"
	return int(v.Index(0).Int())
}

// Unfortunate: should only leak content (level=1)
func index2(x []string) string { // ERROR "leaking param: x to result ~r0 level=0"
	v := reflect.ValueOf(x) // ERROR "x does not escape"
	return v.Index(0).String()
}

func index3(x [3]int) int {
	v := reflect.ValueOf(x) // ERROR "x does not escape"
	return int(v.Index(0).Int())
}

func index4(x [3]string) string { // ERROR "leaking param: x to result ~r0 level=0"
	v := reflect.ValueOf(x) // ERROR "x does not escape"
	return v.Index(0).String()
}

func index5(x string) byte { // ERROR "x does not escape"
	v := reflect.ValueOf(x) // ERROR "x does not escape"
	return byte(v.Index(0).Uint())
}

// Unfortunate: x (the interface storage) doesn't need to escape as the function takes a scalar arg.
func call1(f func(int), x int) { // ERROR "leaking param: f$"
	fv := reflect.ValueOf(f)
	v := reflect.ValueOf(x)     // ERROR "x escapes to heap"
	fv.Call([]reflect.Value{v}) // ERROR "\[\]reflect\.Value{\.\.\.} does not escape"
}

func call2(f func(*int), x *int) { // ERROR "leaking param: f$" "leaking param: x$"
	fv := reflect.ValueOf(f)
	v := reflect.ValueOf(x)
	fv.Call([]reflect.Value{v}) // ERROR "\[\]reflect.Value{\.\.\.} does not escape"
}

func method(x S) reflect.Value { // ERROR "leaking param: x$"
	v := reflect.ValueOf(x) // ERROR "x escapes to heap"
	return v.Method(0)
}

func nummethod(x S) int { // ERROR "x does not escape"
	v := reflect.ValueOf(x) // ERROR "x does not escape"
	return v.NumMethod()
}

// Unfortunate: k doesn't need to escape.
func mapindex(m map[string]string, k string) string { // ERROR "m does not escape" "leaking param: k$"
	mv := reflect.ValueOf(m)
	kv := reflect.ValueOf(k) // ERROR "k escapes to heap"
	return mv.MapIndex(kv).String()
}

func mapkeys(m map[string]string) []reflect.Value { // ERROR "m does not escape"
	mv := reflect.ValueOf(m)
	return mv.MapKeys()
}

func mapiter1(m map[string]string) *reflect.MapIter { // ERROR "leaking param: m$"
	mv := reflect.ValueOf(m)
	return mv.MapRange()
}

func mapiter2(m map[string]string) string { // ERROR "leaking param: m$"
	mv := reflect.ValueOf(m)
	it := mv.MapRange()
	if it.Next() {
		return it.Key().String()
	}
	return ""
}

func mapiter3(m map[string]string, it *reflect.MapIter) { // ERROR "leaking param: m$" "it does not escape"
	mv := reflect.ValueOf(m)
	it.Reset(mv)
}

func recv1(ch chan string) string { // ERROR "ch does not escape"
	v := reflect.ValueOf(ch)
	r, _ := v.Recv()
	return r.String()
}

func recv2(ch chan string) string { // ERROR "ch does not escape"
	v := reflect.ValueOf(ch)
	r, _ := v.TryRecv()
	return r.String()
}

// Unfortunate: x (the interface storage) doesn't need to escape.
func send1(ch chan string, x string) { // ERROR "ch does not escape" "leaking param: x$"
	vc := reflect.ValueOf(ch)
	vx := reflect.ValueOf(x) // ERROR "x escapes to heap"
	vc.Send(vx)
}

// Unfortunate: x (the interface storage) doesn't need to escape.
func send2(ch chan string, x string) bool { // ERROR "ch does not escape" "leaking param: x$"
	vc := reflect.ValueOf(ch)
	vx := reflect.ValueOf(x) // ERROR "x escapes to heap"
	return vc.TrySend(vx)
}

func close1(ch chan string) { // ERROR "ch does not escape"
	v := reflect.ValueOf(ch)
	v.Close()
}

func select1(ch chan string) string { // ERROR "leaking param: ch$"
	v := reflect.ValueOf(ch)
	cas := reflect.SelectCase{Dir: reflect.SelectRecv, Chan: v}
	_, r, _ := reflect.Select([]reflect.SelectCase{cas}) // ERROR "\[\]reflect.SelectCase{...} does not escape"
	return r.String()
}

// Unfortunate: x (the interface storage) doesn't need to escape.
func select2(ch chan string, x string) { // ERROR "leaking param: ch$" "leaking param: x$"
	vc := reflect.ValueOf(ch)
	vx := reflect.ValueOf(x) // ERROR "x escapes to heap"
	cas := reflect.SelectCase{Dir: reflect.SelectSend, Chan: vc, Send: vx}
	reflect.Select([]reflect.SelectCase{cas}) // ERROR "\[\]reflect.SelectCase{...} does not escape"
}

var (
	intTyp    = reflect.TypeOf(int(0))     // ERROR "0 does not escape"
	uintTyp   = reflect.TypeOf(uint(0))    // ERROR "uint\(0\) does not escape"
	stringTyp = reflect.TypeOf(string("")) // ERROR ".. does not escape"
	bytesTyp  = reflect.TypeOf([]byte{})   // ERROR "\[\]byte{} does not escape"
)

// Unfortunate: should not escape.
func convert1(x int) uint {
	v := reflect.ValueOf(x) // ERROR "x escapes to heap"
	return uint(v.Convert(uintTyp).Uint())
}

// Unfortunate: should only escape content to result.
func convert2(x []byte) string { // ERROR "leaking param: x$"
	v := reflect.ValueOf(x) // ERROR "x escapes to heap"
	return v.Convert(stringTyp).String()
}

// Unfortunate: v doesn't need to leak, x (the interface storage) doesn't need to escape.
func set1(v reflect.Value, x int) { // ERROR "leaking param: v$"
	vx := reflect.ValueOf(x) // ERROR "x escapes to heap"
	v.Set(vx)
}

// Unfortunate: a can be stack allocated, x (the interface storage) doesn't need to escape.
func set2(x int) int64 {
	var a int // ERROR "moved to heap: a"
	v := reflect.ValueOf(&a).Elem()
	vx := reflect.ValueOf(x) // ERROR "x escapes to heap"
	v.Set(vx)
	return v.Int()
}

func set3(v reflect.Value, x int) { // ERROR "v does not escape"
	v.SetInt(int64(x))
}

func set4(x int) int {
	var a int
	v := reflect.ValueOf(&a).Elem() // a should not escape, no error printed
	v.SetInt(int64(x))
	return int(v.Int())
}

func set5(v reflect.Value, x string) { // ERROR "v does not escape" "leaking param: x$"
	v.SetString(x)
}

func set6(v reflect.Value, x []byte) { // ERROR "v does not escape" "leaking param: x$"
	v.SetBytes(x)
}

func set7(v reflect.Value, x unsafe.Pointer) { // ERROR "v does not escape" "leaking param: x$"
	v.SetPointer(x)
}

func setmapindex(m map[string]string, k, e string) { // ERROR "m does not escape" "leaking param: k$" "leaking param: e$"
	mv := reflect.ValueOf(m)
	kv := reflect.ValueOf(k) // ERROR "k escapes to heap"
	ev := reflect.ValueOf(e) // ERROR "e escapes to heap"
	mv.SetMapIndex(kv, ev)
}

// Unfortunate: k doesn't need to escape.
func mapdelete(m map[string]string, k string) { // ERROR "m does not escape" "leaking param: k$"
	mv := reflect.ValueOf(m)
	kv := reflect.ValueOf(k) // ERROR "k escapes to heap"
	mv.SetMapIndex(kv, reflect.Value{})
}

// Unfortunate: v doesn't need to leak.
func setiterkey1(v reflect.Value, it *reflect.MapIter) { // ERROR "leaking param: v$" "leaking param content: it$"
	v.SetIterKey(it)
}

// Unfortunate: v doesn't need to leak.
func setiterkey2(v reflect.Value, m map[string]string) { // ERROR "leaking param: v$" "leaking param: m$"
	it := reflect.ValueOf(m).MapRange()
	v.SetIterKey(it)
}

// Unfortunate: v doesn't need to leak.
func setitervalue1(v reflect.Value, it *reflect.MapIter) { // ERROR "leaking param: v$" "leaking param content: it$"
	v.SetIterValue(it)
}

// Unfortunate: v doesn't need to leak.
func setitervalue2(v reflect.Value, m map[string]string) { // ERROR "leaking param: v$" "leaking param: m$"
	it := reflect.ValueOf(m).MapRange()
	v.SetIterValue(it)
}

// Unfortunate: s doesn't need escape, only leak to result.
// And x (interface storage) doesn't need to escape.
func append1(s []int, x int) []int { // ERROR "leaking param: s$"
	sv := reflect.ValueOf(s)     // ERROR "s escapes to heap"
	xv := reflect.ValueOf(x)     // ERROR "x escapes to heap"
	rv := reflect.Append(sv, xv) // ERROR "... argument does not escape"
	return rv.Interface().([]int)
}

// Unfortunate: s doesn't need escape, only leak to result.
func append2(s, x []int) []int { // ERROR "leaking param: s$" "x does not escape"
	sv := reflect.ValueOf(s) // ERROR "s escapes to heap"
	xv := reflect.ValueOf(x) // ERROR "x does not escape"
	rv := reflect.AppendSlice(sv, xv)
	return rv.Interface().([]int)
}
