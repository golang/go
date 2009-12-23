// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package vector


import (
	"fmt"
	"sort"
	"testing"
)

var (
	zero    interface{}
	intzero int
	strzero string
)


func int2Value(x int) int       { return x }
func int2IntValue(x int) int    { return x }
func int2StrValue(x int) string { return string(x) }


func elem2Value(x interface{}) int  { return x.(int) }
func elem2IntValue(x int) int       { return x }
func elem2StrValue(x string) string { return x }


func intf2Value(x interface{}) int       { return x.(int) }
func intf2IntValue(x interface{}) int    { return x.(int) }
func intf2StrValue(x interface{}) string { return x.(string) }


type VectorInterface interface {
	Len() int
	Cap() int
}


func checkSize(t *testing.T, v VectorInterface, len, cap int) {
	if v.Len() != len {
		t.Errorf("%T expected len = %d; found %d", v, len, v.Len())
	}
	if v.Cap() < cap {
		t.Errorf("%T expected cap >= %d; found %d", v, cap, v.Cap())
	}
}


func val(i int) int { return i*991 - 1234 }


func TestSorting(t *testing.T) {
	const n = 100

	a := new(IntVector).Resize(n, 0)
	for i := n - 1; i >= 0; i-- {
		a.Set(i, n-1-i)
	}
	if sort.IsSorted(a) {
		t.Error("int vector not sorted")
	}

	b := new(StringVector).Resize(n, 0)
	for i := n - 1; i >= 0; i-- {
		b.Set(i, fmt.Sprint(n-1-i))
	}
	if sort.IsSorted(b) {
		t.Error("string vector not sorted")
	}
}


func tname(x interface{}) string { return fmt.Sprintf("%T: ", x) }
