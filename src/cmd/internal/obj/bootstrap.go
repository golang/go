// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !go1.8

package obj

import (
	"reflect"
	"sort"
)

func SortSlice(slice interface{}, less func(i, j int) bool) {
	val := reflect.ValueOf(slice)
	tmp := reflect.New(val.Type().Elem()).Elem()
	x := sliceByFn{val: val, tmp: tmp, less: less}
	sort.Sort(x)
}

type sliceByFn struct {
	val  reflect.Value
	tmp  reflect.Value
	less func(i, j int) bool
}

func (x sliceByFn) Len() int           { return x.val.Len() }
func (x sliceByFn) Less(i, j int) bool { return x.less(i, j) }
func (x sliceByFn) Swap(i, j int) {
	a, b := x.val.Index(i), x.val.Index(j)
	x.tmp.Set(a)
	a.Set(b)
	b.Set(x.tmp)
}
