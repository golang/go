// run

// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"log"
	"reflect"
	"sort"
)

func main() {
	const length = 257
	x := make([]int64, length)
	for i := 0; i < length; i++ {
		x[i] = int64(i) * 27644437 % int64(length)
	}

	isLessStatic := func(i, j int) bool {
		return x[i] < x[j]
	}

	isLessReflect := reflect.MakeFunc(reflect.TypeOf(isLessStatic), func(args []reflect.Value) []reflect.Value {
		i := args[0].Int()
		j := args[1].Int()
		b := x[i] < x[j]
		return []reflect.Value{reflect.ValueOf(b)}
	}).Interface().(func(i, j int) bool)

	sort.SliceStable(x, isLessReflect)

	for i := 0; i < length-1; i++ {
		if x[i] >= x[i+1] {
			log.Fatalf("not sorted! (length=%v, idx=%v)\n%v\n", length, i, x)
		}
	}
}
