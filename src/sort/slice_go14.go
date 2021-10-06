// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !go1.8
// +build !go1.8

package sort

import "reflect"

var reflectValueOf = reflect.ValueOf

func reflectSwapper(x interface{}) func(int, int) {
	v := reflectValueOf(x)
	tmp := reflect.New(v.Type().Elem()).Elem()
	return func(i, j int) {
		a, b := v.Index(i), v.Index(j)
		tmp.Set(a)
		a.Set(b)
		b.Set(tmp)
	}
}
