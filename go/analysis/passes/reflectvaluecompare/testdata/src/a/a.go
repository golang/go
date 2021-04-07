// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file contains tests for the reflectvaluecompare checker.

package a

import (
	"reflect"
)

func f() {
	var x, y reflect.Value
	var a, b interface{}
	_ = x == y // want `avoid using == with reflect.Value`
	_ = x == a // want `avoid using == with reflect.Value`
	_ = a == x // want `avoid using == with reflect.Value`
	_ = a == b

	// Comparing to reflect.Value{} is ok.
	_ = a == reflect.Value{}
	_ = reflect.Value{} == a
	_ = reflect.Value{} == reflect.Value{}
}
func g() {
	var x, y reflect.Value
	var a, b interface{}
	_ = x != y // want `avoid using != with reflect.Value`
	_ = x != a // want `avoid using != with reflect.Value`
	_ = a != x // want `avoid using != with reflect.Value`
	_ = a != b

	// Comparing to reflect.Value{} is ok.
	_ = a != reflect.Value{}
	_ = reflect.Value{} != a
	_ = reflect.Value{} != reflect.Value{}
}
func h() {
	var x, y reflect.Value
	var a, b interface{}
	reflect.DeepEqual(x, y) // want `avoid using reflect.DeepEqual with reflect.Value`
	reflect.DeepEqual(x, a) // want `avoid using reflect.DeepEqual with reflect.Value`
	reflect.DeepEqual(a, x) // want `avoid using reflect.DeepEqual with reflect.Value`
	reflect.DeepEqual(a, b)

	// Comparing to reflect.Value{} is ok.
	reflect.DeepEqual(reflect.Value{}, a)
	reflect.DeepEqual(a, reflect.Value{})
	reflect.DeepEqual(reflect.Value{}, reflect.Value{})
}
