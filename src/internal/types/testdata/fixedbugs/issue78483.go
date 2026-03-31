// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

func _() {
	for range f1 /* ERROR "cannot range over f1 (value of type func(func(...int) bool)): yield func of type func(...int) bool cannot be variadic" */ {
	}
	for range f2 /* ERROR "cannot range over f2 (value of type func(func(int, ...int) bool)): yield func of type func(int, ...int) bool cannot be variadic" */ {
	}
}

func f1(func(...int) bool)      {}
func f2(func(int, ...int) bool) {}
