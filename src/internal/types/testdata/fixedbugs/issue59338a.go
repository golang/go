// -lang=go1.20

// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

func g[P any](P)      {}
func h[P, Q any](P) Q { panic(0) }

var _ func(int) = g /* ERROR "implicitly instantiated function in assignment requires go1.21 or later" */
var _ func(int) string = h[ /* ERROR "partially instantiated function in assignment requires go1.21 or later" */ int]

func f1(func(int))      {}
func f2(int, func(int)) {}

func _() {
	f1(g /* ERROR "implicitly instantiated function as argument requires go1.21 or later" */)
	f2(0, g /* ERROR "implicitly instantiated function as argument requires go1.21 or later" */)
}
