// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

func g[P any](P)      {}
func h[P, Q any](P) Q { panic(0) }

var _ func(int) = g
var _ func(int) string = h[int]

func f1(func(int))      {}
func f2(int, func(int)) {}

func _() {
	f1(g)
	f2(0, g)
}
