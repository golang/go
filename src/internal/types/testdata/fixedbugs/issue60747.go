// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

func f[P any](P) P { panic(0) }

var v func(string) int = f // ERROR "type func(string) int of v does not match inferred type func(string) string for func(P) P"

func _() func(string) int {
	return f // ERROR "type func(string) int of result variable does not match inferred type func(string) string for func(P) P"
}
