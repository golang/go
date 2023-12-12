// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

func f[P any](P) P { panic(0) }

var v func(string) int = f // ERROR "inferred type func(string) string for func(P) P does not match type func(string) int of v"

func _() func(string) int {
	return f // ERROR "inferred type func(string) string for func(P) P does not match type func(string) int of result variable"
}
