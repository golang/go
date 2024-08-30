// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

func f(func(int) string) {}

func g2[P, Q any](P) Q    { var q Q; return q }
func g3[P, Q, R any](P) R { var r R; return r }

func _() {
	f(g2)
	f(g2[int])
	f(g2[int, string])

	f(g3[int, bool])
	f(g3[int, bool, string])

	var _ func(int) string = g2
	var _ func(int) string = g2[int]
}
