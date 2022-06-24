// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

func f[F interface{*Q}, G interface{*R}, Q, R any](q Q, r R) {}

func _() {
	f[*float64, *int](1, 2)
	f[*float64](1, 2)
	f(1, 2)
}
