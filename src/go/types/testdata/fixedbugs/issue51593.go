// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

func f[P interface{ m(R) }, R any]() {}

type T = interface { m(int) }

func _() {
	_ = f /* ERROR cannot infer R */ [T] // don't crash in type inference
}
