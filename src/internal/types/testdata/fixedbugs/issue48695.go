// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

func g[P ~func(T) P, T any](P) {}

func _() {
	type F func(int) F
	var f F
	g(f)
	_ = g[F]
}
