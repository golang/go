// -lang=go1.20

// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

func F[P any, Q *P](p P) {}

var _ = F[int]

func G[R any](func(R)) {}

func _() {
	G(F[int])
}
