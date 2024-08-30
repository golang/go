// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

type A = uint8
type E uint8

func f[P ~A](P) {}

func g(e E) {
	f(e)
}
