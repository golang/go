// errorcheck -G=3

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

type T0[P any] struct { // ERROR "invalid recursive type"
	f P
}

type T1 struct {
	_ T0[T1]
}
