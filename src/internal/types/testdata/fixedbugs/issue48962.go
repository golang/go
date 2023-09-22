// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

type T0[P any] struct {
	f P
}

type T1 /* ERROR "invalid recursive type" */ struct {
	_ T0[T1]
}
