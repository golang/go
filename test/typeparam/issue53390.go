// compile

// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

import "unsafe"

func F[T any](v T) uintptr {
	return unsafe.Alignof(func() T {
		func(any) {}(struct{ _ T }{})
		return v
	}())
}

func f() {
	F(0)
}
