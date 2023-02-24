// compile

// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

import "unsafe"

func F[T int](v T) uintptr {
	return unsafe.Offsetof(struct{ f T }{
		func(T) T { return v }(v),
	}.f)
}

func f() {
	F(1)
}
