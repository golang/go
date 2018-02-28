// compile -c=4

// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 20162: embedded interfaces weren't dowidth-ed by the front end,
// leading to races in the backend.

package p

func Foo() {
	_ = (make([]func() interface {
		M(interface{})
	}, 1))
}
