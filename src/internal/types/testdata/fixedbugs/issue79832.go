// -lang=go1.26

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

type Foo struct {
	Bar
}

type Bar struct {
	Baz int
}

var _ = Foo{Baz /* ERROR "use of promoted field Bar.Baz in struct literal of type Foo requires go1.27 or later" */ : 1}
