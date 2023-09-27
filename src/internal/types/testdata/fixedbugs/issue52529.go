// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

type Foo[P any] struct {
	_ *Bar[P]
}

type Bar[Q any] Foo[Q]

func (v *Bar[R]) M() {
	_ = (*Foo[R])(v)
}
