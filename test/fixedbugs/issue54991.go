// compile

// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

import (
	"sync/atomic"
)

type I interface {
	M()
}

type S struct{}

func (*S) M() {}

type T struct {
	I
	x atomic.Int64
}

func F() {
	t := &T{I: &S{}}
	t.M()
}
