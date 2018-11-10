// compile

// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

type T struct{}

var (
	t = T{}
	u = t.New()
)

func x(T) (int, int) { return 0, 0 }

var _, _ = x(u)

func (T) New() T { return T{} }
