// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package one

type I1 interface {
	f()
}

type S1 struct {
}

func (s S1) f() {
}

func F1(i1 I1) {
}
