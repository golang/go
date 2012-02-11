// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

type A struct {
	s struct{int}
}

func (a *A) f() {
	a.s = struct{int}{0}
}

