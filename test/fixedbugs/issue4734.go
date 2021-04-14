// compile

// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Caused gccgo to emit multiple definitions of the same symbol.

package p

type S1 struct{}

func (s *S1) M() {}

type S2 struct {
	F struct{ *S1 }
}

func F() {
	_ = struct{ *S1 }{}
}
