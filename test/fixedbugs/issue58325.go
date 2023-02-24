// compile

// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

type S1 struct {
	s2 S2
}

type S2 struct{}

func (S2) Make() S2 {
	return S2{}
}

func (S1) Make() S1 {
	return S1{s2: S2{}.Make()}
}

var _ = S1{}.Make()
