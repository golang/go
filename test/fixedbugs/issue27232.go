// compile

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

type F = func(T)

type T interface {
	m(F)
}

type t struct{}

func (t) m(F) {}

var _ T = &t{}
