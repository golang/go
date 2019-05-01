// compile

// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Compile with static map literal.

package p

type i interface {
	j()
}

type s struct{}

func (s) j() {}

type foo map[string]i

var f = foo{
	"1": s{},
	"2": s{},
}
