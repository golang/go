// errchk $G $D/$F.go

// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// issue 1910
// error on wrong line

package main

import "container/list"

type Painting struct {
	fragments list.List // private
}

func (p Painting) Foo() {
	for e := p.fragments; e.Front() != nil; {  // ERROR "unexported field|hidden field"
	}
}

// from comment 4 of issue 1910
type Foo interface {
	Run(a int) (a int)  // ERROR "a redeclared|redefinition|previous"
}
