// run

// Copyright 2013 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Bug in method values: escape analysis was off.

package main

import "sync"

var called = false

type T struct {
	once sync.Once
}

func (t *T) M() {
	called = true
}

func main() {
	var t T
	t.once.Do(t.M)
	if !called {
		panic("not called")
	}
}
