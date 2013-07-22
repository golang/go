// run

// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// issue 5753: bad typecheck info causes escape analysis to
// not run on method thunks.

package main

type Thing struct{}

func (t *Thing) broken(s string) []string {
	foo := [1]string{s}
	return foo[:]
}

func main() {
	t := &Thing{}

	f := t.broken
	s := f("foo")
	_ = f("bar")
	if s[0] != "foo" {
		panic(`s[0] != "foo"`)
	}
	
}
