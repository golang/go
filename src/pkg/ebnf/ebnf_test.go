// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ebnf

import (
	"go/token"
	"io/ioutil"
	"testing"
)


var fset = token.NewFileSet()


var grammars = []string{
	`Program = .
	`,

	`Program = foo .
	foo = "foo" .
	`,

	`Program = "a" | "b" "c" .
	`,

	`Program = "a" ... "z" .
	`,

	`Program = Song .
	 Song = { Note } .
	 Note = Do | (Re | Mi | Fa | So | La) | Ti .
	 Do = "c" .
	 Re = "d" .
	 Mi = "e" .
	 Fa = "f" .
	 So = "g" .
	 La = "a" .
	 Ti = ti .
	 ti = "b" .
	`,
}


func check(t *testing.T, filename string, src []byte) {
	grammar, err := Parse(fset, filename, src)
	if err != nil {
		t.Errorf("Parse(%s) failed: %v", src, err)
	}
	if err = Verify(fset, grammar, "Program"); err != nil {
		t.Errorf("Verify(%s) failed: %v", src, err)
	}
}


func TestGrammars(t *testing.T) {
	for _, src := range grammars {
		check(t, "", []byte(src))
	}
}


var files = []string{
// TODO(gri) add some test files
}


func TestFiles(t *testing.T) {
	for _, filename := range files {
		src, err := ioutil.ReadFile(filename)
		if err != nil {
			t.Fatal(err)
		}
		check(t, filename, src)
	}
}
