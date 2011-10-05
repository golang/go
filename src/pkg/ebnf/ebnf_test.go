// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ebnf

import (
	"bytes"
	"testing"
)

var goodGrammars = []string{
	`Program = .`,

	`Program = foo .
	 foo = "foo" .`,

	`Program = "a" | "b" "c" .`,

	`Program = "a" … "z" .`,

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
	 ti = "b" .`,
}

var badGrammars = []string{
	`Program = | .`,
	`Program = | b .`,
	`Program = a … b .`,
	`Program = "a" … .`,
	`Program = … "b" .`,
	`Program = () .`,
	`Program = [] .`,
	`Program = {} .`,
}

func checkGood(t *testing.T, src string) {
	grammar, err := Parse("", bytes.NewBuffer([]byte(src)))
	if err != nil {
		t.Errorf("Parse(%s) failed: %v", src, err)
		return
	}
	if err = Verify(grammar, "Program"); err != nil {
		t.Errorf("Verify(%s) failed: %v", src, err)
	}
}

func checkBad(t *testing.T, src string) {
	_, err := Parse("", bytes.NewBuffer([]byte(src)))
	if err == nil {
		t.Errorf("Parse(%s) should have failed", src)
	}
}

func TestGrammars(t *testing.T) {
	for _, src := range goodGrammars {
		checkGood(t, src)
	}
	for _, src := range badGrammars {
		checkBad(t, src)
	}
}
