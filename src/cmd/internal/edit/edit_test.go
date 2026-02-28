// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package edit

import "testing"

func TestEdit(t *testing.T) {
	b := NewBuffer([]byte("0123456789"))
	b.Insert(8, ",7½,")
	b.Replace(9, 10, "the-end")
	b.Insert(10, "!")
	b.Insert(4, "3.14,")
	b.Insert(4, "π,")
	b.Insert(4, "3.15,")
	b.Replace(3, 4, "three,")
	want := "012three,3.14,π,3.15,4567,7½,8the-end!"

	s := b.String()
	if s != want {
		t.Errorf("b.String() = %q, want %q", s, want)
	}
	sb := b.Bytes()
	if string(sb) != want {
		t.Errorf("b.Bytes() = %q, want %q", sb, want)
	}
}
