// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import "testing"

// We generate memmove for copy(x[1:], x[:]), however we may change it to OpMove,
// because size is known. Check that OpMove is alias-safe, or we did call memmove.
func TestMove(t *testing.T) {
	x := [...]byte{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40}
	copy(x[1:], x[:])
	for i := 1; i < len(x); i++ {
		if int(x[i]) != i {
			t.Errorf("Memmove got converted to OpMove in alias-unsafe way. Got %d insted of %d in position %d", int(x[i]), i, i+1)
		}
	}
}

func TestMoveSmall(t *testing.T) {
	x := [...]byte{1, 2, 3, 4, 5, 6, 7}
	copy(x[1:], x[:])
	for i := 1; i < len(x); i++ {
		if int(x[i]) != i {
			t.Errorf("Memmove got converted to OpMove in alias-unsafe way. Got %d instead of %d in position %d", int(x[i]), i, i+1)
		}
	}
}

func TestSubFlags(t *testing.T) {
	if !subFlags32(0, 1).lt() {
		t.Errorf("subFlags32(0,1).lt() returned false")
	}
	if !subFlags32(0, 1).ult() {
		t.Errorf("subFlags32(0,1).ult() returned false")
	}
}
