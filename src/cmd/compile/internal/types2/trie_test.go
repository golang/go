// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types2

import (
	"fmt"
	"testing"
)

func TestTrie(t *testing.T) {
	strie := make(trie[string])

	okay := func(index ...int) {
		s := fmt.Sprintf("%x", index)
		if p, n := strie.insert(index, s); n != 0 {
			t.Errorf("%s collided with %s (n = %d)", s, p, n)
		}
	}

	fail := func(collision string, index ...int) {
		s := fmt.Sprintf("%x", index)
		if p, n := strie.insert(index, s); n == 0 {
			t.Errorf("%s did not collide", s)
		} else if p != collision {
			t.Errorf("%s collided with %s (n == %d), want %s", s, p, n, collision)
		}
	}

	clear(strie)
	okay(0)
	fail("[0]", 0)

	clear(strie)
	okay(0)
	fail("[0]", 0, 1, 2, 3, 4, 5)

	clear(strie)
	okay(1, 2)
	okay(1, 3)
	okay(1, 4, 5)
	okay(1, 4, 2)
	fail("[1 4 2]", 1, 4)
	fail("[1 4 5]", 1, 4, 5)
	okay(1, 4, 3)
	okay(2, 1)
	okay(2, 2)
	fail("[2 2]", 2, 2, 3)

	clear(strie)
	okay(0, 1, 2, 3, 4, 5)
	okay(0, 1, 2, 3, 4, 6)
	okay(0, 1, 2, 3, 4, 7)
	okay(0, 1, 2, 3, 4, 8, 1)
	okay(0, 1, 2, 3, 4, 4)
	fail("[0 1 2 3 4 4]", 0, 1, 2, 3)
}

func TestAnyValue(t *testing.T) {
	atrie := make(trie[any]) // allow values of any type

	val := new(42)
	alt, n := atrie.insert([]int{0}, val)
	if n != 0 {
		t.Errorf("unexpected collision (n = %d)", n)
	}
	if alt != val {
		t.Errorf("unexpected result (alt = %#x, val = %#x)", alt, val)
	}

	alt, n = atrie.insert([]int{0}, val) // nil is a valid value
	if n == 0 {
		t.Errorf("expected collision")
	}
	if alt != val {
		t.Errorf("unexpected result (alt = %#x, val = %#x)", alt, val)
	}
}
