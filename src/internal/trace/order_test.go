// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package trace

import "testing"

func TestQueue(t *testing.T) {
	var q queue[int]
	check := func(name string, exp []int) {
		for _, v := range exp {
			q.push(v)
		}
		for i, want := range exp {
			if got, ok := q.pop(); !ok {
				t.Fatalf("check %q: expected to be able to pop after %d pops", name, i+1)
			} else if got != want {
				t.Fatalf("check %q: expected value %d after on pop %d, got %d", name, want, i+1, got)
			}
		}
		if _, ok := q.pop(); ok {
			t.Fatalf("check %q: did not expect to be able to pop more values", name)
		}
		if _, ok := q.pop(); ok {
			t.Fatalf("check %q: did not expect to be able to pop more values a second time", name)
		}
	}
	check("one element", []int{4})
	check("two elements", []int{64, 12})
	check("six elements", []int{55, 16423, 2352, 644, 12874, 9372})
	check("one element again", []int{7})
	check("two elements again", []int{77, 6336})
}
