// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package test

import (
	"fmt"
	"sort"
	"testing"
)

// Test that calling methods on generic types doesn't cause allocations.
func genericSorted[T sort.Interface](data T) bool {
	n := data.Len()
	for i := n - 1; i > 0; i-- {
		if data.Less(i, i-1) {
			return false
		}
	}
	return true
}
func TestGenericSorted(t *testing.T) {
	var data = sort.IntSlice{-10, -5, 0, 1, 2, 3, 5, 7, 11, 100, 100, 100, 1000, 10000}
	f := func() {
		genericSorted(data)
	}
	if n := testing.AllocsPerRun(10, f); n > 0 {
		t.Errorf("got %f allocs, want 0", n)
	}
}

// Test that escape analysis correctly tracks escaping inside of methods
// called on generic types.
type fooer interface {
	foo()
}
type P struct {
	p *int
	q int
}

var esc []*int

func (p P) foo() {
	esc = append(esc, p.p) // foo escapes the pointer from inside of p
}
func f[T fooer](t T) {
	t.foo()
}
func TestGenericEscape(t *testing.T) {
	for i := 0; i < 4; i++ {
		var x int = 77 + i
		var p P = P{p: &x}
		f(p)
	}
	for i, p := range esc {
		if got, want := *p, 77+i; got != want {
			panic(fmt.Sprintf("entry %d: got %d, want %d", i, got, want))
		}
	}
}
