// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package trie

import (
	"reflect"
	"strconv"
	"testing"
)

func TestScope(t *testing.T) {
	def := Scope{}
	s0, s1 := newScope(), newScope()
	if s0 == def || s1 == def {
		t.Error("newScope() should never be == to the default scope")
	}
	if s0 == s1 {
		t.Errorf("newScope() %q and %q should not be ==", s0, s1)
	}
	if s0.id == 0 {
		t.Error("s0.id is 0")
	}
	if s1.id == 0 {
		t.Error("s1.id is 0")
	}
	got := s0.String()
	if _, err := strconv.Atoi(got); err != nil {
		t.Errorf("scope{%s}.String() is not an int: got %s with error %s", s0, got, err)
	}
}

func TestCollision(t *testing.T) {
	var x interface{} = 1
	var y interface{} = 2

	if v := TakeLhs(x, y); v != x {
		t.Errorf("TakeLhs(%s, %s) got %s. want %s", x, y, v, x)
	}
	if v := TakeRhs(x, y); v != y {
		t.Errorf("TakeRhs(%s, %s) got %s. want %s", x, y, v, y)
	}
}

func TestDefault(t *testing.T) {
	def := Map{}

	if def.Size() != 0 {
		t.Errorf("default node has non-0 size %d", def.Size())
	}
	if want, got := (Scope{}), def.Scope(); got != want {
		t.Errorf("default is in a non default scope (%s) from b (%s)", got, want)
	}
	if v, ok := def.Lookup(123); !(v == nil && !ok) {
		t.Errorf("Scope{}.Lookup() = (%s, %v) not (nil, false)", v, ok)
	}
	if !def.Range(func(k uint64, v interface{}) bool {
		t.Errorf("Scope{}.Range() called it callback on %d:%s", k, v)
		return true
	}) {
		t.Error("Scope{}.Range() always iterates through all elements")
	}

	if got, want := def.String(), "{}"; got != want {
		t.Errorf("Scope{}.String() got %s. want %s", got, want)
	}

	b := NewBuilder()
	if def == b.Empty() {
		t.Error("Scope{} == to an empty node from a builder")
	}
	if b.Clone(def) != b.Empty() {
		t.Error("b.Clone(Scope{}) should equal b.Empty()")
	}
	if !def.DeepEqual(b.Empty()) {
		t.Error("Scope{}.DeepEqual(b.Empty()) should hold")
	}
}

func TestBuilders(t *testing.T) {
	b0, b1 := NewBuilder(), NewBuilder()
	if b0.Scope() == b1.Scope() {
		t.Errorf("builders have the same scope %s", b0.Scope())
	}

	if b0.Empty() == b1.Empty() {
		t.Errorf("empty nodes from different scopes are disequal")
	}
	if !b0.Empty().DeepEqual(b1.Empty()) {
		t.Errorf("empty nodes from different scopes are not DeepEqual")
	}

	clone := b1.Clone(b0.Empty())
	if clone != b1.Empty() {
		t.Errorf("Clone() empty nodes %v != %v", clone, b1.Empty())
	}
}

func TestEmpty(t *testing.T) {
	b := NewBuilder()
	e := b.Empty()
	if e.Size() != 0 {
		t.Errorf("empty nodes has non-0 size %d", e.Size())
	}
	if e.Scope() != b.Scope() {
		t.Errorf("b.Empty() is in a different scope (%s) from b (%s)", e.Scope(), b.Scope())
	}
	if v, ok := e.Lookup(123); !(v == nil && !ok) {
		t.Errorf("empty.Lookup() = (%s, %v) not (nil, false)", v, ok)
	}
	if l := e.n.find(123); l != nil {
		t.Errorf("empty.find(123) got %v. want nil", l)
	}
	e.Range(func(k uint64, v interface{}) bool {
		t.Errorf("empty.Range() called it callback on %d:%s", k, v)
		return true
	})

	want := "{}"
	if got := e.String(); got != want {
		t.Errorf("empty.String(123) got %s. want %s", got, want)
	}
}

func TestCreate(t *testing.T) {
	// The node orders are printed in lexicographic little-endian.
	b := NewBuilder()
	for _, c := range []struct {
		m    map[uint64]interface{}
		want string
	}{
		{
			map[uint64]interface{}{},
			"{}",
		},
		{
			map[uint64]interface{}{1: "a"},
			"{1: a}",
		},
		{
			map[uint64]interface{}{2: "b", 1: "a"},
			"{1: a, 2: b}",
		},
		{
			map[uint64]interface{}{1: "x", 4: "y", 5: "z"},
			"{1: x, 4: y, 5: z}",
		},
	} {
		m := b.Create(c.m)
		if got := m.String(); got != c.want {
			t.Errorf("Create(%v) got %q. want %q ", c.m, got, c.want)
		}
	}
}

func TestElems(t *testing.T) {
	b := NewBuilder()
	for _, orig := range []map[uint64]interface{}{
		{},
		{1: "a"},
		{1: "a", 2: "b"},
		{1: "x", 4: "y", 5: "z"},
		{1: "x", 4: "y", 5: "z", 123: "abc"},
	} {
		m := b.Create(orig)
		if elems := Elems(m); !reflect.DeepEqual(orig, elems) {
			t.Errorf("Elems(%v) got %q. want %q ", m, elems, orig)
		}
	}
}

func TestRange(t *testing.T) {
	b := NewBuilder()
	m := b.Create(map[uint64]interface{}{1: "x", 3: "y", 5: "z", 6: "stop", 8: "a"})

	calls := 0
	cb := func(k uint64, v interface{}) bool {
		t.Logf("visiting (%d, %v)", k, v)
		calls++
		return k%2 != 0 // stop after the first even number.
	}
	// The nodes are visited in increasing order.
	all := m.Range(cb)
	if all {
		t.Error("expected to stop early")
	}
	want := 4
	if calls != want {
		t.Errorf("# of callbacks (%d) was expected to equal %d (1 + # of evens)",
			calls, want)
	}
}

func TestDeepEqual(t *testing.T) {
	for _, m := range []map[uint64]interface{}{
		{},
		{1: "x"},
		{1: "x", 2: "y"},
	} {
		l := NewBuilder().Create(m)
		r := NewBuilder().Create(m)
		if !l.DeepEqual(r) {
			t.Errorf("Expect %v to be DeepEqual() to %v", l, r)
		}
	}
}

func TestNotDeepEqual(t *testing.T) {
	for _, c := range []struct {
		left  map[uint64]interface{}
		right map[uint64]interface{}
	}{
		{
			map[uint64]interface{}{1: "x"},
			map[uint64]interface{}{},
		},
		{
			map[uint64]interface{}{},
			map[uint64]interface{}{1: "y"},
		},
		{
			map[uint64]interface{}{1: "x"},
			map[uint64]interface{}{1: "y"},
		},
		{
			map[uint64]interface{}{1: "x"},
			map[uint64]interface{}{1: "x", 2: "Y"},
		},
		{
			map[uint64]interface{}{1: "x", 2: "Y"},
			map[uint64]interface{}{1: "x"},
		},
		{
			map[uint64]interface{}{1: "x", 2: "y"},
			map[uint64]interface{}{1: "x", 2: "Y"},
		},
	} {
		l := NewBuilder().Create(c.left)
		r := NewBuilder().Create(c.right)
		if l.DeepEqual(r) {
			t.Errorf("Expect %v to be !DeepEqual() to %v", l, r)
		}
	}
}

func TestMerge(t *testing.T) {
	b := NewBuilder()
	for _, c := range []struct {
		left  map[uint64]interface{}
		right map[uint64]interface{}
		want  string
	}{
		{
			map[uint64]interface{}{},
			map[uint64]interface{}{},
			"{}",
		},
		{
			map[uint64]interface{}{},
			map[uint64]interface{}{1: "a"},
			"{1: a}",
		},
		{
			map[uint64]interface{}{1: "a"},
			map[uint64]interface{}{},
			"{1: a}",
		},
		{
			map[uint64]interface{}{1: "a", 2: "b"},
			map[uint64]interface{}{},
			"{1: a, 2: b}",
		},
		{
			map[uint64]interface{}{1: "x"},
			map[uint64]interface{}{1: "y"},
			"{1: x}", // default collision is left
		},
		{
			map[uint64]interface{}{1: "x"},
			map[uint64]interface{}{2: "y"},
			"{1: x, 2: y}",
		},
		{
			map[uint64]interface{}{4: "y", 5: "z"},
			map[uint64]interface{}{1: "x"},
			"{1: x, 4: y, 5: z}",
		},
		{
			map[uint64]interface{}{1: "x", 5: "z"},
			map[uint64]interface{}{4: "y"},
			"{1: x, 4: y, 5: z}",
		},
		{
			map[uint64]interface{}{1: "x", 4: "y"},
			map[uint64]interface{}{5: "z"},
			"{1: x, 4: y, 5: z}",
		},
		{
			map[uint64]interface{}{1: "a", 4: "c"},
			map[uint64]interface{}{2: "b", 5: "d"},
			"{1: a, 2: b, 4: c, 5: d}",
		},
		{
			map[uint64]interface{}{1: "a", 4: "c"},
			map[uint64]interface{}{2: "b", 5 + 8: "d"},
			"{1: a, 2: b, 4: c, 13: d}",
		},
		{
			map[uint64]interface{}{2: "b", 5 + 8: "d"},
			map[uint64]interface{}{1: "a", 4: "c"},
			"{1: a, 2: b, 4: c, 13: d}",
		},
		{
			map[uint64]interface{}{1: "a", 4: "c"},
			map[uint64]interface{}{2: "b", 5 + 8: "d"},
			"{1: a, 2: b, 4: c, 13: d}",
		},
		{
			map[uint64]interface{}{2: "b", 5 + 8: "d"},
			map[uint64]interface{}{1: "a", 4: "c"},
			"{1: a, 2: b, 4: c, 13: d}",
		},
		{
			map[uint64]interface{}{2: "b", 5 + 8: "d"},
			map[uint64]interface{}{2: "", 3: "a"},
			"{2: b, 3: a, 13: d}",
		},

		{
			// crafted for `!prefixesOverlap(p, m, q, n)`
			left:  map[uint64]interface{}{1: "a", 2 + 1: "b"},
			right: map[uint64]interface{}{4 + 1: "c", 4 + 2: "d"},
			// p: 5, m: 2 q: 1, n: 2
			want: "{1: a, 3: b, 5: c, 6: d}",
		},
		{
			// crafted for `ord(m, n) && !zeroBit(q, m)`
			left:  map[uint64]interface{}{8 + 2 + 1: "a", 16 + 4: "b"},
			right: map[uint64]interface{}{16 + 8 + 2 + 1: "c", 16 + 8 + 4 + 2 + 1: "d"},
			// left: p: 15, m: 16
			// right: q: 27, n: 4
			want: "{11: a, 20: b, 27: c, 31: d}",
		},
		{
			// crafted for `ord(n, m) && !zeroBit(p, n)`
			// p: 6, m: 1 q: 5, n: 2
			left:  map[uint64]interface{}{4 + 2: "b", 4 + 2 + 1: "c"},
			right: map[uint64]interface{}{4: "a", 4 + 2 + 1: "dropped"},
			want:  "{4: a, 6: b, 7: c}",
		},
	} {
		l, r := b.Create(c.left), b.Create(c.right)
		m := b.Merge(l, r)
		if got := m.String(); got != c.want {
			t.Errorf("Merge(%s, %s) got %q. want %q ", l, r, got, c.want)
		}
	}
}

func TestIntersect(t *testing.T) {
	// Most of the test cases go after specific branches of intersect.
	b := NewBuilder()
	for _, c := range []struct {
		left  map[uint64]interface{}
		right map[uint64]interface{}
		want  string
	}{
		{
			left:  map[uint64]interface{}{10: "a", 39: "b"},
			right: map[uint64]interface{}{10: "A", 39: "B", 75: "C"},
			want:  "{10: a, 39: b}",
		},
		{
			left:  map[uint64]interface{}{10: "a", 39: "b"},
			right: map[uint64]interface{}{},
			want:  "{}",
		},
		{
			left:  map[uint64]interface{}{},
			right: map[uint64]interface{}{10: "A", 39: "B", 75: "C"},
			want:  "{}",
		},
		{ // m == n && p == q  && left.(*empty) case
			left:  map[uint64]interface{}{4: 1, 6: 3, 10: 8, 15: "on left"},
			right: map[uint64]interface{}{0: 8, 7: 6, 11: 0, 15: "on right"},
			want:  "{15: on left}",
		},
		{ // m == n && p == q  && right.(*empty) case
			left:  map[uint64]interface{}{0: "on left", 1: 2, 2: 3, 3: 1, 7: 3},
			right: map[uint64]interface{}{0: "on right", 5: 1, 6: 8},
			want:  "{0: on left}",
		},
		{ // m == n && p == q  &&  both left and right are not empty
			left:  map[uint64]interface{}{1: "a", 2: "b", 3: "c"},
			right: map[uint64]interface{}{0: "A", 1: "B", 2: "C"},
			want:  "{1: a, 2: b}",
		},
		{ // m == n && p == q  &&  both left and right are not empty
			left:  map[uint64]interface{}{1: "a", 2: "b", 3: "c"},
			right: map[uint64]interface{}{0: "A", 1: "B", 2: "C"},
			want:  "{1: a, 2: b}",
		},
		{ // !prefixesOverlap(p, m, q, n)
			// p = 1, m = 2, q = 5, n = 2
			left:  map[uint64]interface{}{0b001: 1, 0b011: 3},
			right: map[uint64]interface{}{0b100: 4, 0b111: 7},
			want:  "{}",
		},
		{ // ord(m, n) && zeroBit(q, m)
			// p = 3, m = 4, q = 0, n = 1
			left:  map[uint64]interface{}{0b010: 2, 0b101: 5},
			right: map[uint64]interface{}{0b000: 0, 0b001: 1},
			want:  "{}",
		},

		{ // ord(m, n) && !zeroBit(q, m)
			// p = 29, m = 2, q = 30, n = 1
			left: map[uint64]interface{}{
				0b11101: "29",
				0b11110: "30",
			},
			right: map[uint64]interface{}{
				0b11110: "30 on right",
				0b11111: "31",
			},
			want: "{30: 30}",
		},
		{ // ord(n, m) && zeroBit(p, n)
			// p = 5, m = 2, q = 3, n = 4
			left:  map[uint64]interface{}{0b000: 0, 0b001: 1},
			right: map[uint64]interface{}{0b010: 2, 0b101: 5},
			want:  "{}",
		},
		{ // default case
			// p = 5, m = 2, q = 3, n = 4
			left:  map[uint64]interface{}{0b100: 1, 0b110: 3},
			right: map[uint64]interface{}{0b000: 8, 0b111: 6},
			want:  "{}",
		},
	} {
		l, r := b.Create(c.left), b.Create(c.right)
		m := b.Intersect(l, r)
		if got := m.String(); got != c.want {
			t.Errorf("Intersect(%s, %s) got %q. want %q ", l, r, got, c.want)
		}
	}
}

func TestIntersectWith(t *testing.T) {
	b := NewBuilder()
	l := b.Create(map[uint64]interface{}{10: 2.0, 39: 32.0})
	r := b.Create(map[uint64]interface{}{10: 6.0, 39: 10.0, 75: 1.0})

	prodIfDifferent := func(x interface{}, y interface{}) interface{} {
		if x, ok := x.(float64); ok {
			if y, ok := y.(float64); ok {
				if x == y {
					return x
				}
				return x * y
			}
		}
		return x
	}

	m := b.IntersectWith(prodIfDifferent, l, r)

	want := "{10: %!s(float64=12), 39: %!s(float64=320)}"
	if got := m.String(); got != want {
		t.Errorf("IntersectWith(min, %s, %s) got %q. want %q ", l, r, got, want)
	}
}

func TestRemove(t *testing.T) {
	// Most of the test cases go after specific branches of intersect.
	b := NewBuilder()
	for _, c := range []struct {
		m    map[uint64]interface{}
		key  uint64
		want string
	}{
		{map[uint64]interface{}{}, 10, "{}"},
		{map[uint64]interface{}{10: "a"}, 10, "{}"},
		{map[uint64]interface{}{39: "b"}, 10, "{39: b}"},
		// Branch cases:
		// !matchPrefix(kp, br.prefix, br.branching)
		{map[uint64]interface{}{10: "a", 39: "b"}, 128, "{10: a, 39: b}"},
		// case: left == br.left && right == br.right
		{map[uint64]interface{}{10: "a", 39: "b"}, 16, "{10: a, 39: b}"},
		// left updated and is empty.
		{map[uint64]interface{}{10: "a", 39: "b"}, 10, "{39: b}"},
		// right updated and is empty.
		{map[uint64]interface{}{10: "a", 39: "b"}, 39, "{10: a}"},
		// final b.mkBranch(...) case.
		{map[uint64]interface{}{10: "a", 39: "b", 128: "c"}, 39, "{10: a, 128: c}"},
	} {
		pre := b.Create(c.m)
		post := b.Remove(pre, c.key)
		if got := post.String(); got != c.want {
			t.Errorf("Remove(%s, %d) got %q. want %q ", pre, c.key, got, c.want)
		}
	}
}

func TestRescope(t *testing.T) {
	b := NewBuilder()
	l := b.Create(map[uint64]interface{}{10: "a", 39: "b"})
	r := b.Create(map[uint64]interface{}{10: "A", 39: "B", 75: "C"})

	b.Rescope()

	m := b.Intersect(l, r)
	if got, want := m.String(), "{10: a, 39: b}"; got != want {
		t.Errorf("Intersect(%s, %s) got %q. want %q", l, r, got, want)
	}
	if m.Scope() == l.Scope() {
		t.Errorf("m.Scope() = %v should not equal l.Scope() = %v", m.Scope(), l.Scope())
	}
	if m.Scope() == r.Scope() {
		t.Errorf("m.Scope() = %v should not equal r.Scope() = %v", m.Scope(), r.Scope())
	}
}

func TestSharing(t *testing.T) {
	b := NewBuilder()
	l := b.Create(map[uint64]interface{}{0: "a", 1: "b"})
	r := b.Create(map[uint64]interface{}{1: "B", 2: "C"})

	rleftold := r.n.(*branch).left

	m := b.Merge(l, r)
	if mleft := m.n.(*branch).left; mleft != l.n {
		t.Errorf("unexpected value for left branch of %v. want %v got %v", m, l, mleft)
	}

	if rleftnow := r.n.(*branch).left; rleftnow != rleftold {
		t.Errorf("r.n.(*branch).left was modified by the Merge operation. was %v now %v", rleftold, rleftnow)
	}
}
