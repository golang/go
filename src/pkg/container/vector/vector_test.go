// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package vector

import "testing"
import "sort"
import "fmt"


func TestZeroLen(t *testing.T) {
	var a *Vector;
	if a.Len() != 0 { t.Errorf("A) expected 0, got %d", a.Len()); }
	a = New(0);
	if a.Len() != 0 { t.Errorf("B) expected 0, got %d", a.Len()); }
}


func TestInit(t *testing.T) {
	var a Vector;
	if a.Init(0).Len() != 0 { t.Error("A") }
	if a.Init(1).Len() != 1 { t.Error("B") }
	if a.Init(10).Len() != 10 { t.Error("C") }
}


func TestNew(t *testing.T) {
	if New(0).Len() != 0 { t.Error("A") }
	if New(1).Len() != 1 { t.Error("B") }
	if New(10).Len() != 10 { t.Error("C") }
}


func val(i int) int {
	return i*991 - 1234
}


func TestAccess(t *testing.T) {
	const n = 100;
	var a Vector;
	a.Init(n);
	for i := 0; i < n; i++ {
		a.Set(i, val(i));
	}
	for i := 0; i < n; i++ {
		if a.At(i).(int) != val(i) { t.Error(i) }
	}
}


func TestInsertDeleteClear(t *testing.T) {
	const n = 100;
	a := New(0);

	for i := 0; i < n; i++ {
		if a.Len() != i { t.Errorf("A) wrong len %d (expected %d)", a.Len(), i) }
		a.Insert(0, val(i));
		if a.Last().(int) != val(0) { t.Error("B") }
	}
	for i := n-1; i >= 0; i-- {
		if a.Last().(int) != val(0) { t.Error("C") }
		if a.At(0).(int) != val(i) { t.Error("D") }
		a.Delete(0);
		if a.Len() != i { t.Errorf("E) wrong len %d (expected %d)", a.Len(), i) }
	}

	if a.Len() != 0 { t.Errorf("F) wrong len %d (expected 0)", a.Len()) }
	for i := 0; i < n; i++ {
		a.Push(val(i));
		if a.Len() != i+1 { t.Errorf("G) wrong len %d (expected %d)", a.Len(), i+1) }
		if a.Last().(int) != val(i) { t.Error("H") }
	}
	a.Init(0);
	if a.Len() != 0 { t.Errorf("I wrong len %d (expected 0)", a.Len()) }

	const m = 5;
	for j := 0; j < m; j++ {
		a.Push(j);
		for i := 0; i < n; i++ {
			x := val(i);
			a.Push(x);
			if a.Pop().(int) != x { t.Error("J") }
			if a.Len() != j+1 { t.Errorf("K) wrong len %d (expected %d)", a.Len(), j+1) }
		}
	}
	if a.Len() != m { t.Errorf("L) wrong len %d (expected %d)", a.Len(), m) }
}


func verify_slice(t *testing.T, x *Vector, elt, i, j int) {
	for k := i; k < j; k++ {
		if x.At(k).(int) != elt {
			t.Errorf("M) wrong [%d] element %d (expected %d)", k, x.At(k).(int), elt)
		}
	}

	s := x.Slice(i, j);
	for k, n := 0, j-i; k < n; k++ {
		if s.At(k).(int) != elt {
			t.Errorf("N) wrong [%d] element %d (expected %d)", k, x.At(k).(int), elt)
		}
	}
}


func verify_pattern(t *testing.T, x *Vector, a, b, c int) {
	n := a + b + c;
	if x.Len() != n {
		t.Errorf("O) wrong len %d (expected %d)", x.Len(), n)
	}
	verify_slice(t, x, 0, 0, a);
	verify_slice(t, x, 1, a, a + b);
	verify_slice(t, x, 0, a + b, n);
}


func make_vector(elt, len int) *Vector {
	x := New(len);
	for i := 0; i < len; i++ {
		x.Set(i, elt);
	}
	return x;
}


func TestInsertVector(t *testing.T) {
	// 1
	a := make_vector(0, 0);
	b := make_vector(1, 10);
	a.InsertVector(0, b);
	verify_pattern(t, a, 0, 10, 0);
	// 2
	a = make_vector(0, 10);
	b = make_vector(1, 0);
	a.InsertVector(5, b);
	verify_pattern(t, a, 5, 0, 5);
	// 3
	a = make_vector(0, 10);
	b = make_vector(1, 3);
	a.InsertVector(3, b);
	verify_pattern(t, a, 3, 3, 7);
	// 4
	a = make_vector(0, 10);
	b = make_vector(1, 1000);
	a.InsertVector(8, b);
	verify_pattern(t, a, 8, 1000, 2);
}


// This also tests IntVector and StringVector
func TestSorting(t *testing.T) {
	const n = 100;

	a := NewIntVector(n);
	for i := n-1; i >= 0; i-- {
		a.Set(i, n-1-i);
	}
	if sort.IsSorted(a) { t.Error("int vector not sorted") }

	b := NewStringVector(n);
	for i := n-1; i >= 0; i-- {
		b.Set(i, fmt.Sprint(n-1-i));
	}
	if sort.IsSorted(b) { t.Error("string vector not sorted") }
}


func TestDo(t *testing.T) {
	const n = 25;
	const salt = 17;
	a := NewIntVector(n);
	for i := 0; i < n; i++ {
		a.Set(i, salt * i);
	}
	count := 0;
	a.Do(
		func(e Element) {
			i := e.(int);
			if i != count*salt {
				t.Error("value at", count, "should be", count*salt, "not", i)
			}
			count++;
		}
	);
	if count != n {
		t.Error("should visit", n, "values; did visit", count)
	}
}

func TestIter(t *testing.T) {
	const Len = 100;
	x := New(Len);
	for i := 0; i < Len; i++ {
		x.Set(i, i*i);
	}
	i := 0;
	for v := range x.Iter() {
		if v.(int) != i*i {
			t.Error("Iter expected", i*i, "got", v.(int))
		}
		i++;
	}
	if i != Len {
		t.Error("Iter stopped at", i, "not", Len)
	}
}
