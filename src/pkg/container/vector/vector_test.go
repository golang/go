// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package vector

import "testing"
import "sort"
import "fmt"


func TestZeroLen(t *testing.T) {
	a := new(Vector);
	if a.Len() != 0 {
		t.Errorf("B) expected 0, got %d", a.Len())
	}
}


type VectorInterface interface {
	Len() int;
	Cap() int;
}


func checkSize(t *testing.T, v VectorInterface, len, cap int) {
	if v.Len() != len {
		t.Errorf("expected len = %d; found %d", len, v.Len())
	}
	if v.Cap() < cap {
		t.Errorf("expected cap >= %d; found %d", cap, v.Cap())
	}
}


func TestResize(t *testing.T) {
	var a Vector;
	checkSize(t, &a, 0, 0);
	checkSize(t, a.Resize(0, 5), 0, 5);
	checkSize(t, a.Resize(1, 0), 1, 5);
	checkSize(t, a.Resize(10, 0), 10, 10);
	checkSize(t, a.Resize(5, 0), 5, 10);
	checkSize(t, a.Resize(3, 8), 3, 10);
	checkSize(t, a.Resize(0, 100), 0, 100);
	checkSize(t, a.Resize(11, 100), 11, 100);
}


func TestIntResize(t *testing.T) {
	var a IntVector;
	checkSize(t, &a, 0, 0);
	a.Push(1);
	a.Push(2);
	a.Push(3);
	a.Push(4);
	checkSize(t, &a, 4, 4);
	checkSize(t, a.Resize(10, 0), 10, 10);
	for i := 4; i < a.Len(); i++ {
		if a.At(i) != 0 {
			t.Errorf("expected a.At(%d) == 0; found %d", i, a.At(i))
		}
	}
}


func TestStringResize(t *testing.T) {
	var a StringVector;
	checkSize(t, &a, 0, 0);
	a.Push("1");
	a.Push("2");
	a.Push("3");
	a.Push("4");
	checkSize(t, &a, 4, 4);
	checkSize(t, a.Resize(10, 0), 10, 10);
	for i := 4; i < a.Len(); i++ {
		if a.At(i) != "" {
			t.Errorf("expected a.At(%d) == " "; found %s", i, a.At(i))
		}
	}
}


func checkNil(t *testing.T, a *Vector, i int) {
	for j := 0; j < i; j++ {
		if a.At(j) == nil {
			t.Errorf("expected a.At(%d) == %d; found %v", j, j, a.At(j))
		}
	}
	for ; i < a.Len(); i++ {
		if a.At(i) != nil {
			t.Errorf("expected a.At(%d) == nil; found %v", i, a.At(i))
		}
	}
}


func TestTrailingElements(t *testing.T) {
	var a Vector;
	for i := 0; i < 10; i++ {
		a.Push(i)
	}
	checkNil(t, &a, 10);
	checkSize(t, &a, 10, 16);
	checkSize(t, a.Resize(5, 0), 5, 16);
	checkSize(t, a.Resize(10, 0), 10, 16);
	checkNil(t, &a, 5);
}


func val(i int) int	{ return i*991 - 1234 }


func TestAccess(t *testing.T) {
	const n = 100;
	var a Vector;
	a.Resize(n, 0);
	for i := 0; i < n; i++ {
		a.Set(i, val(i))
	}
	for i := 0; i < n; i++ {
		if a.At(i).(int) != val(i) {
			t.Error(i)
		}
	}
}


func TestInsertDeleteClear(t *testing.T) {
	const n = 100;
	var a Vector;

	for i := 0; i < n; i++ {
		if a.Len() != i {
			t.Errorf("A) wrong len %d (expected %d)", a.Len(), i)
		}
		a.Insert(0, val(i));
		if a.Last().(int) != val(0) {
			t.Error("B")
		}
	}
	for i := n - 1; i >= 0; i-- {
		if a.Last().(int) != val(0) {
			t.Error("C")
		}
		if a.At(0).(int) != val(i) {
			t.Error("D")
		}
		a.Delete(0);
		if a.Len() != i {
			t.Errorf("E) wrong len %d (expected %d)", a.Len(), i)
		}
	}

	if a.Len() != 0 {
		t.Errorf("F) wrong len %d (expected 0)", a.Len())
	}
	for i := 0; i < n; i++ {
		a.Push(val(i));
		if a.Len() != i+1 {
			t.Errorf("G) wrong len %d (expected %d)", a.Len(), i+1)
		}
		if a.Last().(int) != val(i) {
			t.Error("H")
		}
	}
	a.Resize(0, 0);
	if a.Len() != 0 {
		t.Errorf("I wrong len %d (expected 0)", a.Len())
	}

	const m = 5;
	for j := 0; j < m; j++ {
		a.Push(j);
		for i := 0; i < n; i++ {
			x := val(i);
			a.Push(x);
			if a.Pop().(int) != x {
				t.Error("J")
			}
			if a.Len() != j+1 {
				t.Errorf("K) wrong len %d (expected %d)", a.Len(), j+1)
			}
		}
	}
	if a.Len() != m {
		t.Errorf("L) wrong len %d (expected %d)", a.Len(), m)
	}
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
	verify_slice(t, x, 1, a, a+b);
	verify_slice(t, x, 0, a+b, n);
}


func make_vector(elt, len int) *Vector {
	x := new(Vector).Resize(len, 0);
	for i := 0; i < len; i++ {
		x.Set(i, elt)
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

	a := new(IntVector).Resize(n, 0);
	for i := n - 1; i >= 0; i-- {
		a.Set(i, n-1-i)
	}
	if sort.IsSorted(a) {
		t.Error("int vector not sorted")
	}

	b := new(StringVector).Resize(n, 0);
	for i := n - 1; i >= 0; i-- {
		b.Set(i, fmt.Sprint(n-1-i))
	}
	if sort.IsSorted(b) {
		t.Error("string vector not sorted")
	}
}


func TestDo(t *testing.T) {
	const n = 25;
	const salt = 17;
	a := new(IntVector).Resize(n, 0);
	for i := 0; i < n; i++ {
		a.Set(i, salt*i)
	}
	count := 0;
	a.Do(func(e interface{}) {
		i := e.(int);
		if i != count*salt {
			t.Error("value at", count, "should be", count*salt, "not", i)
		}
		count++;
	});
	if count != n {
		t.Error("should visit", n, "values; did visit", count)
	}
}


func TestIter(t *testing.T) {
	const Len = 100;
	x := new(Vector).Resize(Len, 0);
	for i := 0; i < Len; i++ {
		x.Set(i, i*i)
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
