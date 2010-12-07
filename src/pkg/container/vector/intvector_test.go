// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// CAUTION: If this file is not vector_test.go, it was generated
// automatically from vector_test.go - DO NOT EDIT in that case!

package vector

import "testing"


func TestIntZeroLen(t *testing.T) {
	a := new(IntVector)
	if a.Len() != 0 {
		t.Errorf("%T: B1) expected 0, got %d", a, a.Len())
	}
	if len(*a) != 0 {
		t.Errorf("%T: B2) expected 0, got %d", a, len(*a))
	}
	var b IntVector
	if b.Len() != 0 {
		t.Errorf("%T: B3) expected 0, got %d", b, b.Len())
	}
	if len(b) != 0 {
		t.Errorf("%T: B4) expected 0, got %d", b, len(b))
	}
}


func TestIntResize(t *testing.T) {
	var a IntVector
	checkSize(t, &a, 0, 0)
	checkSize(t, a.Resize(0, 5), 0, 5)
	checkSize(t, a.Resize(1, 0), 1, 5)
	checkSize(t, a.Resize(10, 0), 10, 10)
	checkSize(t, a.Resize(5, 0), 5, 10)
	checkSize(t, a.Resize(3, 8), 3, 10)
	checkSize(t, a.Resize(0, 100), 0, 100)
	checkSize(t, a.Resize(11, 100), 11, 100)
}


func TestIntResize2(t *testing.T) {
	var a IntVector
	checkSize(t, &a, 0, 0)
	a.Push(int2IntValue(1))
	a.Push(int2IntValue(2))
	a.Push(int2IntValue(3))
	a.Push(int2IntValue(4))
	checkSize(t, &a, 4, 4)
	checkSize(t, a.Resize(10, 0), 10, 10)
	for i := 4; i < a.Len(); i++ {
		if a.At(i) != intzero {
			t.Errorf("%T: expected a.At(%d) == %v; found %v!", a, i, intzero, a.At(i))
		}
	}
	for i := 4; i < len(a); i++ {
		if a[i] != intzero {
			t.Errorf("%T: expected a[%d] == %v; found %v", a, i, intzero, a[i])
		}
	}
}


func checkIntZero(t *testing.T, a *IntVector, i int) {
	for j := 0; j < i; j++ {
		if a.At(j) == intzero {
			t.Errorf("%T: 1 expected a.At(%d) == %d; found %v", a, j, j, a.At(j))
		}
		if (*a)[j] == intzero {
			t.Errorf("%T: 2 expected (*a)[%d] == %d; found %v", a, j, j, (*a)[j])
		}
	}
	for ; i < a.Len(); i++ {
		if a.At(i) != intzero {
			t.Errorf("%T: 3 expected a.At(%d) == %v; found %v", a, i, intzero, a.At(i))
		}
		if (*a)[i] != intzero {
			t.Errorf("%T: 4 expected (*a)[%d] == %v; found %v", a, i, intzero, (*a)[i])
		}
	}
}


func TestIntTrailingElements(t *testing.T) {
	var a IntVector
	for i := 0; i < 10; i++ {
		a.Push(int2IntValue(i + 1))
	}
	checkIntZero(t, &a, 10)
	checkSize(t, &a, 10, 16)
	checkSize(t, a.Resize(5, 0), 5, 16)
	checkSize(t, a.Resize(10, 0), 10, 16)
	checkIntZero(t, &a, 5)
}


func TestIntAccess(t *testing.T) {
	const n = 100
	var a IntVector
	a.Resize(n, 0)
	for i := 0; i < n; i++ {
		a.Set(i, int2IntValue(val(i)))
	}
	for i := 0; i < n; i++ {
		if elem2IntValue(a.At(i)) != int2IntValue(val(i)) {
			t.Error(i)
		}
	}
	var b IntVector
	b.Resize(n, 0)
	for i := 0; i < n; i++ {
		b[i] = int2IntValue(val(i))
	}
	for i := 0; i < n; i++ {
		if elem2IntValue(b[i]) != int2IntValue(val(i)) {
			t.Error(i)
		}
	}
}


func TestIntInsertDeleteClear(t *testing.T) {
	const n = 100
	var a IntVector

	for i := 0; i < n; i++ {
		if a.Len() != i {
			t.Errorf("%T: A) wrong Len() %d (expected %d)", a, a.Len(), i)
		}
		if len(a) != i {
			t.Errorf("%T: A) wrong len() %d (expected %d)", a, len(a), i)
		}
		a.Insert(0, int2IntValue(val(i)))
		if elem2IntValue(a.Last()) != int2IntValue(val(0)) {
			t.Errorf("%T: B", a)
		}
	}
	for i := n - 1; i >= 0; i-- {
		if elem2IntValue(a.Last()) != int2IntValue(val(0)) {
			t.Errorf("%T: C", a)
		}
		if elem2IntValue(a.At(0)) != int2IntValue(val(i)) {
			t.Errorf("%T: D", a)
		}
		if elem2IntValue(a[0]) != int2IntValue(val(i)) {
			t.Errorf("%T: D2", a)
		}
		a.Delete(0)
		if a.Len() != i {
			t.Errorf("%T: E) wrong Len() %d (expected %d)", a, a.Len(), i)
		}
		if len(a) != i {
			t.Errorf("%T: E) wrong len() %d (expected %d)", a, len(a), i)
		}
	}

	if a.Len() != 0 {
		t.Errorf("%T: F) wrong Len() %d (expected 0)", a, a.Len())
	}
	if len(a) != 0 {
		t.Errorf("%T: F) wrong len() %d (expected 0)", a, len(a))
	}
	for i := 0; i < n; i++ {
		a.Push(int2IntValue(val(i)))
		if a.Len() != i+1 {
			t.Errorf("%T: G) wrong Len() %d (expected %d)", a, a.Len(), i+1)
		}
		if len(a) != i+1 {
			t.Errorf("%T: G) wrong len() %d (expected %d)", a, len(a), i+1)
		}
		if elem2IntValue(a.Last()) != int2IntValue(val(i)) {
			t.Errorf("%T: H", a)
		}
	}
	a.Resize(0, 0)
	if a.Len() != 0 {
		t.Errorf("%T: I wrong Len() %d (expected 0)", a, a.Len())
	}
	if len(a) != 0 {
		t.Errorf("%T: I wrong len() %d (expected 0)", a, len(a))
	}

	const m = 5
	for j := 0; j < m; j++ {
		a.Push(int2IntValue(j))
		for i := 0; i < n; i++ {
			x := val(i)
			a.Push(int2IntValue(x))
			if elem2IntValue(a.Pop()) != int2IntValue(x) {
				t.Errorf("%T: J", a)
			}
			if a.Len() != j+1 {
				t.Errorf("%T: K) wrong Len() %d (expected %d)", a, a.Len(), j+1)
			}
			if len(a) != j+1 {
				t.Errorf("%T: K) wrong len() %d (expected %d)", a, len(a), j+1)
			}
		}
	}
	if a.Len() != m {
		t.Errorf("%T: L) wrong Len() %d (expected %d)", a, a.Len(), m)
	}
	if len(a) != m {
		t.Errorf("%T: L) wrong len() %d (expected %d)", a, len(a), m)
	}
}


func verify_sliceInt(t *testing.T, x *IntVector, elt, i, j int) {
	for k := i; k < j; k++ {
		if elem2IntValue(x.At(k)) != int2IntValue(elt) {
			t.Errorf("%T: M) wrong [%d] element %v (expected %v)", x, k, elem2IntValue(x.At(k)), int2IntValue(elt))
		}
	}

	s := x.Slice(i, j)
	for k, n := 0, j-i; k < n; k++ {
		if elem2IntValue(s.At(k)) != int2IntValue(elt) {
			t.Errorf("%T: N) wrong [%d] element %v (expected %v)", x, k, elem2IntValue(x.At(k)), int2IntValue(elt))
		}
	}
}


func verify_patternInt(t *testing.T, x *IntVector, a, b, c int) {
	n := a + b + c
	if x.Len() != n {
		t.Errorf("%T: O) wrong Len() %d (expected %d)", x, x.Len(), n)
	}
	if len(*x) != n {
		t.Errorf("%T: O) wrong len() %d (expected %d)", x, len(*x), n)
	}
	verify_sliceInt(t, x, 0, 0, a)
	verify_sliceInt(t, x, 1, a, a+b)
	verify_sliceInt(t, x, 0, a+b, n)
}


func make_vectorInt(elt, len int) *IntVector {
	x := new(IntVector).Resize(len, 0)
	for i := 0; i < len; i++ {
		x.Set(i, int2IntValue(elt))
	}
	return x
}


func TestIntInsertVector(t *testing.T) {
	// 1
	a := make_vectorInt(0, 0)
	b := make_vectorInt(1, 10)
	a.InsertVector(0, b)
	verify_patternInt(t, a, 0, 10, 0)
	// 2
	a = make_vectorInt(0, 10)
	b = make_vectorInt(1, 0)
	a.InsertVector(5, b)
	verify_patternInt(t, a, 5, 0, 5)
	// 3
	a = make_vectorInt(0, 10)
	b = make_vectorInt(1, 3)
	a.InsertVector(3, b)
	verify_patternInt(t, a, 3, 3, 7)
	// 4
	a = make_vectorInt(0, 10)
	b = make_vectorInt(1, 1000)
	a.InsertVector(8, b)
	verify_patternInt(t, a, 8, 1000, 2)
}


func TestIntDo(t *testing.T) {
	const n = 25
	const salt = 17
	a := new(IntVector).Resize(n, 0)
	for i := 0; i < n; i++ {
		a.Set(i, int2IntValue(salt*i))
	}
	count := 0
	a.Do(func(e int) {
		i := intf2IntValue(e)
		if i != int2IntValue(count*salt) {
			t.Error(tname(a), "value at", count, "should be", count*salt, "not", i)
		}
		count++
	})
	if count != n {
		t.Error(tname(a), "should visit", n, "values; did visit", count)
	}

	b := new(IntVector).Resize(n, 0)
	for i := 0; i < n; i++ {
		(*b)[i] = int2IntValue(salt * i)
	}
	count = 0
	b.Do(func(e int) {
		i := intf2IntValue(e)
		if i != int2IntValue(count*salt) {
			t.Error(tname(b), "b) value at", count, "should be", count*salt, "not", i)
		}
		count++
	})
	if count != n {
		t.Error(tname(b), "b) should visit", n, "values; did visit", count)
	}

	var c IntVector
	c.Resize(n, 0)
	for i := 0; i < n; i++ {
		c[i] = int2IntValue(salt * i)
	}
	count = 0
	c.Do(func(e int) {
		i := intf2IntValue(e)
		if i != int2IntValue(count*salt) {
			t.Error(tname(c), "c) value at", count, "should be", count*salt, "not", i)
		}
		count++
	})
	if count != n {
		t.Error(tname(c), "c) should visit", n, "values; did visit", count)
	}

}


func TestIntVectorCopy(t *testing.T) {
	// verify Copy() returns a copy, not simply a slice of the original vector
	const Len = 10
	var src IntVector
	for i := 0; i < Len; i++ {
		src.Push(int2IntValue(i * i))
	}
	dest := src.Copy()
	for i := 0; i < Len; i++ {
		src[i] = int2IntValue(-1)
		v := elem2IntValue(dest[i])
		if v != int2IntValue(i*i) {
			t.Error(tname(src), "expected", i*i, "got", v)
		}
	}
}
