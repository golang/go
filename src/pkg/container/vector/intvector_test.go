// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// CAUTION: If this file is not vector_test.go, it was generated
// automatically from vector_test.go - DO NOT EDIT in that case!

package vector

import "testing"


func TestIntZeroLenExp(t *testing.T) {
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


func TestIntResizeExp(t *testing.T) {
	var a IntVector
	checkSizeExp(t, &a, 0, 0)
	checkSizeExp(t, a.Resize(0, 5), 0, 5)
	checkSizeExp(t, a.Resize(1, 0), 1, 5)
	checkSizeExp(t, a.Resize(10, 0), 10, 10)
	checkSizeExp(t, a.Resize(5, 0), 5, 10)
	checkSizeExp(t, a.Resize(3, 8), 3, 10)
	checkSizeExp(t, a.Resize(0, 100), 0, 100)
	checkSizeExp(t, a.Resize(11, 100), 11, 100)
}


func TestIntResize2Exp(t *testing.T) {
	var a IntVector
	checkSizeExp(t, &a, 0, 0)
	a.Push(int2IntValue(1))
	a.Push(int2IntValue(2))
	a.Push(int2IntValue(3))
	a.Push(int2IntValue(4))
	checkSizeExp(t, &a, 4, 4)
	checkSizeExp(t, a.Resize(10, 0), 10, 10)
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


func checkIntZeroExp(t *testing.T, a *IntVector, i int) {
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


func TestIntTrailingElementsExp(t *testing.T) {
	var a IntVector
	for i := 0; i < 10; i++ {
		a.Push(int2IntValue(i + 1))
	}
	checkIntZeroExp(t, &a, 10)
	checkSizeExp(t, &a, 10, 16)
	checkSizeExp(t, a.Resize(5, 0), 5, 16)
	checkSizeExp(t, a.Resize(10, 0), 10, 16)
	checkIntZeroExp(t, &a, 5)
}


func TestIntAccessExp(t *testing.T) {
	const n = 100
	var a IntVector
	a.Resize(n, 0)
	for i := 0; i < n; i++ {
		a.Set(i, int2IntValue(valExp(i)))
	}
	for i := 0; i < n; i++ {
		if elem2IntValue(a.At(i)) != int2IntValue(valExp(i)) {
			t.Error(i)
		}
	}
	var b IntVector
	b.Resize(n, 0)
	for i := 0; i < n; i++ {
		b[i] = int2IntValue(valExp(i))
	}
	for i := 0; i < n; i++ {
		if elem2IntValue(b[i]) != int2IntValue(valExp(i)) {
			t.Error(i)
		}
	}
}


func TestIntInsertDeleteClearExp(t *testing.T) {
	const n = 100
	var a IntVector

	for i := 0; i < n; i++ {
		if a.Len() != i {
			t.Errorf("T%: A) wrong Len() %d (expected %d)", a, a.Len(), i)
		}
		if len(a) != i {
			t.Errorf("T%: A) wrong len() %d (expected %d)", a, len(a), i)
		}
		a.Insert(0, int2IntValue(valExp(i)))
		if elem2IntValue(a.Last()) != int2IntValue(valExp(0)) {
			t.Error("T%: B", a)
		}
	}
	for i := n - 1; i >= 0; i-- {
		if elem2IntValue(a.Last()) != int2IntValue(valExp(0)) {
			t.Error("T%: C", a)
		}
		if elem2IntValue(a.At(0)) != int2IntValue(valExp(i)) {
			t.Error("T%: D", a)
		}
		if elem2IntValue(a[0]) != int2IntValue(valExp(i)) {
			t.Error("T%: D2", a)
		}
		a.Delete(0)
		if a.Len() != i {
			t.Errorf("T%: E) wrong Len() %d (expected %d)", a, a.Len(), i)
		}
		if len(a) != i {
			t.Errorf("T%: E) wrong len() %d (expected %d)", a, len(a), i)
		}
	}

	if a.Len() != 0 {
		t.Errorf("T%: F) wrong Len() %d (expected 0)", a, a.Len())
	}
	if len(a) != 0 {
		t.Errorf("T%: F) wrong len() %d (expected 0)", a, len(a))
	}
	for i := 0; i < n; i++ {
		a.Push(int2IntValue(valExp(i)))
		if a.Len() != i+1 {
			t.Errorf("T%: G) wrong Len() %d (expected %d)", a, a.Len(), i+1)
		}
		if len(a) != i+1 {
			t.Errorf("T%: G) wrong len() %d (expected %d)", a, len(a), i+1)
		}
		if elem2IntValue(a.Last()) != int2IntValue(valExp(i)) {
			t.Error("T%: H", a)
		}
	}
	a.Resize(0, 0)
	if a.Len() != 0 {
		t.Errorf("T%: I wrong Len() %d (expected 0)", a, a.Len())
	}
	if len(a) != 0 {
		t.Errorf("T%: I wrong len() %d (expected 0)", a, len(a))
	}

	const m = 5
	for j := 0; j < m; j++ {
		a.Push(int2IntValue(j))
		for i := 0; i < n; i++ {
			x := valExp(i)
			a.Push(int2IntValue(x))
			if elem2IntValue(a.Pop()) != int2IntValue(x) {
				t.Error("T%: J", a)
			}
			if a.Len() != j+1 {
				t.Errorf("T%: K) wrong Len() %d (expected %d)", a, a.Len(), j+1)
			}
			if len(a) != j+1 {
				t.Errorf("T%: K) wrong len() %d (expected %d)", a, len(a), j+1)
			}
		}
	}
	if a.Len() != m {
		t.Errorf("T%: L) wrong Len() %d (expected %d)", a, a.Len(), m)
	}
	if len(a) != m {
		t.Errorf("T%: L) wrong len() %d (expected %d)", a, len(a), m)
	}
}


func verify_sliceIntExp(t *testing.T, x *IntVector, elt, i, j int) {
	for k := i; k < j; k++ {
		if elem2IntValue(x.At(k)) != int2IntValue(elt) {
			t.Errorf("T%: M) wrong [%d] element %v (expected %v)", x, k, elem2IntValue(x.At(k)), int2IntValue(elt))
		}
	}

	s := x.Slice(i, j)
	for k, n := 0, j-i; k < n; k++ {
		if elem2IntValue(s.At(k)) != int2IntValue(elt) {
			t.Errorf("T%: N) wrong [%d] element %v (expected %v)", x, k, elem2IntValue(x.At(k)), int2IntValue(elt))
		}
	}
}


func verify_patternIntExp(t *testing.T, x *IntVector, a, b, c int) {
	n := a + b + c
	if x.Len() != n {
		t.Errorf("T%: O) wrong Len() %d (expected %d)", x, x.Len(), n)
	}
	if len(*x) != n {
		t.Errorf("T%: O) wrong len() %d (expected %d)", x, len(*x), n)
	}
	verify_sliceIntExp(t, x, 0, 0, a)
	verify_sliceIntExp(t, x, 1, a, a+b)
	verify_sliceIntExp(t, x, 0, a+b, n)
}


func make_vectorIntExp(elt, len int) *IntVector {
	x := new(IntVector).Resize(len, 0)
	for i := 0; i < len; i++ {
		x.Set(i, int2IntValue(elt))
	}
	return x
}


func TestIntInsertVectorExp(t *testing.T) {
	// 1
	a := make_vectorIntExp(0, 0)
	b := make_vectorIntExp(1, 10)
	a.InsertVector(0, b)
	verify_patternIntExp(t, a, 0, 10, 0)
	// 2
	a = make_vectorIntExp(0, 10)
	b = make_vectorIntExp(1, 0)
	a.InsertVector(5, b)
	verify_patternIntExp(t, a, 5, 0, 5)
	// 3
	a = make_vectorIntExp(0, 10)
	b = make_vectorIntExp(1, 3)
	a.InsertVector(3, b)
	verify_patternIntExp(t, a, 3, 3, 7)
	// 4
	a = make_vectorIntExp(0, 10)
	b = make_vectorIntExp(1, 1000)
	a.InsertVector(8, b)
	verify_patternIntExp(t, a, 8, 1000, 2)
}


func TestIntDoExp(t *testing.T) {
	const n = 25
	const salt = 17
	a := new(IntVector).Resize(n, 0)
	for i := 0; i < n; i++ {
		a.Set(i, int2IntValue(salt*i))
	}
	count := 0
	a.Do(func(e interface{}) {
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
	b.Do(func(e interface{}) {
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
	c.Do(func(e interface{}) {
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


func TestIntIterExp(t *testing.T) {
	const Len = 100
	x := new(IntVector).Resize(Len, 0)
	for i := 0; i < Len; i++ {
		x.Set(i, int2IntValue(i*i))
	}
	i := 0
	for v := range x.Iter() {
		if elem2IntValue(v) != int2IntValue(i*i) {
			t.Error(tname(x), "Iter expected", i*i, "got", elem2IntValue(v))
		}
		i++
	}
	if i != Len {
		t.Error(tname(x), "Iter stopped at", i, "not", Len)
	}
	y := new(IntVector).Resize(Len, 0)
	for i := 0; i < Len; i++ {
		(*y)[i] = int2IntValue(i * i)
	}
	i = 0
	for v := range y.Iter() {
		if elem2IntValue(v) != int2IntValue(i*i) {
			t.Error(tname(y), "y, Iter expected", i*i, "got", elem2IntValue(v))
		}
		i++
	}
	if i != Len {
		t.Error(tname(y), "y, Iter stopped at", i, "not", Len)
	}
	var z IntVector
	z.Resize(Len, 0)
	for i := 0; i < Len; i++ {
		z[i] = int2IntValue(i * i)
	}
	i = 0
	for v := range z.Iter() {
		if elem2IntValue(v) != int2IntValue(i*i) {
			t.Error(tname(z), "z, Iter expected", i*i, "got", elem2IntValue(v))
		}
		i++
	}
	if i != Len {
		t.Error(tname(z), "z, Iter stopped at", i, "not", Len)
	}
}

func TestIntVectorData(t *testing.T) {
	// verify Data() returns a slice of a copy, not a slice of the original vector
	const Len = 10
	var src IntVector
	for i := 0; i < Len; i++ {
		src.Push(int2IntValue(i * i))
	}
	dest := src.Data()
	for i := 0; i < Len; i++ {
		src[i] = int2IntValue(-1)
		v := elem2IntValue(dest[i])
		if v != int2IntValue(i*i) {
			t.Error(tname(src), "expected", i*i, "got", v)
		}
	}
}
