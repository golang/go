// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// CAUTION: If this file is not vector_test.go, it was generated
// automatically from vector_test.go - DO NOT EDIT in that case!

package vector

import "testing"


func TestStrZeroLen(t *testing.T) {
	a := new(StringVector)
	if a.Len() != 0 {
		t.Errorf("%T: B1) expected 0, got %d", a, a.Len())
	}
	if len(*a) != 0 {
		t.Errorf("%T: B2) expected 0, got %d", a, len(*a))
	}
	var b StringVector
	if b.Len() != 0 {
		t.Errorf("%T: B3) expected 0, got %d", b, b.Len())
	}
	if len(b) != 0 {
		t.Errorf("%T: B4) expected 0, got %d", b, len(b))
	}
}


func TestStrResize(t *testing.T) {
	var a StringVector
	checkSize(t, &a, 0, 0)
	checkSize(t, a.Resize(0, 5), 0, 5)
	checkSize(t, a.Resize(1, 0), 1, 5)
	checkSize(t, a.Resize(10, 0), 10, 10)
	checkSize(t, a.Resize(5, 0), 5, 10)
	checkSize(t, a.Resize(3, 8), 3, 10)
	checkSize(t, a.Resize(0, 100), 0, 100)
	checkSize(t, a.Resize(11, 100), 11, 100)
}


func TestStrResize2(t *testing.T) {
	var a StringVector
	checkSize(t, &a, 0, 0)
	a.Push(int2StrValue(1))
	a.Push(int2StrValue(2))
	a.Push(int2StrValue(3))
	a.Push(int2StrValue(4))
	checkSize(t, &a, 4, 4)
	checkSize(t, a.Resize(10, 0), 10, 10)
	for i := 4; i < a.Len(); i++ {
		if a.At(i) != strzero {
			t.Errorf("%T: expected a.At(%d) == %v; found %v!", a, i, strzero, a.At(i))
		}
	}
	for i := 4; i < len(a); i++ {
		if a[i] != strzero {
			t.Errorf("%T: expected a[%d] == %v; found %v", a, i, strzero, a[i])
		}
	}
}


func checkStrZero(t *testing.T, a *StringVector, i int) {
	for j := 0; j < i; j++ {
		if a.At(j) == strzero {
			t.Errorf("%T: 1 expected a.At(%d) == %d; found %v", a, j, j, a.At(j))
		}
		if (*a)[j] == strzero {
			t.Errorf("%T: 2 expected (*a)[%d] == %d; found %v", a, j, j, (*a)[j])
		}
	}
	for ; i < a.Len(); i++ {
		if a.At(i) != strzero {
			t.Errorf("%T: 3 expected a.At(%d) == %v; found %v", a, i, strzero, a.At(i))
		}
		if (*a)[i] != strzero {
			t.Errorf("%T: 4 expected (*a)[%d] == %v; found %v", a, i, strzero, (*a)[i])
		}
	}
}


func TestStrTrailingElements(t *testing.T) {
	var a StringVector
	for i := 0; i < 10; i++ {
		a.Push(int2StrValue(i + 1))
	}
	checkStrZero(t, &a, 10)
	checkSize(t, &a, 10, 16)
	checkSize(t, a.Resize(5, 0), 5, 16)
	checkSize(t, a.Resize(10, 0), 10, 16)
	checkStrZero(t, &a, 5)
}


func TestStrAccess(t *testing.T) {
	const n = 100
	var a StringVector
	a.Resize(n, 0)
	for i := 0; i < n; i++ {
		a.Set(i, int2StrValue(val(i)))
	}
	for i := 0; i < n; i++ {
		if elem2StrValue(a.At(i)) != int2StrValue(val(i)) {
			t.Error(i)
		}
	}
	var b StringVector
	b.Resize(n, 0)
	for i := 0; i < n; i++ {
		b[i] = int2StrValue(val(i))
	}
	for i := 0; i < n; i++ {
		if elem2StrValue(b[i]) != int2StrValue(val(i)) {
			t.Error(i)
		}
	}
}


func TestStrInsertDeleteClear(t *testing.T) {
	const n = 100
	var a StringVector

	for i := 0; i < n; i++ {
		if a.Len() != i {
			t.Errorf("%T: A) wrong Len() %d (expected %d)", a, a.Len(), i)
		}
		if len(a) != i {
			t.Errorf("%T: A) wrong len() %d (expected %d)", a, len(a), i)
		}
		a.Insert(0, int2StrValue(val(i)))
		if elem2StrValue(a.Last()) != int2StrValue(val(0)) {
			t.Errorf("%T: B", a)
		}
	}
	for i := n - 1; i >= 0; i-- {
		if elem2StrValue(a.Last()) != int2StrValue(val(0)) {
			t.Errorf("%T: C", a)
		}
		if elem2StrValue(a.At(0)) != int2StrValue(val(i)) {
			t.Errorf("%T: D", a)
		}
		if elem2StrValue(a[0]) != int2StrValue(val(i)) {
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
		a.Push(int2StrValue(val(i)))
		if a.Len() != i+1 {
			t.Errorf("%T: G) wrong Len() %d (expected %d)", a, a.Len(), i+1)
		}
		if len(a) != i+1 {
			t.Errorf("%T: G) wrong len() %d (expected %d)", a, len(a), i+1)
		}
		if elem2StrValue(a.Last()) != int2StrValue(val(i)) {
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
		a.Push(int2StrValue(j))
		for i := 0; i < n; i++ {
			x := val(i)
			a.Push(int2StrValue(x))
			if elem2StrValue(a.Pop()) != int2StrValue(x) {
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


func verify_sliceStr(t *testing.T, x *StringVector, elt, i, j int) {
	for k := i; k < j; k++ {
		if elem2StrValue(x.At(k)) != int2StrValue(elt) {
			t.Errorf("%T: M) wrong [%d] element %v (expected %v)", x, k, elem2StrValue(x.At(k)), int2StrValue(elt))
		}
	}

	s := x.Slice(i, j)
	for k, n := 0, j-i; k < n; k++ {
		if elem2StrValue(s.At(k)) != int2StrValue(elt) {
			t.Errorf("%T: N) wrong [%d] element %v (expected %v)", x, k, elem2StrValue(x.At(k)), int2StrValue(elt))
		}
	}
}


func verify_patternStr(t *testing.T, x *StringVector, a, b, c int) {
	n := a + b + c
	if x.Len() != n {
		t.Errorf("%T: O) wrong Len() %d (expected %d)", x, x.Len(), n)
	}
	if len(*x) != n {
		t.Errorf("%T: O) wrong len() %d (expected %d)", x, len(*x), n)
	}
	verify_sliceStr(t, x, 0, 0, a)
	verify_sliceStr(t, x, 1, a, a+b)
	verify_sliceStr(t, x, 0, a+b, n)
}


func make_vectorStr(elt, len int) *StringVector {
	x := new(StringVector).Resize(len, 0)
	for i := 0; i < len; i++ {
		x.Set(i, int2StrValue(elt))
	}
	return x
}


func TestStrInsertVector(t *testing.T) {
	// 1
	a := make_vectorStr(0, 0)
	b := make_vectorStr(1, 10)
	a.InsertVector(0, b)
	verify_patternStr(t, a, 0, 10, 0)
	// 2
	a = make_vectorStr(0, 10)
	b = make_vectorStr(1, 0)
	a.InsertVector(5, b)
	verify_patternStr(t, a, 5, 0, 5)
	// 3
	a = make_vectorStr(0, 10)
	b = make_vectorStr(1, 3)
	a.InsertVector(3, b)
	verify_patternStr(t, a, 3, 3, 7)
	// 4
	a = make_vectorStr(0, 10)
	b = make_vectorStr(1, 1000)
	a.InsertVector(8, b)
	verify_patternStr(t, a, 8, 1000, 2)
}


func TestStrDo(t *testing.T) {
	const n = 25
	const salt = 17
	a := new(StringVector).Resize(n, 0)
	for i := 0; i < n; i++ {
		a.Set(i, int2StrValue(salt*i))
	}
	count := 0
	a.Do(func(e string) {
		i := intf2StrValue(e)
		if i != int2StrValue(count*salt) {
			t.Error(tname(a), "value at", count, "should be", count*salt, "not", i)
		}
		count++
	})
	if count != n {
		t.Error(tname(a), "should visit", n, "values; did visit", count)
	}

	b := new(StringVector).Resize(n, 0)
	for i := 0; i < n; i++ {
		(*b)[i] = int2StrValue(salt * i)
	}
	count = 0
	b.Do(func(e string) {
		i := intf2StrValue(e)
		if i != int2StrValue(count*salt) {
			t.Error(tname(b), "b) value at", count, "should be", count*salt, "not", i)
		}
		count++
	})
	if count != n {
		t.Error(tname(b), "b) should visit", n, "values; did visit", count)
	}

	var c StringVector
	c.Resize(n, 0)
	for i := 0; i < n; i++ {
		c[i] = int2StrValue(salt * i)
	}
	count = 0
	c.Do(func(e string) {
		i := intf2StrValue(e)
		if i != int2StrValue(count*salt) {
			t.Error(tname(c), "c) value at", count, "should be", count*salt, "not", i)
		}
		count++
	})
	if count != n {
		t.Error(tname(c), "c) should visit", n, "values; did visit", count)
	}

}


func TestStrVectorCopy(t *testing.T) {
	// verify Copy() returns a copy, not simply a slice of the original vector
	const Len = 10
	var src StringVector
	for i := 0; i < Len; i++ {
		src.Push(int2StrValue(i * i))
	}
	dest := src.Copy()
	for i := 0; i < Len; i++ {
		src[i] = int2StrValue(-1)
		v := elem2StrValue(dest[i])
		if v != int2StrValue(i*i) {
			t.Error(tname(src), "expected", i*i, "got", v)
		}
	}
}
