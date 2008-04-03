// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package vector

export Vector, New;

/*
	import vector "vector"
	v := vector.New();
	v.Insert(0, new(Foo));
	v.Append(new(Foo));
	v.Delete(0);
	for i := 0; i < v.Len(); i++ { f(v.At(i)); }
*/

type Element interface {
}

type Vector struct {
	nalloc int;
	nelem int;
	elem *[]Element;
}

// BUG: workaround for non-constant allocation.
// i must be a power of 10.
func Alloc(i int) *[]Element {
	switch i {
	case 1:
		return new([1]Element);
	case 10:
		return new([10]Element);
	case 100:
		return new([100]Element);
	case 1000:
		return new([1000]Element);
	}
	print "bad size ", i, "\n";
	panic "not known size\n";
}

func is_pow10(i int) bool {
	switch i {
	case 1, 10, 100, 1000:
		return true;
	}
	return false;
}

func New() *Vector {
	v := new(Vector);
	v.nelem = 0;
	v.nalloc = 1;
	v.elem = Alloc(v.nalloc);
	return v;
}

func (v *Vector) Len() int {
	return v.nelem;
}

func (v *Vector) At(i int) Element {
	if i < 0 || i >= v.nelem {
		//return nil;  // BUG
		panic "At out of range\n";
	}
	return v.elem[i];
}

func (v *Vector) Delete(i int) {
	if i < 0 || i >= v.nelem {
		panic "Delete out of range\n";
	}
	for j := i+1; j < v.nelem; j++ {
		v.elem[j-1] = v.elem[j];
	}
	v.nelem--;
	v.elem[v.nelem] = nil;
}

func (v *Vector) Insert(i int, e Element) {
	if i > v.nelem {
		panic "Del too large\n";
	}
	if v.nelem == v.nalloc && is_pow10(v.nalloc) {
		n := Alloc(v.nalloc * 10);
		for j := 0; j < v.nalloc; j++ {
			n[j] = v.elem[j];
		}
		v.elem = n;
		v.nalloc *= 10;
	}
	// make a hole
	for j := v.nelem; j > i; j-- {
		v.elem[j] = v.elem[j-1];
	}
	v.elem[i] = e;
	v.nelem++;
}

func (v *Vector) Append(e Element) {
	v.Insert(v.nelem, e);
}

/*
type I struct { val int; };  // BUG: can't be local;

func Test() {
	i0 := new(I); i0.val = 0;
	i1 := new(I); i1.val = 11;
	i2 := new(I); i2.val = 222;
	i3 := new(I); i3.val = 3333;
	i4 := new(I); i4.val = 44444;
	v := New();
	print "hi\n";
	v.Insert(0, i4);
	v.Insert(0, i3);
	v.Insert(0, i2);
	v.Insert(0, i1);
	v.Insert(0, i0);
	for i := 0; i < v.Len(); i++ {
		var x *I;
		x = v.At(i);
		print i, " ", x.val, "\n";  // BUG: can't use I(v.At(i))
	}
}

export Test;
*/
