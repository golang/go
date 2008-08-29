// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package vector

//export Vector, New;

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


export type Vector struct {
	elem *[]Element;
}


export func New() *Vector {
	v := new(Vector);
	v.elem = new([]Element, 1) [0 : 0];  // capacity must be > 0!
	return v;
}


func (v *Vector) RangeError(op string, i int) {
	panic("Vector.", op, ": index ", i, " out of range (len = ", len(v.elem), ")\n");
}


func (v *Vector) Len() int {
	return len(v.elem);
}


func (v *Vector) At(i int) Element {
	n := v.Len();
	if i < 0 || i >= n {
		v.RangeError("At", i);
		var e Element;
		return e;  // don't return nil - may not be legal in the future
	}
	return v.elem[i];
}


// TODO(r) It would be better if this were called 'Remove' and if
// it were returning the removed element. This way it would be 
// symmetric with 'Insert', provide the functionality of 'Delete'
// and allow to get the appropriate entry w/ an extra call.

func (v *Vector) Delete(i int) {
	n := v.Len();
	if i < 0 || i >= n {
		v.RangeError("Delete", i);
	}
	for j := i + 1; j < n; j++ {
		v.elem[j - 1] = v.elem[j];
	}
	var e Element;
	v.elem[n - 1] = e;  // don't set to nil - may not be legal in the future
	v.elem = v.elem[0 : n - 1];
}


func (v *Vector) Insert(i int, e Element) {
	n := v.Len();
	if i < 0 || i > n {
		v.RangeError("Insert", i);
	}

	// grow array by doubling its capacity
	if n == cap(v.elem) {
		a := new([]Element, n*2);
		for j := 0; j < n; j++ {
			a[j] = v.elem[j];
		}
		v.elem = a;
	}

	// make a hole
	v.elem = v.elem[0 : n + 1];
	for j := n; j > i; j-- {
		v.elem[j] = v.elem[j-1];
	}
	
	v.elem[i] = e;
}


func (v *Vector) Append(e Element) {
	v.Insert(len(v.elem), e);
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
	print("hi\n");
	v.Insert(0, i4);
	v.Insert(0, i3);
	v.Insert(0, i2);
	v.Insert(0, i1);
	v.Insert(0, i0);
	for i := 0; i < v.Len(); i++ {
		x := convert(*I, v.At(i));
		print(i, " ", v.At(i).(*I).val, "\n");
	}
}

export Test;
*/
