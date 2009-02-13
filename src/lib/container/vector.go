// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package vector

type (
	Element interface {};
	Vector struct {
		a []Element
	}
)


func copy(dst, src []Element) {
	for i := 0; i < len(src); i++ {
		dst[i] = src[i]
	}
}


// Insert n elements at position i.
func expand(a []Element, i, n int) []Element {
	// make sure we have enough space
	len0 := len(a);
	len1 := len0 + n;
	if len1 < cap(a) {
		// enough space - just expand
		a = a[0 : len1]
	} else {
		// not enough space - double capacity
		capb := cap(a)*2;
		if capb < len1 {
			// still not enough - use required length
			capb = len1
		}
		// capb >= len1
		b := make([]Element, len1, capb);
		copy(b, a);
		a = b
	}

	// make a hole
	for j := len0-1; j >= i ; j-- {
		a[j+n] = a[j]
	}
	return a
}


func (p *Vector) Init(initial_len int) *Vector {
	a := p.a;

	if cap(a) == 0 || cap(a) < initial_len {
		n := 8;  // initial capacity
		if initial_len > n {
			n = initial_len
		}
		a = make([]Element, n);
	} else {
		// nil out entries
		for j := len(a) - 1; j >= 0; j-- {
			a[j] = nil
		}
	}

	p.a = a[0 : initial_len];
	return p
}


func New(len int) *Vector {
	return new(Vector).Init(len)
}


func (p *Vector) Len() int {
	return len(p.a)
}


func (p *Vector) At(i int) Element {
	return p.a[i]
}


func (p *Vector) Set(i int, x Element) {
	p.a[i] = x
}


func (p *Vector) Last() Element {
	return p.a[len(p.a) - 1]
}


func (p *Vector) Insert(i int, x Element) {
	p.a = expand(p.a, i, 1);
	p.a[i] = x;
}


func (p *Vector) Delete(i int) Element {
	a := p.a;
	n := len(a);

	x := a[i];
	copy(a[i : n-1], a[i+1 : n]);
	a[n-1] = nil;  // support GC, nil out entry
	p.a = a[0 : n-1];

	return x
}


func (p *Vector) InsertVector(i int, x *Vector) {
	p.a = expand(p.a, i, len(x.a));
	copy(p.a[i : i + len(x.a)], x.a);
}


func (p *Vector) Cut(i, j int) {
	a := p.a;
	n := len(a);
	m := n - (j - i);

	copy(a[i : m], a[j : n]);
	for k := m; k < n; k++ {
		a[k] = nil  // support GC, nil out entries
	}

	p.a = a[0 : m];
}


func (p *Vector) Slice(i, j int) *Vector {
	s := New(j - i);  // will fail in Init() if j < j
	copy(s.a, p.a[i : j]);
	return s;
}


func (p *Vector) Do(f func(elem Element)) {
	for i := 0; i < len(p.a); i++ {
		f(p.a[i])	// not too safe if f changes the Vector
	}
}


// Convenience wrappers

func (p *Vector) Push(x Element) {
	p.Insert(len(p.a), x)
}


func (p *Vector) Pop() Element {
	return p.Delete(len(p.a) - 1)
}


func (p *Vector) AppendVector(x *Vector) {
	p.InsertVector(len(p.a), x);
}


// Partial SortInterface support

type LessInterface interface {
	Less(y Element) bool
}


func (p *Vector) Less(i, j int) bool {
	return p.a[i].(LessInterface).Less(p.a[j])
}


func (p *Vector) Swap(i, j int) {
	a := p.a;
	a[i], a[j] = a[j], a[i]
}
