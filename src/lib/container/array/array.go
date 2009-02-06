// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package array

type (
	Element interface {};
	Array struct {
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


func (p *Array) Init(initial_len int) *Array {
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


func New(len int) *Array {
	return new(Array).Init(len)
}


func (p *Array) Len() int {
	return len(p.a)
}


func (p *Array) At(i int) Element {
	return p.a[i]
}


func (p *Array) Set(i int, x Element) {
	p.a[i] = x
}


func (p *Array) Last() Element {
	return p.a[len(p.a) - 1]
}


func (p *Array) Insert(i int, x Element) {
	p.a = expand(p.a, i, 1);
	p.a[i] = x;
}


func (p *Array) Delete(i int) Element {
	a := p.a;
	n := len(a);

	x := a[i];
	copy(a[i : n-1], a[i+1 : n]);
	a[n-1] = nil;  // support GC, nil out entry
	p.a = a[0 : n-1];

	return x
}


func (p *Array) InsertArray(i int, x *Array) {
	p.a = expand(p.a, i, len(x.a));
	copy(p.a[i : i + len(x.a)], x.a);
}


func (p *Array) Cut(i, j int) {
	a := p.a;
	n := len(a);
	m := n - (j - i);

	copy(a[i : m], a[j : n]);
	for k := m; k < n; k++ {
		a[k] = nil  // support GC, nil out entries
	}

	p.a = a[0 : m];
}


func (p *Array) Slice(i, j int) *Array {
	s := New(j - i);  // will fail in Init() if j < j
	copy(s.a, p.a[i : j]);
	return s;
}


func (p *Array) Do(f func(elem Element)) {
	for i := 0; i < len(p.a); i++ {
		f(p.a[i])	// not too safe if f changes the Array
	}
}


// Convenience wrappers

func (p *Array) Push(x Element) {
	p.Insert(len(p.a), x)
}


func (p *Array) Pop() Element {
	return p.Delete(len(p.a) - 1)
}


func (p *Array) AppendArray(x *Array) {
	p.InsertArray(len(p.a), x);
}


// Partial SortInterface support

type LessInterface interface {
	Less(y Element) bool
}


func (p *Array) Less(i, j int) bool {
	return p.a[i].(LessInterface).Less(p.a[j])
}


func (p *Array) Swap(i, j int) {
	a := p.a;
	a[i], a[j] = a[j], a[i]
}
