// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package array

export type Element interface {
}


export type Array struct {
	// TODO do not export field
	a []Element
}


func (p *Array) Init(initial_len int) *Array {
	a := p.a;

	if cap(a) == 0 || cap(a) < initial_len {
		n := 8;  // initial capacity
		if initial_len > n {
			n = initial_len
		}
		a = new([]Element, n);
	} else {
		// nil out entries
		for j := len(a) - 1; j >= 0; j-- {
			a[j] = nil
		}
	}

	p.a = a[0 : initial_len];
	return p
}


export func New(len int) *Array {
	return new(*Array).Init(len)
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
	a := p.a;
	n := len(a);

	// grow array by doubling its capacity
	if n == cap(a) {
		b := new([]Element, 2*n);
		for j := n-1; j >= 0; j-- {
			b[j] = a[j];
		}
		a = b
	}

	// make a hole
	a = a[0 : n+1];
	for j := n; j > i; j-- {
		a[j] = a[j-1]
	}

	a[i] = x;
	p.a = a
}


func (p *Array) Remove(i int) Element {
	a := p.a;
	n := len(a);

	x := a[i];
	for j := i+1; j < n; j++ {
		a[j-1] = a[j]
	}

	a[n-1] = nil;  // support GC, nil out entry
	p.a = a[0 : n-1];

	return x
}


func (p *Array) Push(x Element) {
	p.Insert(len(p.a), x)
}


func (p *Array) Pop() Element {
	return p.Remove(len(p.a) - 1)
}


// Partial SortInterface support

export type LessInterface interface {
	Less(y Element) bool
}


func (p *Array) Less(i, j int) bool {
	return p.a[i].(LessInterface).Less(p.a[j])
}


func (p *Array) Swap(i, j int) {
	a := p.a;
	a[i], a[j] = a[j], a[i]
}
