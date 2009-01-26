// $G $D/$F.go && $L $F.$A && ./$A.out

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

const	size	= 16;

var	a	[size]byte;
var	p	[]byte;
var	m	map[int]byte;

func
f(k int) byte
{
	return byte(k*10007 % size);
}

func
init()
{
	p = make([]byte, size);
	m = make(map[int]byte);
	for k:=0; k<size; k++ {
		v := f(k);
		a[k] = v;
		p[k] = v;
		m[k] = v;
	}
}

func
main()
{
	var i int;

	/*
	 * key only
	 */
	i = 0;
	for k := range a {
		v := a[k];
		if v != f(k) {
			panicln("key array range", k, v, a[k]);
		}
		i++;
	}
	if i != size {
		panicln("key array size", i);
	}

	i = 0;
	for k := range p {
		v := p[k];
		if v != f(k) {
			panicln("key pointer range", k, v, p[k]);
		}
		i++;
	}
	if i != size {
		panicln("key pointer size", i);
	}

	i = 0;
	for k := range m {
		v := m[k];
		if v != f(k) {
			panicln("key map range", k, v, m[k]);
		}
		i++;
	}
	if i != size {
		panicln("key map size", i);
	}

	/*
	 * key,value
	 */
	i = 0;
	for k,v := range a {
		if v != f(k) {
			panicln("key:value array range", k, v, a[k]);
		}
		i++;
	}
	if i != size {
		panicln("key:value array size", i);
	}

	i = 0;
	for k,v := range p {
		if v != f(k) {
			panicln("key:value pointer range", k, v, p[k]);
		}
		i++;
	}
	if i != size {
		panicln("key:value pointer size", i);
	}

	i = 0;
	for k,v := range m {
		if v != f(k) {
			panicln("key:value map range", k, v, m[k]);
		}
		i++;
	}
	if i != size {
		panicln("key:value map size", i);
	}
}
