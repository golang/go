// run

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test interface assignment.

package main

type	Iputs	interface {
	puts	(s string) string;
}

// ---------

type	Print	struct {
	whoami	int;
	put	Iputs;
}

func (p *Print) dop() string {
	r := " print " + string(p.whoami + '0')
	return r + p.put.puts("abc");
}

// ---------

type	Bio	struct {
	whoami	int;
	put	Iputs;
}

func (b *Bio) puts(s string) string {
	r := " bio " + string(b.whoami + '0')
	return r + b.put.puts(s);
}

// ---------

type	File	struct {
	whoami	int;
	put	Iputs;
}

func (f *File) puts(s string) string {
	return " file " + string(f.whoami + '0') + " -- " + s
}

func
main() {
	p := new(Print);
	b := new(Bio);
	f := new(File);

	p.whoami = 1;
	p.put = b;

	b.whoami = 2;
	b.put = f;

	f.whoami = 3;

	r := p.dop();
	expected := " print 1 bio 2 file 3 -- abc"
	if r != expected {
		panic(r + " != " + expected)
	}
}
