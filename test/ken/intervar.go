// $G $D/$F.go && $L $F.$A && ./$A.out

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

type	Iputs	interface {
	puts	(s string);
}

// ---------

type	Print	struct {
	whoami	int;
	put	Iputs;
}

func (p *Print) dop() {
	print(" print ", p.whoami);
	p.put.puts("abc");
}

// ---------

type	Bio	struct {
	whoami	int;
	put	Iputs;
}

func (b *Bio) puts(s string) {
	print(" bio ", b.whoami);
	b.put.puts(s);
}

// ---------

type	File	struct {
	whoami	int;
	put	Iputs;
}

func (f *File) puts(s string) {
	print(" file ", f.whoami, " -- ", s);
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

	p.dop();
	print("\n");
}
