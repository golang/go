// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tabwriter

import (
	"os";
	"io";
	"tabwriter";
	"testing";
)


type Buffer struct {
	a *[]byte;
}


func (b *Buffer) Init(n int) {
	b.a = new([]byte, n)[0 : 0];
}


func (b *Buffer) Write(buf *[]byte) (written int, err *os.Error) {
	n := len(b.a);
	m := len(buf);
	if n + m <= cap(b.a) {
		b.a = b.a[0 : n + m];
		for i := 0; i < m; i++ {
			b.a[n+i] = buf[i];
		}
	} else {
		panicln("buffer too small", n, m, cap(b.a));
	}
	return len(buf), nil;
}


func (b *Buffer) String() string {
	return string(b.a);
}


func Check(t *testing.T, tabwidth, padding int, usetabs bool, src, expected string) {
	var b Buffer;
	b.Init(1000);

	var w tabwriter.Writer;
	w.Init(&b, tabwidth, padding, usetabs);

	io.WriteString(&w, src);

	res := b.String();
	if res != expected {
		t.Errorf("src:\n%s\nfound:\n%s\nexpected:\n%s\n", src, res, expected)
	}
}


export func Test1(t *testing.T) {
	Check(
		t, 8, 1, false,
		"\n",
		"\n"
	);

	Check(
		t, 8, 1, false,
		"Hello, world!\n",
		"Hello, world!\n"
	);

	Check(
		t, 8, 1, false,
		"a\tb\tc\naa\tbbb\tcccc\naaa\tbbbb\n\n",
		"a       b       c\n"
		"aa      bbb     cccc\n"
		"aaa     bbbb\n\n"
	);
}
