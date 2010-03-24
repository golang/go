// $G $D/$F.go && $L $F.$A && ./$A.out

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "fmt"

type Buffer int

func (*Buffer) Read() {}

type Reader interface {
	Read()
}

func f() *Buffer { return nil }

func g() Reader {
	// implicit interface conversion in assignment during return
	return f()
}

func h() (b *Buffer, ok bool) { return }

func i() (r Reader, ok bool) {
	// implicit interface conversion in multi-assignment during return
	return h()
}

func fmter() (s string, i int, t string) { return "%#x %q", 100, "hello" }

func main() {
	b := g()
	bb, ok := b.(*Buffer)
	_, _, _ = b, bb, ok

	b, ok = i()
	bb, ok = b.(*Buffer)
	_, _, _ = b, bb, ok

	s := fmt.Sprintf(fmter())
	if s != "0x64 \"hello\"" {
		println(s)
		panic("fail")
	}
}
