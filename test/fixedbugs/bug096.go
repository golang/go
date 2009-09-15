// $G $D/$F.go || echo BUG should compile

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

type A []int;

func main() {
	a := &A{0};
	b := &A{0, 1};
	_, _ = a, b;
}

/*
uetli:~/Source/go1/test/bugs gri$ 6g bug096.go && 6l bug096.6 && 6.out
Trace/BPT trap
uetli:~/Source/go1/test/bugs gri$
*/

/*
It appears that the first assignment changes the size of A from open
into a fixed array.
*/
