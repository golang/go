// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// $G $D/$F.go || echo BUG: compilation should succeed

package main

type T  // BUG forward declaration should be accepted
type S struct {
  p *T
}

type T struct {
  p *S
}

func main() {
  var s S;
}

/*
Per discussion w/ Ken, some time ago, we came to the conclusion that explicit
forward declarations (as on line 5 in this program) are preferrable over
implicit forward declarations because they make it explicit in which scope a
type is to be declared fully, eventually. As an aside, the machinery for it is
almost free in the compiler (one extra 'if' as far as I can tell).
*/
