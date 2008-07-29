// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// $G $D/$F.go && echo BUG: compilation succeeds incorrectly

package main

type S struct {
  p *T  // BUG T never declared
}

func main() {
  var s S;
}

/*
Another problem with implicit forward declarations (as in this program on line 6)
is that it is not clear in which scope the type (here "T") should be declared.
This is the main reason why we should not allow implicit forward declarations at all,
and instead have an explicit type forward declaration. For more on this subject
see bug042.go.
*/
