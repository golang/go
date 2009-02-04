// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// ! $G $D/$F.go || echo BUG: compilation succeeds incorrectly

package main

type t struct
type s struct {
  p *t  // BUG t never declared
}

func main() {
  var s1 s;
}
