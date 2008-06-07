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
