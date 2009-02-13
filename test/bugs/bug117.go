// errchk $G $D/$F.go

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main
type S struct { a int }
type PS *S
func (p *S) get() int {
  return p.a
}
func fn(p PS) int {
  return p.get()
}
func main() {
  s := S(1);
  if s.get() != 1 {
    panic()
  }
}
