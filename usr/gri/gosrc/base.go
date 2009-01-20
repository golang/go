// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Base for the decls.go tests.

package base

type Foo int

type Bar *float;

type Node struct {
  left, right *Node;
  val bool;
  f Foo;
}

func (p *Node) F(x int) {};

type I interface {
  f();
}
