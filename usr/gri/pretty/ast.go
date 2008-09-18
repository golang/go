// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package AST;


export type Expr interface {
       pos() int;
       print();
}


export type Stat interface {
       pos() int;
       print();
}


// ---------------------------------------------------------------------
// Concrete nodes

export type Ident struct {
       pos_ int;
       val_ string;
}


func (p *Ident) pos() int {
     return p.pos_;
}


func (p *Ident) print() {
     print("x");  // TODO fix this
}


// TODO: complete this
