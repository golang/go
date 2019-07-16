// errorcheck

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test that interface{M()} = *interface{M()} produces a compiler error.
// Does not compile.

package main

type Inst interface {
	Next() *Inst
}

type Regexp struct {
	code  []Inst
	start Inst
}

type Start struct {
	foo *Inst
}

func (start *Start) Next() *Inst { return nil }


func AddInst(Inst) *Inst {
	print("ok in addinst\n")
	return nil
}

func main() {
	print("call addinst\n")
	var x Inst = AddInst(new(Start)) // ERROR "pointer to interface"
	print("return from  addinst\n")
	var y *Inst = new(Start)  // ERROR "pointer to interface|incompatible type"
}
