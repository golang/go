// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"./p"
)

type Exported interface {
	private()
}

type Implementation struct{}

func (p *Implementation) private() {}


func main() {
	// nothing unusual here
	var x Exported
	x = new(Implementation)
	x.private()  //  main.Implementation.private()

	// same here - should be and is legal
	var px p.Exported
	px = p.X
	
	// this assignment is correctly illegal:
	//	px.private undefined (cannot refer to unexported field or method private)
	// px.private()

	// this assignment is correctly illegal:
	//	*Implementation does not implement p.Exported (missing p.private method)
	// px = new(Implementation)

	// this assignment is correctly illegal:
	//	p.Exported does not implement Exported (missing private method)
	// x = px

	// this assignment unexpectedly compiles and then executes
	defer func() {
		recover()
	}()
	x = px.(Exported)
	
	println("should not get this far")

	// this is a legitimate call, but because of the previous assignment,
	// it invokes the method private in p!
	x.private()  // p.Implementation.private()
}
