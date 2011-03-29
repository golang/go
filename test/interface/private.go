// $G $D/${F}1.go && errchk $G $D/$F.go

// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "./private1"

type Exported interface {
	private()
}

type Implementation struct{}

func (p *Implementation) private() {}

func main() {
	var x Exported
	x = new(Implementation)
	x.private()

	var px p.Exported
	px = p.X

	px.private()			// ERROR "private"

	px = new(Implementation)	// ERROR "private"

	x = px				// ERROR "private"
}
