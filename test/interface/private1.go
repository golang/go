// true  # used by private.go

// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

type Exported interface {
	private()
}

type Implementation struct{}

func (p *Implementation) private() {}

var X = new(Implementation)

