// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Imported by private.go, which should not be able to see the private method.

package p

type Exported interface {
	private()
}

type Implementation struct{}

func (p *Implementation) private() {}

var X = new(Implementation)

