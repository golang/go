// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package blank is a go/doc test for the handling of _.
// See issue 5397.
package blank

import "os"

type T int

// T constants counting from a blank constant.
const (
	_ T = iota
	T1
	T2
)

// T constants counting from unexported constants.
const (
	tweedledee T = iota
	tweedledum
	C1
	C2
	alice
	C3
	redQueen int = iota
	C4
)

// Constants with an imported type that needs to be propagated.
const (
	zero     os.FileMode = 0
	Default              = 0644
	Useless              = 0312
	WideOpen             = 0777
)

// Package constants.
const (
	_ int = iota
	I1
	I2
)

// Blanks not in doc output:

// S has a padding field.
type S struct {
	H uint32
	_ uint8
	A uint8
}

func _() {}

type _ T

var _ = T(55)
