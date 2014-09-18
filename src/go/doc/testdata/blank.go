// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package blank is a go/doc test for the handling of _.
// See issue 5397.
package blank

type T int

// T constants.
const (
	_ T = iota
	T1
	T2
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
