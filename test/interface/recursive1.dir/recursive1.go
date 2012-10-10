// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Mutually recursive type definitions imported and used by recursive1.go.

package p

type I1 interface {
	F() I2
}

type I2 interface {
	I1
}
