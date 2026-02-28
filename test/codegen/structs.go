// asmcheck

//go:build !goexperiment.cgocheck2

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package codegen

// This file contains code generation tests related to the handling of
// struct types.

// ------------- //
//    Zeroing    //
// ------------- //

type Z1 struct {
	a, b, c int
}

func Zero1(t *Z1) { // Issue #18370
	// amd64:`MOVUPS X[0-9]+, \(.*\)`,`MOVQ \$0, 16\(.*\)`
	*t = Z1{}
}

type Z2 struct {
	a, b, c *int
}

func Zero2(t *Z2) {
	// amd64:`MOVUPS X[0-9]+, \(.*\)`,`MOVQ \$0, 16\(.*\)`
	// amd64:`.*runtime[.]gcWriteBarrier.*\(SB\)`
	*t = Z2{}
}

// ------------------ //
//    Initializing    //
// ------------------ //

type I1 struct {
	a, b, c, d int
}

func Init1(p *I1) { // Issue #18872
	// amd64:`MOVQ [$]1`,`MOVQ [$]2`,`MOVQ [$]3`,`MOVQ [$]4`
	*p = I1{1, 2, 3, 4}
}
