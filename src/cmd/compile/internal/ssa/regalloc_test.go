// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import "testing"

func TestLiveControlOps(t *testing.T) {
	c := testConfig(t)
	f := Fun(c, "entry",
		Bloc("entry",
			Valu("mem", OpInitMem, TypeMem, 0, ".mem"),
			Valu("x", OpAMD64MOVBconst, TypeInt8, 0, 1),
			Valu("y", OpAMD64MOVBconst, TypeInt8, 0, 2),
			Valu("a", OpAMD64TESTB, TypeFlags, 0, nil, "x", "y"),
			Valu("b", OpAMD64TESTB, TypeFlags, 0, nil, "y", "x"),
			Eq("a", "if", "exit"),
		),
		Bloc("if",
			Eq("b", "plain", "exit"),
		),
		Bloc("plain",
			Goto("exit"),
		),
		Bloc("exit",
			Exit("mem"),
		),
	)
	flagalloc(f.f)
	regalloc(f.f)
	checkFunc(f.f)
}
