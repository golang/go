// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import (
	"cmd/compile/internal/types"
	"testing"
)

// Test that a trivial 'if' is eliminated
func TestBranchElimIf(t *testing.T) {
	var testData = []struct {
		arch    string
		intType string
		ok      bool
	}{
		{"arm64", "int32", true},
		{"amd64", "int32", true},
		{"amd64", "int8", false},
	}

	for _, data := range testData {
		t.Run(data.arch+"/"+data.intType, func(t *testing.T) {
			c := testConfigArch(t, data.arch)
			boolType := c.config.Types.Bool
			var intType *types.Type
			switch data.intType {
			case "int32":
				intType = c.config.Types.Int32
			case "int8":
				intType = c.config.Types.Int8
			default:
				t.Fatal("invalid integer type:", data.intType)
			}
			fun := c.Fun("entry",
				Bloc("entry",
					Valu("start", OpInitMem, types.TypeMem, 0, nil),
					Valu("sb", OpSB, c.config.Types.Uintptr, 0, nil),
					Valu("const1", OpConst32, intType, 1, nil),
					Valu("const2", OpConst32, intType, 2, nil),
					Valu("addr", OpAddr, boolType.PtrTo(), 0, nil, "sb"),
					Valu("cond", OpLoad, boolType, 0, nil, "addr", "start"),
					If("cond", "b2", "b3")),
				Bloc("b2",
					Goto("b3")),
				Bloc("b3",
					Valu("phi", OpPhi, intType, 0, nil, "const1", "const2"),
					Valu("retstore", OpStore, types.TypeMem, 0, nil, "phi", "sb", "start"),
					Exit("retstore")))

			CheckFunc(fun.f)
			branchelim(fun.f)
			CheckFunc(fun.f)
			Deadcode(fun.f)
			CheckFunc(fun.f)

			if data.ok {

				if len(fun.f.Blocks) != 1 {
					t.Fatalf("expected 1 block after branchelim and deadcode; found %d", len(fun.f.Blocks))
				}
				if fun.values["phi"].Op != OpCondSelect {
					t.Fatalf("expected phi op to be CondSelect; found op %s", fun.values["phi"].Op)
				}
				if fun.values["phi"].Args[2] != fun.values["cond"] {
					t.Errorf("expected CondSelect condition to be %s; found %s", fun.values["cond"], fun.values["phi"].Args[2])
				}
				if fun.blocks["entry"].Kind != BlockExit {
					t.Errorf("expected entry to be BlockExit; found kind %s", fun.blocks["entry"].Kind.String())
				}
			} else {
				if len(fun.f.Blocks) != 3 {
					t.Fatalf("expected 3 block after branchelim and deadcode; found %d", len(fun.f.Blocks))
				}
			}
		})
	}
}

// Test that a trivial if/else is eliminated
func TestBranchElimIfElse(t *testing.T) {
	for _, arch := range []string{"arm64", "amd64"} {
		t.Run(arch, func(t *testing.T) {
			c := testConfigArch(t, arch)
			boolType := c.config.Types.Bool
			intType := c.config.Types.Int32
			fun := c.Fun("entry",
				Bloc("entry",
					Valu("start", OpInitMem, types.TypeMem, 0, nil),
					Valu("sb", OpSB, c.config.Types.Uintptr, 0, nil),
					Valu("const1", OpConst32, intType, 1, nil),
					Valu("const2", OpConst32, intType, 2, nil),
					Valu("addr", OpAddr, boolType.PtrTo(), 0, nil, "sb"),
					Valu("cond", OpLoad, boolType, 0, nil, "addr", "start"),
					If("cond", "b2", "b3")),
				Bloc("b2",
					Goto("b4")),
				Bloc("b3",
					Goto("b4")),
				Bloc("b4",
					Valu("phi", OpPhi, intType, 0, nil, "const1", "const2"),
					Valu("retstore", OpStore, types.TypeMem, 0, nil, "phi", "sb", "start"),
					Exit("retstore")))

			CheckFunc(fun.f)
			branchelim(fun.f)
			CheckFunc(fun.f)
			Deadcode(fun.f)
			CheckFunc(fun.f)

			if len(fun.f.Blocks) != 1 {
				t.Fatalf("expected 1 block after branchelim; found %d", len(fun.f.Blocks))
			}
			if fun.values["phi"].Op != OpCondSelect {
				t.Fatalf("expected phi op to be CondSelect; found op %s", fun.values["phi"].Op)
			}
			if fun.values["phi"].Args[2] != fun.values["cond"] {
				t.Errorf("expected CondSelect condition to be %s; found %s", fun.values["cond"], fun.values["phi"].Args[2])
			}
			if fun.blocks["entry"].Kind != BlockExit {
				t.Errorf("expected entry to be BlockExit; found kind %s", fun.blocks["entry"].Kind.String())
			}
		})
	}
}

// Test that an if/else CFG that loops back
// into itself does *not* get eliminated.
func TestNoBranchElimLoop(t *testing.T) {
	for _, arch := range []string{"arm64", "amd64"} {
		t.Run(arch, func(t *testing.T) {
			c := testConfigArch(t, arch)
			boolType := c.config.Types.Bool
			intType := c.config.Types.Int32

			// The control flow here is totally bogus,
			// but a dead cycle seems like the only plausible
			// way to arrive at a diamond CFG that is also a loop.
			fun := c.Fun("entry",
				Bloc("entry",
					Valu("start", OpInitMem, types.TypeMem, 0, nil),
					Valu("sb", OpSB, c.config.Types.Uintptr, 0, nil),
					Valu("const2", OpConst32, intType, 2, nil),
					Valu("const3", OpConst32, intType, 3, nil),
					Goto("b5")),
				Bloc("b2",
					Valu("addr", OpAddr, boolType.PtrTo(), 0, nil, "sb"),
					Valu("cond", OpLoad, boolType, 0, nil, "addr", "start"),
					Valu("phi", OpPhi, intType, 0, nil, "const2", "const3"),
					If("cond", "b3", "b4")),
				Bloc("b3",
					Goto("b2")),
				Bloc("b4",
					Goto("b2")),
				Bloc("b5",
					Exit("start")))

			CheckFunc(fun.f)
			branchelim(fun.f)
			CheckFunc(fun.f)

			if len(fun.f.Blocks) != 5 {
				t.Errorf("expected 5 block after branchelim; found %d", len(fun.f.Blocks))
			}
			if fun.values["phi"].Op != OpPhi {
				t.Errorf("expected phi op to be CondSelect; found op %s", fun.values["phi"].Op)
			}
		})
	}
}
