// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import "testing"

func TestDeadLoop(t *testing.T) {
	c := NewConfig("amd64", DummyFrontend{t})
	fun := Fun(c, "entry",
		Bloc("entry",
			Valu("mem", OpArg, TypeMem, 0, ".mem"),
			Goto("exit")),
		Bloc("exit",
			Exit("mem")),
		// dead loop
		Bloc("deadblock",
			// dead value in dead block
			Valu("deadval", OpConst, TypeBool, 0, true),
			If("deadval", "deadblock", "exit")))

	CheckFunc(fun.f)
	Deadcode(fun.f)
	CheckFunc(fun.f)

	for _, b := range fun.f.Blocks {
		if b == fun.blocks["deadblock"] {
			t.Errorf("dead block not removed")
		}
		for _, v := range b.Values {
			if v == fun.values["deadval"] {
				t.Errorf("control value of dead block not removed")
			}
		}
	}
}

func TestDeadValue(t *testing.T) {
	c := NewConfig("amd64", DummyFrontend{t})
	fun := Fun(c, "entry",
		Bloc("entry",
			Valu("mem", OpArg, TypeMem, 0, ".mem"),
			Valu("deadval", OpConst, TypeInt64, 37, nil),
			Goto("exit")),
		Bloc("exit",
			Exit("mem")))

	CheckFunc(fun.f)
	Deadcode(fun.f)
	CheckFunc(fun.f)

	for _, b := range fun.f.Blocks {
		for _, v := range b.Values {
			if v == fun.values["deadval"] {
				t.Errorf("dead value not removed")
			}
		}
	}
}

func TestNeverTaken(t *testing.T) {
	c := NewConfig("amd64", DummyFrontend{t})
	fun := Fun(c, "entry",
		Bloc("entry",
			Valu("cond", OpConst, TypeBool, 0, false),
			Valu("mem", OpArg, TypeMem, 0, ".mem"),
			If("cond", "then", "else")),
		Bloc("then",
			Goto("exit")),
		Bloc("else",
			Goto("exit")),
		Bloc("exit",
			Exit("mem")))

	CheckFunc(fun.f)
	Opt(fun.f)
	Deadcode(fun.f)
	CheckFunc(fun.f)

	if fun.blocks["entry"].Kind != BlockPlain {
		t.Errorf("if(false) not simplified")
	}
	for _, b := range fun.f.Blocks {
		if b == fun.blocks["then"] {
			t.Errorf("then block still present")
		}
		for _, v := range b.Values {
			if v == fun.values["cond"] {
				t.Errorf("constant condition still present")
			}
		}
	}

}
