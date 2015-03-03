// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// TODO: these tests are pretty verbose.  Is there a way to simplify
// building a small Func for testing?

package ssa_test

import (
	. "cmd/internal/ssa"
	"testing"
)

func TestDeadLoop(t *testing.T) {
	f := new(Func)
	entry := f.NewBlock(BlockPlain)
	exit := f.NewBlock(BlockExit)
	f.Entry = entry
	addEdge(entry, exit)
	mem := entry.NewValue(OpArg, TypeMem, ".mem")
	exit.Control = mem

	// dead loop
	deadblock := f.NewBlock(BlockIf)
	addEdge(deadblock, deadblock)
	addEdge(deadblock, exit)

	// dead value in dead block
	deadval := deadblock.NewValue(OpConstBool, TypeBool, true)
	deadblock.Control = deadval

	CheckFunc(f)
	Deadcode(f)
	CheckFunc(f)

	for _, b := range f.Blocks {
		if b == deadblock {
			t.Errorf("dead block not removed")
		}
		for _, v := range b.Values {
			if v == deadval {
				t.Errorf("control value of dead block not removed")
			}
		}
	}
}

func TestDeadValue(t *testing.T) {
	f := new(Func)
	entry := f.NewBlock(BlockPlain)
	exit := f.NewBlock(BlockExit)
	f.Entry = entry
	addEdge(entry, exit)
	mem := entry.NewValue(OpArg, TypeMem, ".mem")
	exit.Control = mem

	deadval := entry.NewValue(OpConstInt, TypeInt, 37)

	CheckFunc(f)
	Deadcode(f)
	CheckFunc(f)

	for _, b := range f.Blocks {
		for _, v := range b.Values {
			if v == deadval {
				t.Errorf("dead value not removed")
			}
		}
	}
}

func TestNeverTaken(t *testing.T) {
	f := new(Func)
	entry := f.NewBlock(BlockIf)
	exit := f.NewBlock(BlockExit)
	then := f.NewBlock(BlockPlain)
	else_ := f.NewBlock(BlockPlain)
	f.Entry = entry
	addEdge(entry, then)
	addEdge(entry, else_)
	addEdge(then, exit)
	addEdge(else_, exit)
	mem := entry.NewValue(OpArg, TypeMem, ".mem")
	exit.Control = mem

	cond := entry.NewValue(OpConstBool, TypeBool, false)
	entry.Control = cond

	CheckFunc(f)
	Deadcode(f)
	CheckFunc(f)

	if entry.Kind != BlockPlain {
		t.Errorf("if(false) not simplified")
	}
	for _, b := range f.Blocks {
		if b == then {
			t.Errorf("then block still present")
		}
		for _, v := range b.Values {
			if v == cond {
				t.Errorf("constant condition still present")
			}
		}
	}
}

func addEdge(b, c *Block) {
	b.Succs = append(b.Succs, c)
	c.Preds = append(c.Preds, b)
}
