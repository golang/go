// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import "testing"

func TestSchedule(t *testing.T) {
	c := NewConfig("amd64", DummyFrontend{})
	cases := []fun{
		Fun(c, "entry",
			Bloc("entry",
				Valu("mem0", OpArg, TypeMem, ".mem"),
				Valu("ptr", OpConst, TypeInt64, 0xABCD),
				Valu("v", OpConst, TypeInt64, 12),
				Valu("mem1", OpStore, TypeMem, 32, "ptr", "v", "mem0"),
				Valu("mem2", OpStore, TypeMem, 32, "ptr", "v", "mem1"),
				Valu("mem3", OpStore, TypeInt64, "ptr", "sum", "mem2"),
				Valu("l1", OpLoad, TypeInt64, 16, "ptr", "mem1"),
				Valu("l2", OpLoad, TypeInt64, 8, "ptr", "mem2"),
				Valu("sum", OpAdd, TypeInt64, "l1", "l2"),
				Goto("exit")),
			Bloc("exit",
				Exit("mem3"))),
	}
	for _, c := range cases {
		schedule(c.f)
		if !isSingleLiveMem(c.f) {
			t.Error("single-live-mem restriction not enforced by schedule for func:")
			printFunc(c.f)
		}
	}
}

func isSingleLiveMem(f *Func) bool {
	for _, b := range f.Blocks {
		var liveMem *Value
		for _, v := range b.Values {
			for _, w := range v.Args {
				if w.Type.IsMemory() {
					if liveMem == nil {
						liveMem = w
						continue
					}
					if w != liveMem {
						return false
					}
				}
			}
			if v.Type.IsMemory() {
				liveMem = v
			}
		}
	}
	return true
}
