// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import "testing"

func TestLogosAutovecVectorBackedgeIsPreemptible(t *testing.T) {
	autovecTestSettings(t)
	for _, arch := range []string{"amd64", "arm64"} {
		t.Run(arch, func(t *testing.T) {
			fun := autovecTestLoop(t, arch)
			f := fun.f
			f.ABIDefault = f.Config.ABI1
			autovec(f)
			CheckFunc(f)

			var vectorStore *Value
			for _, block := range f.Blocks {
				for _, v := range block.Values {
					if v.Op == OpStore && v.Args[1].Type.IsSIMD() {
						if vectorStore != nil {
							t.Fatal("multiple vector stores")
						}
						vectorStore = v
					}
				}
			}
			if vectorStore == nil {
				t.Fatal("vectorization did not produce a vector store")
			}
			body := vectorStore.Block
			header := body.Succs[0].b
			mem := vectorStore.Args[2]
			if mem.Op != OpPhi || mem.Block != header {
				t.Fatal("vector store does not use the vector loop memory phi")
			}

			insertLoopReschedChecks(f)
			CheckFunc(f)
			if vectorStore.Op != OpStore || !vectorStore.Args[1].Type.IsSIMD() {
				t.Fatal("vector store disappeared after reschedule insertion")
			}
			test := body.Succs[0].b
			if test.Kind != BlockIf || test.Controls[0].Op != OpLess64U || len(test.Succs) != 2 {
				t.Fatal("vector backedge does not pass through a reschedule check")
			}
			sched := test.Succs[0].b
			if test.Succs[1].b != header || sched.Kind != BlockPlain || sched.Succs[0].b != header {
				t.Fatal("vector reschedule check does not rejoin the vector header")
			}
			var call *Value
			for _, v := range sched.Values {
				if v.Op == OpStaticCall {
					call = v
				}
			}
			if call == nil || call.Aux.(*AuxCall).Fn.Name != "goschedguarded" || call.Args[0] != vectorStore {
				t.Fatal("vector backedge does not call goschedguarded with the stored memory")
			}
			fast := test.Succs[1].i
			slow := sched.Succs[0].i
			if mem.Args[fast] != vectorStore || mem.Args[slow].Op != OpSelectN || mem.Args[slow].Args[0] != call {
				t.Fatal("vector memory phi does not merge fast and rescheduled backedges")
			}
		})
	}
}
