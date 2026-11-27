// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package wasm

import (
	"cmd/internal/obj"
	"testing"
)

// A testBlock is a basic block. It can call a function, jump to basic block
// cond if a condition holds, and then jump to basic block jmp, if not 0.
type testBlock struct {
	call bool
	cond int
	jmp  int
}

var loopTests = []struct {
	name   string
	blocks []testBlock
	loops  int // number of WebAssembly loops
}{
	{"simple", []testBlock{{}, {cond: 3}, {jmp: 1}, {}}, 1},
	{"self", []testBlock{{}, {cond: 1}, {}}, 1},
	{"adjacent", []testBlock{{}, {cond: 1}, {cond: 2}, {}}, 2},
	{"jumpToHeader", []testBlock{{cond: 2}, {}, {cond: 2}, {}}, 1},
	{"ifelse", []testBlock{{}, {cond: 6}, {cond: 4}, {jmp: 5}, {}, {jmp: 1}, {}}, 1},
	{"nested", []testBlock{{}, {cond: 5}, {cond: 4}, {jmp: 2}, {jmp: 1}, {}}, 2},
	{"nestedExits", []testBlock{{}, {cond: 6}, {cond: 5}, {cond: 1, jmp: 2}, {}, {jmp: 1}, {}}, 2},
	{"nestedSameEnd", []testBlock{{}, {}, {cond: 2, jmp: 1}, {}}, 2},
	{"call", []testBlock{{}, {cond: 3}, {call: true, jmp: 1}, {}}, 0},
	{"callOutside", []testBlock{{}, {cond: 5}, {cond: 4}, {jmp: 2}, {call: true, jmp: 1}, {}}, 1},
	{"sideEntry", []testBlock{{cond: 2}, {}, {jmp: 1}, {}}, 0},
	{"sideEntryFromBehind", []testBlock{{}, {cond: 4}, {}, {jmp: 1}, {cond: 2}, {}}, 0},
	{"crossing", []testBlock{{}, {}, {}, {cond: 1}, {cond: 2}, {}}, 1},
	{"irreducible", []testBlock{{}, {cond: 3}, {}, {cond: 2}, {jmp: 1}, {}}, 0},
	{"backwardExit", []testBlock{{}, {call: true}, {cond: 4}, {cond: 1, jmp: 2}, {}}, 1},
}

func TestStructureLoops(t *testing.T) {
	for _, test := range loopTests {
		t.Run(test.name, func(t *testing.T) {
			ctxt := obj.Linknew(&Linkwasm)
			s := ctxt.Lookup("test." + test.name)
			s.NewFuncInfo()
			alloc := func() *obj.Prog { return new(obj.Prog) }

			// Each basic block starts with an I32Const of its number.
			p := alloc()
			p.As = obj.ATEXT
			s.Func().Text = p
			add := func(as obj.As) *obj.Prog {
				p = obj.Appendp(p, alloc)
				p.As = as
				return p
			}
			firsts := make([]*obj.Prog, len(test.blocks))
			want := make(map[*obj.Prog]int) // jump -> basic block
			jump := func(v int) {
				want[add(obj.AJMP)] = v
			}
			for i, b := range test.blocks {
				firsts[i] = add(AI32Const)
				firsts[i].From = constAddr(int64(i))
				add(ADrop)
				if b.call {
					add(obj.ACALL)
				}
				if b.cond != 0 {
					add(AI32Const)
					add(AIf)
					jump(b.cond)
					add(AEnd)
				}
				if b.jmp != 0 {
					jump(b.jmp)
				}
				add(ARESUMEPOINT)
			}
			for p, v := range want {
				p.To = obj.Addr{Type: obj.TYPE_BRANCH, Val: firsts[v]}
			}

			structureLoops(s, alloc)

			// Find the End of each block and the depth of each instruction.
			ends := make(map[*obj.Prog]*obj.Prog)
			depths := make(map[*obj.Prog]int)
			var open []*obj.Prog
			loops := 0
			for p := s.Func().Text; p != nil; p = p.Link {
				depths[p] = len(open)
				switch p.As {
				case ALoop:
					loops++
					open = append(open, p)
				case ABlock, AIf:
					open = append(open, p)
				case AEnd:
					if len(open) == 0 {
						t.Fatal("End without block")
					}
					ends[open[len(open)-1]] = p
					open = open[:len(open)-1]
				case ARESUMEPOINT, obj.ACALL:
					if len(open) != 0 {
						t.Errorf("%v inside a block", p.As)
					}
				}
			}
			if len(open) != 0 {
				t.Fatalf("%d blocks without End", len(open))
			}
			if loops != test.loops {
				t.Errorf("got %d loops, want %d", loops, test.loops)
			}

			// lands returns the basic block that execution continues with at p.
			lands := func(p *obj.Prog) int {
				for p.As == ALoop || p.As == ABlock {
					p = p.Link
				}
				if p.As != AI32Const {
					t.Fatalf("branch to %v, not to a basic block", p.As)
				}
				return int(p.From.Offset)
			}
			open = open[:0]
			for p := s.Func().Text; p != nil; p = p.Link {
				switch p.As {
				case ALoop, ABlock, AIf:
					open = append(open, p)
				case AEnd:
					open = open[:len(open)-1]
				case ABr:
					label := p.To.Val.(*obj.Prog)
					if d := depths[label]; d >= len(open) || open[d] != label {
						t.Fatal("branch to a block that is not open")
					}
					next := label.Link
					if label.As != ALoop {
						next = ends[label].Link
					}
					if got := lands(next); got != want[p] {
						t.Errorf("branch goes to basic block %d, want %d", got, want[p])
					}
				case obj.AJMP:
					// entryPointLoop can not branch into a loop.
					target := p.To.Val.(*obj.Prog)
					if depths[target] != 0 {
						t.Error("jump into a loop")
					}
					if got := lands(target); got != want[p] {
						t.Errorf("jump goes to basic block %d, want %d", got, want[p])
					}
				}
			}
		})
	}
}
