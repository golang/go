// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import (
	"cmd/internal/src"
	"math/bits"
	"testing"
)

func TestStackAllocLiveSetEquations(t *testing.T) {
	for _, name := range []string{
		"differingPhiDiamond", "duplicateEdges", "selfLoop", "loopCarriedPhi",
		"forcedSpillAtDefinition", "ordinarySameBlock", "phiSpillRegisterSource",
		"disconnectedCyclesAndIsolated", "idGapsAndEmptyBlocks", "reusedMergePredecessor",
		"requeueAfterPopWithDistinctBits",
		"nonIDBlockOrder",
		"denseSlots63", "denseSlots64", "denseSlots65",
		"nonadjacentDenseWords",
	} {
		t.Run(name, func(t *testing.T) {
			c := testConfig(t)
			f := c.config.NewFunc(c.Frontend(), new(Cache))
			f.pass = &pass{name: "stack live oracle"}
			block := func() *Block {
				b := f.NewBlock(BlockPlain)
				if f.Entry == nil {
					f.Entry = b
				}
				return b
			}
			value := func(b *Block, op Op, args ...*Value) *Value {
				v := b.NewValue0(src.NoXPos, op, c.config.Types.Int64)
				v.AddArgs(args...)
				return v
			}
			var noSlot []*Value
			var spills [][2]*Value
			var forced []struct {
				b *Block
				v *Value
			}
			force := func(b *Block, v *Value) {
				forced = append(forced, struct {
					b *Block
					v *Value
				}{b, v})
			}
			switch name {
			case "differingPhiDiamond":
				e, l, r, j := block(), block(), block(), block()
				e.AddEdgeTo(l)
				e.AddEdgeTo(r)
				l.AddEdgeTo(j)
				r.AddEdgeTo(j)
				a, b := value(l, OpArg), value(r, OpArg)
				spill := value(l, OpStoreReg, a)
				spills = append(spills, [2]*Value{a, spill})
				value(j, OpPhi, a, b)
			case "duplicateEdges":
				e, j := block(), block()
				e.AddEdgeTo(j)
				e.AddEdgeTo(j)
				a, b := value(e, OpArg), value(e, OpArg)
				value(j, OpPhi, a, b)
				value(j, OpPhi, b, a)
			case "selfLoop":
				b := block()
				b.AddEdgeTo(b)
				a := value(b, OpArg)
				value(b, OpPhi, a)
			case "loopCarriedPhi":
				e, h, body := block(), block(), block()
				e.AddEdgeTo(h)
				h.AddEdgeTo(body)
				body.AddEdgeTo(h)
				a := value(e, OpArg)
				p := value(h, OpPhi)
				next := value(body, OpAdd64, p, a)
				p.AddArgs(a, next)
			case "forcedSpillAtDefinition":
				e, end := block(), block()
				e.AddEdgeTo(end)
				a := value(e, OpArg)
				s := value(e, OpStoreReg, a)
				force(e, s)
				force(e, s)
			case "ordinarySameBlock":
				b := block()
				a := value(b, OpArg)
				value(b, OpAdd64, a, a)
			case "phiSpillRegisterSource":
				e, j := block(), block()
				e.AddEdgeTo(j)
				a := value(e, OpArg)
				s := value(e, OpStoreReg, a)
				noSlot = append(noSlot, a)
				spills = append(spills, [2]*Value{a, s})
				value(j, OpPhi, a)
			case "disconnectedCyclesAndIsolated":
				block()
				a, b, isolated := block(), block(), block()
				a.AddEdgeTo(b)
				b.AddEdgeTo(a)
				x := value(a, OpArg)
				value(b, OpCopy, x)
				y := value(isolated, OpArg)
				force(isolated, y)
			case "idGapsAndEmptyBlocks":
				e, gap, middle, end := block(), block(), block(), block()
				f.Blocks = append(f.Blocks[:1], f.Blocks[2:]...)
				_ = gap
				e.AddEdgeTo(middle)
				middle.AddEdgeTo(end)
				dead := value(e, OpArg)
				a := value(e, OpArg)
				e.Values = e.Values[1:]
				_ = dead
				value(end, OpCopy, a)
				block()
			case "nonIDBlockOrder":
				e, middle, end := block(), block(), block()
				e.AddEdgeTo(middle)
				middle.AddEdgeTo(end)
				a := value(e, OpArg)
				noSlot = append(noSlot, value(end, OpCopy, a))
				f.Blocks[1], f.Blocks[2] = f.Blocks[2], f.Blocks[1]
			case "reusedMergePredecessor":
				e, j, k := block(), block(), block()
				e.AddEdgeTo(j)
				e.AddEdgeTo(k)
				j.AddEdgeTo(k)
				a, b := value(e, OpArg), value(e, OpArg)
				p := value(j, OpPhi, a)
				value(k, OpPhi, b, p)
				value(k, OpPhi, a, p)
			case "requeueAfterPopWithDistinctBits":
				d, h, u, v := block(), block(), block(), block()
				d.AddEdgeTo(h)
				h.AddEdgeTo(u)
				h.AddEdgeTo(v)
				a, b := value(d, OpArg), value(d, OpArg)
				noSlot = append(noSlot, value(u, OpCopy, a), value(v, OpCopy, b))
			case "denseSlots63", "denseSlots64", "denseSlots65":
				n := map[string]int{"denseSlots63": 63, "denseSlots64": 64, "denseSlots65": 65}[name]
				e, gap, middle, end := block(), block(), block(), block()
				f.Blocks = append(f.Blocks[:1], f.Blocks[2:]...)
				_ = gap
				e.AddEdgeTo(middle)
				middle.AddEdgeTo(end)
				for i := range n {
					value(e, OpArg)
					e.Values = e.Values[:len(e.Values)-1]
					noSlot = append(noSlot, value(e, OpArg))
					a := value(e, OpArg)
					useBlock := middle
					if i%2 != 0 {
						useBlock = end
					}
					noSlot = append(noSlot, value(useBlock, OpCopy, a))
				}
			case "nonadjacentDenseWords":
				e, middle, end := block(), block(), block()
				e.AddEdgeTo(middle)
				middle.AddEdgeTo(end)
				useBlocks := make([]*Block, 129)
				for i := range useBlocks {
					useBlocks[i] = e
				}
				useBlocks[0], useBlocks[128] = middle, end
				for _, useBlock := range useBlocks {
					value(e, OpArg)
					e.Values = e.Values[:len(e.Values)-1]
					noSlot = append(noSlot, value(e, OpArg))
					a := value(e, OpArg)
					noSlot = append(noSlot, value(useBlock, OpCopy, a))
				}
			}
			s := &stackAllocState{f: f, values: make([]stackValState, f.NumValues())}
			for _, b := range f.Blocks {
				for _, v := range b.Values {
					s.values[v.ID] = stackValState{typ: v.Type, needSlot: true, defBlock: b.ID, isArg: hasAnyArgOp(v)}
				}
			}
			for _, v := range noSlot {
				s.values[v.ID].needSlot = false
			}
			for _, pair := range spills {
				s.values[pair[0].ID].spill = pair[1]
			}
			spillLive := make([][]ID, f.NumBlocks())
			for _, seed := range forced {
				spillLive[seed.b.ID] = append(spillLive[seed.b.ID], seed.v.ID)
			}
			want := stackLiveEquationOracle(s, spillLive)
			s.computeLive(spillLive)
			for bid := range want {
				got := stackLiveOracleIDs(t, s.live, ID(bid))
				if len(got) != len(want[bid]) {
					t.Errorf("b%d live = %v, want set %v", bid, got, want[bid])
				}
				for i, id := range got {
					if !want[bid][id] {
						t.Errorf("b%d unexpected live v%d; want %v", bid, id, want[bid])
					}
					if i > 0 && got[i-1] >= id {
						t.Errorf("b%d live is not ascending unique: %v", bid, got)
					}
				}
			}
		})
	}
}

func stackLiveOracleIDs(t *testing.T, live stackLive, bid ID) []ID {
	t.Helper()
	var got []ID
	for _, chunk := range live.blockWords(bid) {
		word := chunk.bits
		for word != 0 {
			pos := chunk.index*64 + bits.TrailingZeros64(word)
			word &= word - 1
			if pos < 0 || pos >= len(live.ids) {
				t.Errorf("b%d has set padding position %d beyond %d slot IDs", bid, pos, len(live.ids))
				continue
			}
			got = append(got, live.ids[pos])
		}
	}
	return got
}

func stackLiveEquationOracle(s *stackAllocState, spillLive [][]ID) []map[ID]bool {
	n := s.f.NumBlocks()
	defs, uses, forced, in, out := make([]map[ID]bool, n), make([]map[ID]bool, n), make([]map[ID]bool, n), make([]map[ID]bool, n), make([]map[ID]bool, n)
	for i := range n {
		defs[i], uses[i], forced[i], in[i], out[i] = make(map[ID]bool), make(map[ID]bool), make(map[ID]bool), make(map[ID]bool), make(map[ID]bool)
	}
	seed := func(b ID, v ID) {
		if s.values[v].needSlot {
			forced[b][v] = true
		}
	}
	for _, b := range s.f.Blocks {
		for _, id := range spillLive[b.ID] {
			seed(b.ID, id)
		}
		for _, v := range b.Values {
			if s.values[v.ID].needSlot {
				defs[b.ID][v.ID] = true
			}
			for i, a := range v.Args {
				if v.Op == OpPhi {
					pred := b.Preds[i].b.ID
					seed(pred, a.ID)
					if spill := s.values[a.ID].spill; spill != nil {
						seed(pred, spill.ID)
					}
				} else if s.values[a.ID].needSlot {
					uses[b.ID][a.ID] = true
				}
			}
		}
	}
	for changed := true; changed; {
		changed = false
		for _, b := range s.f.Blocks {
			addOut := func(id ID) {
				if !out[b.ID][id] {
					out[b.ID][id] = true
					changed = true
				}
			}
			for id := range forced[b.ID] {
				addOut(id)
			}
			for _, edge := range b.Succs {
				for id := range in[edge.b.ID] {
					addOut(id)
				}
			}
			addIn := func(id ID) {
				if !defs[b.ID][id] && !in[b.ID][id] {
					in[b.ID][id] = true
					changed = true
				}
			}
			for id := range uses[b.ID] {
				addIn(id)
			}
			for id := range out[b.ID] {
				addIn(id)
			}
		}
	}
	return out
}
