// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package arm64

import (
	"cmd/internal/obj"
	"cmd/internal/obj/arm64"
	"cmd/internal/src"
	"testing"
)

func TestPairSpills(t *testing.T) {
	movdLoad := func(dst, base int16, off int64) *obj.Prog {
		return &obj.Prog{
			As:   arm64.AMOVD,
			From: obj.Addr{Type: obj.TYPE_MEM, Reg: base, Offset: off, Name: obj.NAME_AUTO},
			To:   obj.Addr{Type: obj.TYPE_REG, Reg: dst},
		}
	}
	movdStore := func(src, base int16, off int64) *obj.Prog {
		return &obj.Prog{
			As:   arm64.AMOVD,
			From: obj.Addr{Type: obj.TYPE_REG, Reg: src},
			To:   obj.Addr{Type: obj.TYPE_MEM, Reg: base, Offset: off, Name: obj.NAME_AUTO},
		}
	}
	// param rewrites p's memory operand to NAME_PARAM.
	param := func(p *obj.Prog) *obj.Prog {
		if p.From.Type == obj.TYPE_MEM {
			p.From.Name = obj.NAME_PARAM
		} else {
			p.To.Name = obj.NAME_PARAM
		}
		return p
	}
	chain := func(progs ...*obj.Prog) *obj.Prog {
		for i := 0; i < len(progs)-1; i++ {
			progs[i].Link = progs[i+1]
		}
		return progs[0]
	}
	countAs := func(head *obj.Prog) map[obj.As]int {
		m := map[obj.As]int{}
		for p := head; p != nil; p = p.Link {
			m[p.As]++
		}
		return m
	}

	// pairWant describes the expected operands of the fused LDP/STP:
	// MOVDs of lo and hi (lo at the lower address off, hi at off+8)
	// against base, with the memory operand's addressing class name.
	type pairWant struct {
		as     obj.As
		base   int16
		off    int64
		lo, hi int16
		name   obj.AddrName
	}

	tests := []struct {
		name      string
		framesize int64
		setup     func() (*obj.Prog, []obj.JumpTable)
		wantLDP   int
		wantSTP   int
		wantMOVD  int
		wantNOP   int
		wantPair  *pairWant
	}{
		{
			name: "adjacent LDRs fuse to LDP",
			setup: func() (*obj.Prog, []obj.JumpTable) {
				return chain(
					movdLoad(arm64.REG_R0, arm64.REGSP, 16),
					movdLoad(arm64.REG_R1, arm64.REGSP, 24),
				), nil
			},
			wantLDP: 1, wantNOP: 1,
			wantPair: &pairWant{arm64.ALDP, arm64.REGSP, 16, arm64.REG_R0, arm64.REG_R1, obj.NAME_AUTO},
		},
		{
			name: "adjacent STRs fuse to STP",
			setup: func() (*obj.Prog, []obj.JumpTable) {
				return chain(
					movdStore(arm64.REG_R0, arm64.REGSP, 16),
					movdStore(arm64.REG_R1, arm64.REGSP, 24),
				), nil
			},
			wantSTP: 1, wantNOP: 1,
			wantPair: &pairWant{arm64.ASTP, arm64.REGSP, 16, arm64.REG_R0, arm64.REG_R1, obj.NAME_AUTO},
		},
		{
			name: "reverse-order LDRs fuse to LDP",
			setup: func() (*obj.Prog, []obj.JumpTable) {
				// p loads from the higher address, q from the lower.
				return chain(
					movdLoad(arm64.REG_R0, arm64.REGSP, 24),
					movdLoad(arm64.REG_R1, arm64.REGSP, 16),
				), nil
			},
			wantLDP: 1, wantNOP: 1,
			wantPair: &pairWant{arm64.ALDP, arm64.REGSP, 16, arm64.REG_R1, arm64.REG_R0, obj.NAME_AUTO},
		},
		{
			name: "reverse-order STRs fuse to STP",
			setup: func() (*obj.Prog, []obj.JumpTable) {
				// p stores to the higher address, q to the lower.
				return chain(
					movdStore(arm64.REG_R0, arm64.REGSP, 24),
					movdStore(arm64.REG_R1, arm64.REGSP, 16),
				), nil
			},
			wantSTP: 1, wantNOP: 1,
			wantPair: &pairWant{arm64.ASTP, arm64.REGSP, 16, arm64.REG_R1, arm64.REG_R0, obj.NAME_AUTO},
		},
		{
			name: "PARAM pair fuses",
			setup: func() (*obj.Prog, []obj.JumpTable) {
				return chain(
					param(movdLoad(arm64.REG_R0, arm64.REGSP, 16)),
					param(movdLoad(arm64.REG_R1, arm64.REGSP, 24)),
				), nil
			},
			wantLDP: 1, wantNOP: 1,
			wantPair: &pairWant{arm64.ALDP, arm64.REGSP, 16, arm64.REG_R0, arm64.REG_R1, obj.NAME_PARAM},
		},
		{
			name: "mixed AUTO and PARAM do not fuse",
			setup: func() (*obj.Prog, []obj.JumpTable) {
				return chain(
					movdLoad(arm64.REG_R0, arm64.REGSP, 16),
					param(movdLoad(arm64.REG_R1, arm64.REGSP, 24)),
				), nil
			},
			wantMOVD: 2,
		},
		{
			name: "non-adjacent offsets do not fuse",
			setup: func() (*obj.Prog, []obj.JumpTable) {
				return chain(
					movdLoad(arm64.REG_R0, arm64.REGSP, 16),
					movdLoad(arm64.REG_R1, arm64.REGSP, 32),
				), nil
			},
			wantMOVD: 2,
		},
		{
			name: "different base registers do not fuse",
			setup: func() (*obj.Prog, []obj.JumpTable) {
				return chain(
					movdLoad(arm64.REG_R0, arm64.REGSP, 16),
					movdLoad(arm64.REG_R1, arm64.REG_R28, 24),
				), nil
			},
			wantMOVD: 2,
		},
		{
			name: "load followed by store does not fuse",
			setup: func() (*obj.Prog, []obj.JumpTable) {
				return chain(
					movdLoad(arm64.REG_R0, arm64.REGSP, 16),
					movdStore(arm64.REG_R1, arm64.REGSP, 24),
				), nil
			},
			wantMOVD: 2,
		},
		{
			name: "same destination register does not fuse",
			setup: func() (*obj.Prog, []obj.JumpTable) {
				return chain(
					movdLoad(arm64.REG_R0, arm64.REGSP, 16),
					movdLoad(arm64.REG_R0, arm64.REGSP, 24),
				), nil
			},
			wantMOVD: 2,
		},
		{
			name: "first load writing the second's base does not fuse",
			setup: func() (*obj.Prog, []obj.JumpTable) {
				// Executed sequentially, the second load computes its
				// address from the value the first load just wrote into
				// R1; LDP would compute both addresses from the original
				// R1.
				return chain(
					movdLoad(arm64.REG_R1, arm64.REG_R1, 16),
					movdLoad(arm64.REG_R2, arm64.REG_R1, 24),
				), nil
			},
			wantMOVD: 2,
		},
		{
			name: "second load writing the base fuses",
			setup: func() (*obj.Prog, []obj.JumpTable) {
				// No address depends on the base after the second load
				// overwrites it, so LDP has identical semantics.
				return chain(
					movdLoad(arm64.REG_R0, arm64.REG_R5, 16),
					movdLoad(arm64.REG_R5, arm64.REG_R5, 24),
				), nil
			},
			wantLDP: 1, wantNOP: 1,
			wantPair: &pairWant{arm64.ALDP, arm64.REG_R5, 16, arm64.REG_R0, arm64.REG_R5, obj.NAME_AUTO},
		},
		{
			name: "store whose source is the base fuses",
			setup: func() (*obj.Prog, []obj.JumpTable) {
				// Stores never write registers, so there is no base
				// hazard for STP.
				return chain(
					movdStore(arm64.REG_R5, arm64.REG_R5, 16),
					movdStore(arm64.REG_R1, arm64.REG_R5, 24),
				), nil
			},
			wantSTP: 1, wantNOP: 1,
			wantPair: &pairWant{arm64.ASTP, arm64.REG_R5, 16, arm64.REG_R5, arm64.REG_R1, obj.NAME_AUTO},
		},
		{
			name: "skip when second instruction is a branch target",
			setup: func() (*obj.Prog, []obj.JumpTable) {
				p1 := movdLoad(arm64.REG_R0, arm64.REGSP, 16)
				p2 := movdLoad(arm64.REG_R1, arm64.REGSP, 24)
				br := &obj.Prog{As: arm64.AB, To: obj.Addr{Type: obj.TYPE_BRANCH}}
				br.To.SetTarget(p2)
				return chain(br, p1, p2), nil
			},
			wantMOVD: 2,
		},
		{
			name: "skip when second instruction is a backward-branch target",
			setup: func() (*obj.Prog, []obj.JumpTable) {
				// The branch sits after the pair, so target collection
				// must consider the whole Prog list, not just what
				// precedes the pair.
				p1 := movdLoad(arm64.REG_R0, arm64.REGSP, 16)
				p2 := movdLoad(arm64.REG_R1, arm64.REGSP, 24)
				br := &obj.Prog{As: arm64.AB, To: obj.Addr{Type: obj.TYPE_BRANCH}}
				br.To.SetTarget(p2)
				return chain(p1, p2, br), nil
			},
			wantMOVD: 2,
		},
		{
			name: "skip when second instruction is a jump-table target",
			setup: func() (*obj.Prog, []obj.JumpTable) {
				p1 := movdLoad(arm64.REG_R0, arm64.REGSP, 16)
				p2 := movdLoad(arm64.REG_R1, arm64.REGSP, 24)
				return chain(p1, p2), []obj.JumpTable{{Targets: []*obj.Prog{p2}}}
			},
			wantMOVD: 2,
		},
		{
			name: "skip when second instruction is a statement boundary",
			setup: func() (*obj.Prog, []obj.JumpTable) {
				// Statement-marked instructions (including those genssa
				// reuses as inline marks, which it promotes to
				// statements) must not become zero-sized. The statement
				// bit needs a known position to stick to.
				var tab src.PosTable
				pos := tab.XPos(src.MakePos(src.NewFileBase("f.go", "f.go"), 1, 1))
				p1 := movdLoad(arm64.REG_R0, arm64.REGSP, 16)
				p2 := movdLoad(arm64.REG_R1, arm64.REGSP, 24)
				p2.Pos = pos.WithIsStmt()
				return chain(p1, p2), nil
			},
			wantMOVD: 2,
		},
		{
			name: "at the encodable bound fuses",
			setup: func() (*obj.Prog, []obj.JumpTable) {
				// Resolved offset 496+0+8 = 504, LDP's maximum.
				return chain(
					movdLoad(arm64.REG_R0, arm64.REGSP, 496),
					movdLoad(arm64.REG_R1, arm64.REGSP, 504),
				), nil
			},
			wantLDP: 1, wantNOP: 1,
			wantPair: &pairWant{arm64.ALDP, arm64.REGSP, 496, arm64.REG_R0, arm64.REG_R1, obj.NAME_AUTO},
		},
		{
			name: "resolved offset out of LDP range does not fuse",
			setup: func() (*obj.Prog, []obj.JumpTable) {
				// Resolved offset 504+0+8 = 512, just past the bound.
				return chain(
					movdLoad(arm64.REG_R0, arm64.REGSP, 504),
					movdLoad(arm64.REG_R1, arm64.REGSP, 512),
				), nil
			},
			wantMOVD: 2,
		},
		{
			name:      "large frame pushes resolved offset out of range",
			framesize: 1024,
			setup: func() (*obj.Prog, []obj.JumpTable) {
				// 16+1024+8 = 1048: fusing would force an
				// assembler-synthesized address, no smaller than the
				// original pair.
				return chain(
					movdLoad(arm64.REG_R0, arm64.REGSP, 16),
					movdLoad(arm64.REG_R1, arm64.REGSP, 24),
				), nil
			},
			wantMOVD: 2,
		},
		{
			name:      "deep spill slots fuse in a large frame",
			framesize: 1024,
			setup: func() (*obj.Prog, []obj.JumpTable) {
				// -528+1024+8 = 504: encodable even though the
				// pre-resolution offset is far below -512.
				return chain(
					movdLoad(arm64.REG_R0, arm64.REGSP, -528),
					movdLoad(arm64.REG_R1, arm64.REGSP, -520),
				), nil
			},
			wantLDP: 1, wantNOP: 1,
			wantPair: &pairWant{arm64.ALDP, arm64.REGSP, -528, arm64.REG_R0, arm64.REG_R1, obj.NAME_AUTO},
		},
		{
			name: "misaligned offsets do not fuse",
			setup: func() (*obj.Prog, []obj.JumpTable) {
				return chain(
					movdLoad(arm64.REG_R0, arm64.REGSP, -12),
					movdLoad(arm64.REG_R1, arm64.REGSP, -4),
				), nil
			},
			wantMOVD: 2,
		},
		{
			name: "three adjacent loads fuse greedily",
			setup: func() (*obj.Prog, []obj.JumpTable) {
				return chain(
					movdLoad(arm64.REG_R0, arm64.REGSP, 16),
					movdLoad(arm64.REG_R1, arm64.REGSP, 24),
					movdLoad(arm64.REG_R2, arm64.REGSP, 32),
				), nil
			},
			wantLDP: 1, wantNOP: 1, wantMOVD: 1,
		},
		{
			name: "intervening instruction blocks fusion",
			setup: func() (*obj.Prog, []obj.JumpTable) {
				return chain(
					movdLoad(arm64.REG_R0, arm64.REGSP, 16),
					&obj.Prog{As: arm64.AHINT},
					movdLoad(arm64.REG_R1, arm64.REGSP, 24),
				), nil
			},
			wantMOVD: 2,
		},
		{
			name: "pre-indexed addressing does not fuse",
			setup: func() (*obj.Prog, []obj.JumpTable) {
				p1 := movdLoad(arm64.REG_R0, arm64.REGSP, 16)
				p1.Scond = arm64.C_XPRE
				return chain(p1, movdLoad(arm64.REG_R1, arm64.REGSP, 24)), nil
			},
			wantMOVD: 2,
		},
		{
			name: "post-indexed addressing does not fuse",
			setup: func() (*obj.Prog, []obj.JumpTable) {
				p1 := movdLoad(arm64.REG_R0, arm64.REGSP, 16)
				p1.Scond = arm64.C_XPOST
				return chain(p1, movdLoad(arm64.REG_R1, arm64.REGSP, 24)), nil
			},
			wantMOVD: 2,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			head, jumpTables := tt.setup()
			pairSpills(head, tt.framesize, jumpTables)
			got := countAs(head)
			if got[arm64.ALDP] != tt.wantLDP {
				t.Errorf("ALDP count = %d, want %d", got[arm64.ALDP], tt.wantLDP)
			}
			if got[arm64.ASTP] != tt.wantSTP {
				t.Errorf("ASTP count = %d, want %d", got[arm64.ASTP], tt.wantSTP)
			}
			if got[arm64.AMOVD] != tt.wantMOVD {
				t.Errorf("AMOVD count = %d, want %d", got[arm64.AMOVD], tt.wantMOVD)
			}
			if got[obj.ANOP] != tt.wantNOP {
				t.Errorf("ANOP count = %d, want %d", got[obj.ANOP], tt.wantNOP)
			}
			if w := tt.wantPair; w != nil {
				var fused *obj.Prog
				for p := head; p != nil; p = p.Link {
					if p.As == arm64.ALDP || p.As == arm64.ASTP {
						fused = p
						break
					}
				}
				if fused == nil {
					t.Fatal("no LDP/STP emitted")
				}
				mem, regs := &fused.From, &fused.To // LDP order
				if fused.As == arm64.ASTP {
					regs, mem = &fused.From, &fused.To
				}
				if fused.As != w.as {
					t.Errorf("fused As = %v, want %v", fused.As, w.as)
				}
				if mem.Type != obj.TYPE_MEM || mem.Reg != w.base || mem.Offset != w.off || mem.Name != w.name {
					t.Errorf("memory operand = {Type %v, Reg %v, Offset %d, Name %v}, want {TYPE_MEM, %v, %d, %v}",
						mem.Type, mem.Reg, mem.Offset, mem.Name, w.base, w.off, w.name)
				}
				if regs.Type != obj.TYPE_REGREG || regs.Reg != w.lo || regs.Offset != int64(w.hi) {
					t.Errorf("register pair = {Type %v, Reg %v, Offset %v}, want {TYPE_REGREG, %v, %v}",
						regs.Type, regs.Reg, regs.Offset, w.lo, w.hi)
				}
			}
		})
	}
}
