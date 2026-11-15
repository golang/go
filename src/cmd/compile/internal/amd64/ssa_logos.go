// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package amd64

import (
	"cmd/compile/internal/ssa"
	"cmd/compile/internal/ssagen"
	"cmd/internal/obj"
	"cmd/internal/obj/x86"
)

func ssaGenRORX(s *ssagen.State, v *ssa.Value) {
	maxRotate := int64(63)
	if v.Op == ssa.OpAMD64RORXL {
		maxRotate = 31
	}
	if v.AuxInt < 0 || v.AuxInt > maxRotate {
		v.Fatalf("rotate amount out of range [0,%d]: %d", maxRotate, v.AuxInt)
	}
	p := s.Prog(v.Op.Asm())
	p.From.Type = obj.TYPE_CONST
	p.From.Offset = v.AuxInt
	p.AddRestSourceReg(v.Args[0].Reg())
	p.To.Type = obj.TYPE_REG
	p.To.Reg = v.Reg()
}

func ssaGenLogosArenaBump(s *ssagen.State, v *ssa.Value) {
	chunk := v.Args[0].Reg()
	res := v.Reg0()
	tmp := v.RegTmp()
	rnd := v.AuxInt
	opregreg(s, x86.ATESTQ, chunk, chunk)
	fail := s.Prog(x86.AJEQ)
	fail.To.Type = obj.TYPE_BRANCH
	p := s.Prog(x86.AMOVQ)
	p.From.Type = obj.TYPE_MEM
	p.From.Reg = chunk
	p.From.Offset = 16
	p.To.Type = obj.TYPE_REG
	p.To.Reg = tmp
	p = s.Prog(x86.ALEAQ)
	p.From.Type = obj.TYPE_MEM
	p.From.Reg = tmp
	p.From.Offset = 7
	p.To.Type = obj.TYPE_REG
	p.To.Reg = tmp
	p = s.Prog(x86.AANDQ)
	p.From.Type = obj.TYPE_CONST
	p.From.Offset = -8
	p.To.Type = obj.TYPE_REG
	p.To.Reg = tmp
	p = s.Prog(x86.AMOVQ)
	p.From.Type = obj.TYPE_REG
	p.From.Reg = tmp
	p.To.Type = obj.TYPE_REG
	p.To.Reg = res
	p = s.Prog(x86.ALEAQ)
	p.From.Type = obj.TYPE_MEM
	p.From.Reg = tmp
	p.From.Offset = rnd
	p.To.Type = obj.TYPE_REG
	p.To.Reg = tmp
	p = s.Prog(x86.ACMPQ)
	p.From.Type = obj.TYPE_REG
	p.From.Reg = tmp
	p.To.Type = obj.TYPE_MEM
	p.To.Reg = chunk
	p.To.Offset = 24
	over := s.Prog(x86.AJHI)
	over.To.Type = obj.TYPE_BRANCH
	p = s.Prog(x86.AMOVQ)
	p.From.Type = obj.TYPE_REG
	p.From.Reg = tmp
	p.To.Type = obj.TYPE_MEM
	p.To.Reg = chunk
	p.To.Offset = 16
	done := s.Prog(obj.AJMP)
	done.To.Type = obj.TYPE_BRANCH
	fail.To.SetTarget(s.Pc())
	over.To.SetTarget(s.Pc())
	opregreg(s, x86.AXORL, res, res)
	done.To.SetTarget(s.Pc())
}

func ssaGenLogosRegionBump(s *ssagen.State, v *ssa.Value) {
	h := v.Args[0].Reg()
	gen := v.Args[1].Reg()
	res := v.Reg0()
	tmp := v.RegTmp()
	opregreg(s, x86.ATESTQ, h, h)
	failNil := s.Prog(x86.AJEQ)
	failNil.To.Type = obj.TYPE_BRANCH
	getgFromTLS(s, res)
	p := s.Prog(x86.AMOVQ)
	p.From.Type = obj.TYPE_MEM
	p.From.Reg = h
	p.From.Offset = ssa.LogosRegionOwnerOffset
	p.To.Type = obj.TYPE_REG
	p.To.Reg = tmp
	p = s.Prog(x86.ACMPQ)
	p.From.Type = obj.TYPE_REG
	p.From.Reg = tmp
	p.To.Type = obj.TYPE_REG
	p.To.Reg = res
	ownerStale := s.Prog(x86.AJNE)
	ownerStale.To.Type = obj.TYPE_BRANCH
	p = s.Prog(x86.AMOVQ)
	p.From.Type = obj.TYPE_MEM
	p.From.Reg = h
	p.From.Offset = ssa.LogosRegionChunkOffset
	p.To.Type = obj.TYPE_REG
	p.To.Reg = tmp
	opregreg(s, x86.ATESTQ, tmp, tmp)
	chunkNil := s.Prog(x86.AJEQ)
	chunkNil.To.Type = obj.TYPE_BRANCH
	p = s.Prog(x86.ACMPQ)
	p.From.Type = obj.TYPE_REG
	p.From.Reg = gen
	p.To.Type = obj.TYPE_MEM
	p.To.Reg = h
	p.To.Offset = ssa.LogosRegionGenOffset
	genStale := s.Prog(x86.AJNE)
	genStale.To.Type = obj.TYPE_BRANCH
	p = s.Prog(x86.AMOVQ)
	p.From.Type = obj.TYPE_MEM
	p.From.Reg = tmp
	p.From.Offset = ssa.LogosRegionNoscanCurOffset
	p.To.Type = obj.TYPE_REG
	p.To.Reg = res
	p = s.Prog(x86.ALEAQ)
	p.From.Type = obj.TYPE_MEM
	p.From.Reg = res
	p.From.Offset = 7
	p.To.Type = obj.TYPE_REG
	p.To.Reg = res
	p = s.Prog(x86.AANDQ)
	p.From.Type = obj.TYPE_CONST
	p.From.Offset = -8
	p.To.Type = obj.TYPE_REG
	p.To.Reg = res
	p = s.Prog(x86.ALEAQ)
	p.From.Type = obj.TYPE_MEM
	p.From.Reg = res
	p.From.Offset = v.AuxInt
	p.To.Type = obj.TYPE_REG
	p.To.Reg = res
	p = s.Prog(x86.ACMPQ)
	p.From.Type = obj.TYPE_REG
	p.From.Reg = res
	p.To.Type = obj.TYPE_MEM
	p.To.Reg = tmp
	p.To.Offset = ssa.LogosRegionScanCurOffset
	over := s.Prog(x86.AJHI)
	over.To.Type = obj.TYPE_BRANCH
	p = s.Prog(x86.AMOVQ)
	p.From.Type = obj.TYPE_REG
	p.From.Reg = res
	p.To.Type = obj.TYPE_MEM
	p.To.Reg = tmp
	p.To.Offset = ssa.LogosRegionNoscanCurOffset
	p = s.Prog(x86.ASUBQ)
	p.From.Type = obj.TYPE_CONST
	p.From.Offset = v.AuxInt
	p.To.Type = obj.TYPE_REG
	p.To.Reg = res
	done := s.Prog(obj.AJMP)
	done.To.Type = obj.TYPE_BRANCH
	failNil.To.SetTarget(s.Pc())
	ownerStale.To.SetTarget(s.Pc())
	chunkNil.To.SetTarget(s.Pc())
	genStale.To.SetTarget(s.Pc())
	over.To.SetTarget(s.Pc())
	opregreg(s, x86.AXORL, res, res)
	done.To.SetTarget(s.Pc())
}
