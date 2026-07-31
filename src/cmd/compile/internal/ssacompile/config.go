// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssacompile

import (
	"cmd/compile/internal/abi"
	"cmd/compile/internal/base"
	"cmd/compile/internal/ssa"
	"cmd/compile/internal/ssa/ssaop"
	"cmd/compile/internal/ssarewrite/rewrite386"
	"cmd/compile/internal/ssarewrite/rewrite386splitload"
	"cmd/compile/internal/ssarewrite/rewriteamd64"
	"cmd/compile/internal/ssarewrite/rewriteamd64latelower"
	"cmd/compile/internal/ssarewrite/rewriteamd64splitload"
	"cmd/compile/internal/ssarewrite/rewritearm"
	"cmd/compile/internal/ssarewrite/rewritearm64"
	"cmd/compile/internal/ssarewrite/rewritearm64latelower"
	"cmd/compile/internal/ssarewrite/rewriteloong64"
	"cmd/compile/internal/ssarewrite/rewriteloong64latelower"
	"cmd/compile/internal/ssarewrite/rewritemips"
	"cmd/compile/internal/ssarewrite/rewritemips64"
	"cmd/compile/internal/ssarewrite/rewritemips64latelower"
	"cmd/compile/internal/ssarewrite/rewriteppc64"
	"cmd/compile/internal/ssarewrite/rewriteppc64latelower"
	"cmd/compile/internal/ssarewrite/rewriteriscv64"
	"cmd/compile/internal/ssarewrite/rewriteriscv64latelower"
	"cmd/compile/internal/ssarewrite/rewrites390x"
	"cmd/compile/internal/ssarewrite/rewritewasm"
	"cmd/internal/obj"
)

func NewConfig(softFloat bool) *ssa.Config {
	types_ := ssa.NewTypes()
	return newConfig(base.Ctxt.Arch.Name, *types_, base.Ctxt, base.Flag.N == 0, softFloat)
}

// newConfig returns a new configuration object for the given architecture.
func newConfig(arch string, types ssa.Types, ctxt *obj.Link, optimize, softfloat bool) *ssa.Config {
	c := &ssa.Config{Arch: arch, Types: types}
	switch arch {
	case "amd64":
		c.PtrSize = 8
		c.RegSize = 8
		c.LowerBlock = rewriteamd64.RewriteBlock
		c.LowerValue = rewriteamd64.RewriteValue
		c.LateLowerBlock = rewriteamd64latelower.RewriteBlock
		c.LateLowerValue = rewriteamd64latelower.RewriteValue
		c.SplitLoad = rewriteamd64splitload.RewriteValue
		c.Registers = registersAMD64[:]
		c.GpRegMask = gpRegMaskAMD64
		c.FpRegMask = fpRegMaskAMD64
		c.SimdRegMask = simdRegMaskAMD64
		c.SpecialRegMask = specialRegMaskAMD64
		c.IntParamRegs = paramIntRegAMD64
		c.FloatParamRegs = paramFloatRegAMD64
		c.FPReg = framepointerRegAMD64
		c.LinkReg = linkRegAMD64
		c.HasGReg = true
		c.UnalignedOK = true
		c.HaveBswap64 = true
		c.HaveBswap32 = true
		c.HaveBswap16 = true
		c.HaveCondSelect = true
	case "386":
		c.PtrSize = 4
		c.RegSize = 4
		c.LowerBlock = rewrite386.RewriteBlock
		c.LowerValue = rewrite386.RewriteValue
		c.SplitLoad = rewrite386splitload.RewriteValue
		c.Registers = registers386[:]
		c.GpRegMask = gpRegMask386
		c.FpRegMask = fpRegMask386
		c.FPReg = framepointerReg386
		c.LinkReg = linkReg386
		c.HasGReg = false
		c.UnalignedOK = true
		c.HaveBswap32 = true
		c.HaveBswap16 = true
	case "arm":
		c.PtrSize = 4
		c.RegSize = 4
		c.LowerBlock = rewritearm.RewriteBlock
		c.LowerValue = rewritearm.RewriteValue
		c.Registers = registersARM[:]
		c.GpRegMask = gpRegMaskARM
		c.FpRegMask = fpRegMaskARM
		c.FPReg = framepointerRegARM
		c.LinkReg = linkRegARM
		c.HasGReg = true
	case "arm64":
		c.PtrSize = 8
		c.RegSize = 8
		c.LowerBlock = rewritearm64.RewriteBlock
		c.LowerValue = rewritearm64.RewriteValue
		c.LateLowerBlock = rewritearm64latelower.RewriteBlock
		c.LateLowerValue = rewritearm64latelower.RewriteValue
		c.Registers = registersARM64[:]
		c.GpRegMask = gpRegMaskARM64
		c.FpRegMask = fpRegMaskARM64
		c.SimdRegMask = simdRegMaskARM64
		c.SpecialRegMask = specialRegMaskARM64
		c.IntParamRegs = paramIntRegARM64
		c.FloatParamRegs = paramFloatRegARM64
		c.FPReg = framepointerRegARM64
		c.LinkReg = linkRegARM64
		c.HasGReg = true
		c.UnalignedOK = true
		c.HaveBswap64 = true
		c.HaveBswap32 = true
		c.HaveBswap16 = true
		c.HaveCondSelect = true
	case "ppc64":
		c.BigEndian = true
		fallthrough
	case "ppc64le":
		c.PtrSize = 8
		c.RegSize = 8
		c.LowerBlock = rewriteppc64.RewriteBlock
		c.LowerValue = rewriteppc64.RewriteValue
		c.LateLowerBlock = rewriteppc64latelower.RewriteBlock
		c.LateLowerValue = rewriteppc64latelower.RewriteValue
		c.Registers = registersPPC64[:]
		c.GpRegMask = gpRegMaskPPC64
		c.FpRegMask = fpRegMaskPPC64
		c.SpecialRegMask = specialRegMaskPPC64
		c.IntParamRegs = paramIntRegPPC64
		c.FloatParamRegs = paramFloatRegPPC64
		c.FPReg = framepointerRegPPC64
		c.LinkReg = linkRegPPC64
		c.HasGReg = true
		c.UnalignedOK = true
		// Note: ppc64 has register bswap ops only when GOPPC64>=10.
		// But it has bswap+load and bswap+store ops for all ppc64 variants.
		// That is the sense we're using them here - they are only used
		// in contexts where they can be merged with a load or store.
		c.HaveBswap64 = true
		c.HaveBswap32 = true
		c.HaveBswap16 = true
		c.HaveCondSelect = true
	case "mips64":
		c.BigEndian = true
		fallthrough
	case "mips64le":
		c.PtrSize = 8
		c.RegSize = 8
		c.LowerBlock = rewritemips64.RewriteBlock
		c.LowerValue = rewritemips64.RewriteValue
		c.LateLowerBlock = rewritemips64latelower.RewriteBlock
		c.LateLowerValue = rewritemips64latelower.RewriteValue
		c.Registers = registersMIPS64[:]
		c.GpRegMask = gpRegMaskMIPS64
		c.FpRegMask = fpRegMaskMIPS64
		c.SpecialRegMask = specialRegMaskMIPS64
		c.FPReg = framepointerRegMIPS64
		c.LinkReg = linkRegMIPS64
		c.HasGReg = true
	case "loong64":
		c.PtrSize = 8
		c.RegSize = 8
		c.LowerBlock = rewriteloong64.RewriteBlock
		c.LowerValue = rewriteloong64.RewriteValue
		c.LateLowerBlock = rewriteloong64latelower.RewriteBlock
		c.LateLowerValue = rewriteloong64latelower.RewriteValue
		c.Registers = registersLOONG64[:]
		c.GpRegMask = gpRegMaskLOONG64
		c.FpRegMask = fpRegMaskLOONG64
		c.IntParamRegs = paramIntRegLOONG64
		c.FloatParamRegs = paramFloatRegLOONG64
		c.FPReg = framepointerRegLOONG64
		c.LinkReg = linkRegLOONG64
		c.HasGReg = true
		c.UnalignedOK = true
		c.HaveBswap64 = true
		c.HaveBswap32 = true
		c.HaveBswap16 = true
		c.HaveCondSelect = true
	case "s390x":
		c.PtrSize = 8
		c.RegSize = 8
		c.LowerBlock = rewrites390x.RewriteBlock
		c.LowerValue = rewrites390x.RewriteValue
		c.Registers = registersS390X[:]
		c.GpRegMask = gpRegMaskS390X
		c.FpRegMask = fpRegMaskS390X
		c.IntParamRegs = paramIntRegS390X
		c.FloatParamRegs = paramFloatRegS390X
		c.FPReg = framepointerRegS390X
		c.LinkReg = linkRegS390X
		c.HasGReg = true
		c.BigEndian = true
		c.UnalignedOK = true
		c.HaveBswap64 = true
		c.HaveBswap32 = true
		c.HaveBswap16 = true // only for loads&stores, see ppc64 comment
	case "mips":
		c.BigEndian = true
		fallthrough
	case "mipsle":
		c.PtrSize = 4
		c.RegSize = 4
		c.LowerBlock = rewritemips.RewriteBlock
		c.LowerValue = rewritemips.RewriteValue
		c.Registers = registersMIPS[:]
		c.GpRegMask = gpRegMaskMIPS
		c.FpRegMask = fpRegMaskMIPS
		c.SpecialRegMask = specialRegMaskMIPS
		c.FPReg = framepointerRegMIPS
		c.LinkReg = linkRegMIPS
		c.HasGReg = true
	case "riscv64":
		c.PtrSize = 8
		c.RegSize = 8
		c.LowerBlock = rewriteriscv64.RewriteBlock
		c.LowerValue = rewriteriscv64.RewriteValue
		c.LateLowerBlock = rewriteriscv64latelower.RewriteBlock
		c.LateLowerValue = rewriteriscv64latelower.RewriteValue
		c.Registers = registersRISCV64[:]
		c.GpRegMask = gpRegMaskRISCV64
		c.FpRegMask = fpRegMaskRISCV64
		c.IntParamRegs = paramIntRegRISCV64
		c.FloatParamRegs = paramFloatRegRISCV64
		c.FPReg = framepointerRegRISCV64
		c.HasGReg = true
	case "wasm":
		c.PtrSize = 8
		c.RegSize = 8
		c.LowerBlock = rewritewasm.RewriteBlock
		c.LowerValue = rewritewasm.RewriteValue
		c.Registers = registersWasm[:]
		c.GpRegMask = gpRegMaskWasm
		c.FpRegMask = fpRegMaskWasm
		c.Fp32RegMask = fp32RegMaskWasm
		c.Fp64RegMask = fp64RegMaskWasm
		c.SimdRegMask = simdRegMaskWasm
		c.FPReg = framepointerRegWasm
		c.LinkReg = linkRegWasm
		c.HasGReg = true
		c.UnalignedOK = true
		c.HaveCondSelect = true
	default:
		ctxt.Diag("arch %s not implemented", arch)
	}
	c.Ctxt = ctxt
	c.Optimize = optimize
	c.SoftFloat = softfloat
	if softfloat {
		c.FloatParamRegs = nil // no FP registers in softfloat mode
	}

	c.ABI0 = abi.NewABIConfig(0, 0, ctxt.Arch.FixedFrameSize, 0)
	c.ABI1 = abi.NewABIConfig(len(c.IntParamRegs), len(c.FloatParamRegs), ctxt.Arch.FixedFrameSize, 1)

	if ctxt.Flag_shared {
		// LoweredWB is secretly a CALL and CALLs on 386 in
		// shared mode get rewritten by obj6.go to go through
		// the GOT, which clobbers BX.
		ssaop.OpcodeTable[ssaop.Op386LoweredWB].Reg.Clobbers = ssaop.OpcodeTable[ssaop.Op386LoweredWB].Reg.Clobbers.AddReg(3) // BX
	}

	c.BuildRecipes(arch)

	return c
}
