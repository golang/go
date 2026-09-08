// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssacompile

import (
	"cmd/compile/internal/abi"
	"cmd/compile/internal/base"
	"cmd/compile/internal/ssa"
	"cmd/compile/internal/ssa/rewrite/amd64"
	"cmd/compile/internal/ssa/rewrite/amd64latelower"
	"cmd/compile/internal/ssa/rewrite/amd64splitload"
	"cmd/compile/internal/ssa/rewrite/arm"
	"cmd/compile/internal/ssa/rewrite/arm64"
	"cmd/compile/internal/ssa/rewrite/arm64latelower"
	"cmd/compile/internal/ssa/rewrite/i386"
	"cmd/compile/internal/ssa/rewrite/i386splitload"
	"cmd/compile/internal/ssa/rewrite/loong64"
	"cmd/compile/internal/ssa/rewrite/loong64latelower"
	"cmd/compile/internal/ssa/rewrite/mips"
	"cmd/compile/internal/ssa/rewrite/mips64"
	"cmd/compile/internal/ssa/rewrite/mips64latelower"
	"cmd/compile/internal/ssa/rewrite/ppc64"
	"cmd/compile/internal/ssa/rewrite/ppc64latelower"
	"cmd/compile/internal/ssa/rewrite/riscv64"
	"cmd/compile/internal/ssa/rewrite/riscv64latelower"
	"cmd/compile/internal/ssa/rewrite/s390x"
	"cmd/compile/internal/ssa/rewrite/wasm"
	"cmd/compile/internal/ssa/ssaop"
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
		c.LowerBlock = amd64.RewriteBlock
		c.LowerValue = amd64.RewriteValue
		c.LateLowerBlock = amd64latelower.RewriteBlock
		c.LateLowerValue = amd64latelower.RewriteValue
		c.SplitLoad = amd64splitload.RewriteValue
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
		c.LowerBlock = i386.RewriteBlock
		c.LowerValue = i386.RewriteValue
		c.SplitLoad = i386splitload.RewriteValue
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
		c.LowerBlock = arm.RewriteBlock
		c.LowerValue = arm.RewriteValue
		c.Registers = registersARM[:]
		c.GpRegMask = gpRegMaskARM
		c.FpRegMask = fpRegMaskARM
		c.FPReg = framepointerRegARM
		c.LinkReg = linkRegARM
		c.HasGReg = true
	case "arm64":
		c.PtrSize = 8
		c.RegSize = 8
		c.LowerBlock = arm64.RewriteBlock
		c.LowerValue = arm64.RewriteValue
		c.LateLowerBlock = arm64latelower.RewriteBlock
		c.LateLowerValue = arm64latelower.RewriteValue
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
		c.LowerBlock = ppc64.RewriteBlock
		c.LowerValue = ppc64.RewriteValue
		c.LateLowerBlock = ppc64latelower.RewriteBlock
		c.LateLowerValue = ppc64latelower.RewriteValue
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
		c.LowerBlock = mips64.RewriteBlock
		c.LowerValue = mips64.RewriteValue
		c.LateLowerBlock = mips64latelower.RewriteBlock
		c.LateLowerValue = mips64latelower.RewriteValue
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
		c.LowerBlock = loong64.RewriteBlock
		c.LowerValue = loong64.RewriteValue
		c.LateLowerBlock = loong64latelower.RewriteBlock
		c.LateLowerValue = loong64latelower.RewriteValue
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
		c.LowerBlock = s390x.RewriteBlock
		c.LowerValue = s390x.RewriteValue
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
		c.LowerBlock = mips.RewriteBlock
		c.LowerValue = mips.RewriteValue
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
		c.LowerBlock = riscv64.RewriteBlock
		c.LowerValue = riscv64.RewriteValue
		c.LateLowerBlock = riscv64latelower.RewriteBlock
		c.LateLowerValue = riscv64latelower.RewriteValue
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
		c.LowerBlock = wasm.RewriteBlock
		c.LowerValue = wasm.RewriteValue
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
