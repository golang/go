// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import (
	"cmd/compile/internal/abi"
	"cmd/compile/internal/ssa/ssacore"
	"cmd/compile/internal/ssa/ssaop"
	"cmd/internal/obj"
)

// NewConfig returns a new configuration object for the given architecture.
func NewConfig(arch string, types ssacore.Types, ctxt *obj.Link, optimize, softfloat bool) *ssacore.Config {
	c := &ssacore.Config{Arch: arch, Types: types}
	switch arch {
	case "amd64":
		c.PtrSize = 8
		c.RegSize = 8
		c.LowerBlock = rewriteBlockAMD64
		c.LowerValue = rewriteValueAMD64
		c.LateLowerBlock = rewriteBlockAMD64latelower
		c.LateLowerValue = rewriteValueAMD64latelower
		c.SplitLoad = rewriteValueAMD64splitload
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
		c.LowerBlock = rewriteBlock386
		c.LowerValue = rewriteValue386
		c.SplitLoad = rewriteValue386splitload
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
		c.LowerBlock = rewriteBlockARM
		c.LowerValue = rewriteValueARM
		c.Registers = registersARM[:]
		c.GpRegMask = gpRegMaskARM
		c.FpRegMask = fpRegMaskARM
		c.FPReg = framepointerRegARM
		c.LinkReg = linkRegARM
		c.HasGReg = true
	case "arm64":
		c.PtrSize = 8
		c.RegSize = 8
		c.LowerBlock = rewriteBlockARM64
		c.LowerValue = rewriteValueARM64
		c.LateLowerBlock = rewriteBlockARM64latelower
		c.LateLowerValue = rewriteValueARM64latelower
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
		c.LowerBlock = rewriteBlockPPC64
		c.LowerValue = rewriteValuePPC64
		c.LateLowerBlock = rewriteBlockPPC64latelower
		c.LateLowerValue = rewriteValuePPC64latelower
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
		c.LowerBlock = rewriteBlockMIPS64
		c.LowerValue = rewriteValueMIPS64
		c.LateLowerBlock = rewriteBlockMIPS64latelower
		c.LateLowerValue = rewriteValueMIPS64latelower
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
		c.LowerBlock = rewriteBlockLOONG64
		c.LowerValue = rewriteValueLOONG64
		c.LateLowerBlock = rewriteBlockLOONG64latelower
		c.LateLowerValue = rewriteValueLOONG64latelower
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
		c.LowerBlock = rewriteBlockS390X
		c.LowerValue = rewriteValueS390X
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
		c.LowerBlock = rewriteBlockMIPS
		c.LowerValue = rewriteValueMIPS
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
		c.LowerBlock = rewriteBlockRISCV64
		c.LowerValue = rewriteValueRISCV64
		c.LateLowerBlock = rewriteBlockRISCV64latelower
		c.LateLowerValue = rewriteValueRISCV64latelower
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
		c.LowerBlock = rewriteBlockWasm
		c.LowerValue = rewriteValueWasm
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
