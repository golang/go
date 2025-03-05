// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import (
	"cmd/compile/internal/abi"
	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/types"
	"cmd/internal/obj"
	"cmd/internal/src"
	"internal/buildcfg"
)

// A Config holds readonly compilation information.
// It is created once, early during compilation,
// and shared across all compilations.
type Config struct {
	arch           string // "amd64", etc.
	PtrSize        int64  // 4 or 8; copy of cmd/internal/sys.Arch.PtrSize
	RegSize        int64  // 4 or 8; copy of cmd/internal/sys.Arch.RegSize
	Types          Types
	lowerBlock     blockRewriter  // block lowering function, first round
	lowerValue     valueRewriter  // value lowering function, first round
	lateLowerBlock blockRewriter  // block lowering function that needs to be run after the first round; only used on some architectures
	lateLowerValue valueRewriter  // value lowering function that needs to be run after the first round; only used on some architectures
	splitLoad      valueRewriter  // function for splitting merged load ops; only used on some architectures
	registers      []Register     // machine registers
	gpRegMask      regMask        // general purpose integer register mask
	fpRegMask      regMask        // floating point register mask
	fp32RegMask    regMask        // floating point register mask
	fp64RegMask    regMask        // floating point register mask
	specialRegMask regMask        // special register mask
	intParamRegs   []int8         // register numbers of integer param (in/out) registers
	floatParamRegs []int8         // register numbers of floating param (in/out) registers
	ABI1           *abi.ABIConfig // "ABIInternal" under development // TODO change comment when this becomes current
	ABI0           *abi.ABIConfig
	FPReg          int8      // register number of frame pointer, -1 if not used
	LinkReg        int8      // register number of link register if it is a general purpose register, -1 if not used
	hasGReg        bool      // has hardware g register
	ctxt           *obj.Link // Generic arch information
	optimize       bool      // Do optimization
	noDuffDevice   bool      // Don't use Duff's device
	useAvg         bool      // Use optimizations that need Avg* operations
	useHmul        bool      // Use optimizations that need Hmul* operations
	SoftFloat      bool      //
	Race           bool      // race detector enabled
	BigEndian      bool      //
	UseFMA         bool      // Use hardware FMA operation
	unalignedOK    bool      // Unaligned loads/stores are ok
	haveBswap64    bool      // architecture implements Bswap64
	haveBswap32    bool      // architecture implements Bswap32
	haveBswap16    bool      // architecture implements Bswap16
}

type (
	blockRewriter func(*Block) bool
	valueRewriter func(*Value) bool
)

type Types struct {
	Bool       *types.Type
	Int8       *types.Type
	Int16      *types.Type
	Int32      *types.Type
	Int64      *types.Type
	UInt8      *types.Type
	UInt16     *types.Type
	UInt32     *types.Type
	UInt64     *types.Type
	Int        *types.Type
	Float32    *types.Type
	Float64    *types.Type
	UInt       *types.Type
	Uintptr    *types.Type
	String     *types.Type
	BytePtr    *types.Type // TODO: use unsafe.Pointer instead?
	Int32Ptr   *types.Type
	UInt32Ptr  *types.Type
	IntPtr     *types.Type
	UintptrPtr *types.Type
	Float32Ptr *types.Type
	Float64Ptr *types.Type
	BytePtrPtr *types.Type
}

// NewTypes creates and populates a Types.
func NewTypes() *Types {
	t := new(Types)
	t.SetTypPtrs()
	return t
}

// SetTypPtrs populates t.
func (t *Types) SetTypPtrs() {
	t.Bool = types.Types[types.TBOOL]
	t.Int8 = types.Types[types.TINT8]
	t.Int16 = types.Types[types.TINT16]
	t.Int32 = types.Types[types.TINT32]
	t.Int64 = types.Types[types.TINT64]
	t.UInt8 = types.Types[types.TUINT8]
	t.UInt16 = types.Types[types.TUINT16]
	t.UInt32 = types.Types[types.TUINT32]
	t.UInt64 = types.Types[types.TUINT64]
	t.Int = types.Types[types.TINT]
	t.Float32 = types.Types[types.TFLOAT32]
	t.Float64 = types.Types[types.TFLOAT64]
	t.UInt = types.Types[types.TUINT]
	t.Uintptr = types.Types[types.TUINTPTR]
	t.String = types.Types[types.TSTRING]
	t.BytePtr = types.NewPtr(types.Types[types.TUINT8])
	t.Int32Ptr = types.NewPtr(types.Types[types.TINT32])
	t.UInt32Ptr = types.NewPtr(types.Types[types.TUINT32])
	t.IntPtr = types.NewPtr(types.Types[types.TINT])
	t.UintptrPtr = types.NewPtr(types.Types[types.TUINTPTR])
	t.Float32Ptr = types.NewPtr(types.Types[types.TFLOAT32])
	t.Float64Ptr = types.NewPtr(types.Types[types.TFLOAT64])
	t.BytePtrPtr = types.NewPtr(types.NewPtr(types.Types[types.TUINT8]))
}

type Logger interface {
	// Logf logs a message from the compiler.
	Logf(string, ...interface{})

	// Log reports whether logging is not a no-op
	// some logging calls account for more than a few heap allocations.
	Log() bool

	// Fatalf reports a compiler error and exits.
	Fatalf(pos src.XPos, msg string, args ...interface{})

	// Warnl writes compiler messages in the form expected by "errorcheck" tests
	Warnl(pos src.XPos, fmt_ string, args ...interface{})

	// Forwards the Debug flags from gc
	Debug_checknil() bool
}

type Frontend interface {
	Logger

	// StringData returns a symbol pointing to the given string's contents.
	StringData(string) *obj.LSym

	// Given the name for a compound type, returns the name we should use
	// for the parts of that compound type.
	SplitSlot(parent *LocalSlot, suffix string, offset int64, t *types.Type) LocalSlot

	// Syslook returns a symbol of the runtime function/variable with the
	// given name.
	Syslook(string) *obj.LSym

	// UseWriteBarrier reports whether write barrier is enabled
	UseWriteBarrier() bool

	// Func returns the ir.Func of the function being compiled.
	Func() *ir.Func
}

// NewConfig returns a new configuration object for the given architecture.
func NewConfig(arch string, types Types, ctxt *obj.Link, optimize, softfloat bool) *Config {
	c := &Config{arch: arch, Types: types}
	c.useAvg = true
	c.useHmul = true
	switch arch {
	case "amd64":
		c.PtrSize = 8
		c.RegSize = 8
		c.lowerBlock = rewriteBlockAMD64
		c.lowerValue = rewriteValueAMD64
		c.lateLowerBlock = rewriteBlockAMD64latelower
		c.lateLowerValue = rewriteValueAMD64latelower
		c.splitLoad = rewriteValueAMD64splitload
		c.registers = registersAMD64[:]
		c.gpRegMask = gpRegMaskAMD64
		c.fpRegMask = fpRegMaskAMD64
		c.specialRegMask = specialRegMaskAMD64
		c.intParamRegs = paramIntRegAMD64
		c.floatParamRegs = paramFloatRegAMD64
		c.FPReg = framepointerRegAMD64
		c.LinkReg = linkRegAMD64
		c.hasGReg = true
		c.unalignedOK = true
		c.haveBswap64 = true
		c.haveBswap32 = true
		c.haveBswap16 = true
	case "386":
		c.PtrSize = 4
		c.RegSize = 4
		c.lowerBlock = rewriteBlock386
		c.lowerValue = rewriteValue386
		c.splitLoad = rewriteValue386splitload
		c.registers = registers386[:]
		c.gpRegMask = gpRegMask386
		c.fpRegMask = fpRegMask386
		c.FPReg = framepointerReg386
		c.LinkReg = linkReg386
		c.hasGReg = false
		c.unalignedOK = true
		c.haveBswap32 = true
		c.haveBswap16 = true
	case "arm":
		c.PtrSize = 4
		c.RegSize = 4
		c.lowerBlock = rewriteBlockARM
		c.lowerValue = rewriteValueARM
		c.registers = registersARM[:]
		c.gpRegMask = gpRegMaskARM
		c.fpRegMask = fpRegMaskARM
		c.FPReg = framepointerRegARM
		c.LinkReg = linkRegARM
		c.hasGReg = true
	case "arm64":
		c.PtrSize = 8
		c.RegSize = 8
		c.lowerBlock = rewriteBlockARM64
		c.lowerValue = rewriteValueARM64
		c.lateLowerBlock = rewriteBlockARM64latelower
		c.lateLowerValue = rewriteValueARM64latelower
		c.registers = registersARM64[:]
		c.gpRegMask = gpRegMaskARM64
		c.fpRegMask = fpRegMaskARM64
		c.intParamRegs = paramIntRegARM64
		c.floatParamRegs = paramFloatRegARM64
		c.FPReg = framepointerRegARM64
		c.LinkReg = linkRegARM64
		c.hasGReg = true
		c.unalignedOK = true
		c.haveBswap64 = true
		c.haveBswap32 = true
		c.haveBswap16 = true
	case "ppc64":
		c.BigEndian = true
		fallthrough
	case "ppc64le":
		c.PtrSize = 8
		c.RegSize = 8
		c.lowerBlock = rewriteBlockPPC64
		c.lowerValue = rewriteValuePPC64
		c.lateLowerBlock = rewriteBlockPPC64latelower
		c.lateLowerValue = rewriteValuePPC64latelower
		c.registers = registersPPC64[:]
		c.gpRegMask = gpRegMaskPPC64
		c.fpRegMask = fpRegMaskPPC64
		c.specialRegMask = specialRegMaskPPC64
		c.intParamRegs = paramIntRegPPC64
		c.floatParamRegs = paramFloatRegPPC64
		c.FPReg = framepointerRegPPC64
		c.LinkReg = linkRegPPC64
		c.hasGReg = true
		c.unalignedOK = true
		// Note: ppc64 has register bswap ops only when GOPPC64>=10.
		// But it has bswap+load and bswap+store ops for all ppc64 variants.
		// That is the sense we're using them here - they are only used
		// in contexts where they can be merged with a load or store.
		c.haveBswap64 = true
		c.haveBswap32 = true
		c.haveBswap16 = true
	case "mips64":
		c.BigEndian = true
		fallthrough
	case "mips64le":
		c.PtrSize = 8
		c.RegSize = 8
		c.lowerBlock = rewriteBlockMIPS64
		c.lowerValue = rewriteValueMIPS64
		c.registers = registersMIPS64[:]
		c.gpRegMask = gpRegMaskMIPS64
		c.fpRegMask = fpRegMaskMIPS64
		c.specialRegMask = specialRegMaskMIPS64
		c.FPReg = framepointerRegMIPS64
		c.LinkReg = linkRegMIPS64
		c.hasGReg = true
	case "loong64":
		c.PtrSize = 8
		c.RegSize = 8
		c.lowerBlock = rewriteBlockLOONG64
		c.lowerValue = rewriteValueLOONG64
		c.registers = registersLOONG64[:]
		c.gpRegMask = gpRegMaskLOONG64
		c.fpRegMask = fpRegMaskLOONG64
		c.intParamRegs = paramIntRegLOONG64
		c.floatParamRegs = paramFloatRegLOONG64
		c.FPReg = framepointerRegLOONG64
		c.LinkReg = linkRegLOONG64
		c.hasGReg = true
	case "s390x":
		c.PtrSize = 8
		c.RegSize = 8
		c.lowerBlock = rewriteBlockS390X
		c.lowerValue = rewriteValueS390X
		c.registers = registersS390X[:]
		c.gpRegMask = gpRegMaskS390X
		c.fpRegMask = fpRegMaskS390X
		c.FPReg = framepointerRegS390X
		c.LinkReg = linkRegS390X
		c.hasGReg = true
		c.noDuffDevice = true
		c.BigEndian = true
		c.unalignedOK = true
		c.haveBswap64 = true
		c.haveBswap32 = true
		c.haveBswap16 = true // only for loads&stores, see ppc64 comment
	case "mips":
		c.BigEndian = true
		fallthrough
	case "mipsle":
		c.PtrSize = 4
		c.RegSize = 4
		c.lowerBlock = rewriteBlockMIPS
		c.lowerValue = rewriteValueMIPS
		c.registers = registersMIPS[:]
		c.gpRegMask = gpRegMaskMIPS
		c.fpRegMask = fpRegMaskMIPS
		c.specialRegMask = specialRegMaskMIPS
		c.FPReg = framepointerRegMIPS
		c.LinkReg = linkRegMIPS
		c.hasGReg = true
		c.noDuffDevice = true
	case "riscv64":
		c.PtrSize = 8
		c.RegSize = 8
		c.lowerBlock = rewriteBlockRISCV64
		c.lowerValue = rewriteValueRISCV64
		c.lateLowerBlock = rewriteBlockRISCV64latelower
		c.lateLowerValue = rewriteValueRISCV64latelower
		c.registers = registersRISCV64[:]
		c.gpRegMask = gpRegMaskRISCV64
		c.fpRegMask = fpRegMaskRISCV64
		c.intParamRegs = paramIntRegRISCV64
		c.floatParamRegs = paramFloatRegRISCV64
		c.FPReg = framepointerRegRISCV64
		c.hasGReg = true
	case "wasm":
		c.PtrSize = 8
		c.RegSize = 8
		c.lowerBlock = rewriteBlockWasm
		c.lowerValue = rewriteValueWasm
		c.registers = registersWasm[:]
		c.gpRegMask = gpRegMaskWasm
		c.fpRegMask = fpRegMaskWasm
		c.fp32RegMask = fp32RegMaskWasm
		c.fp64RegMask = fp64RegMaskWasm
		c.FPReg = framepointerRegWasm
		c.LinkReg = linkRegWasm
		c.hasGReg = true
		c.noDuffDevice = true
		c.useAvg = false
		c.useHmul = false
	default:
		ctxt.Diag("arch %s not implemented", arch)
	}
	c.ctxt = ctxt
	c.optimize = optimize
	c.UseFMA = true
	c.SoftFloat = softfloat
	if softfloat {
		c.floatParamRegs = nil // no FP registers in softfloat mode
	}

	c.ABI0 = abi.NewABIConfig(0, 0, ctxt.Arch.FixedFrameSize, 0)
	c.ABI1 = abi.NewABIConfig(len(c.intParamRegs), len(c.floatParamRegs), ctxt.Arch.FixedFrameSize, 1)

	// On Plan 9, floating point operations are not allowed in note handler.
	if buildcfg.GOOS == "plan9" {
		// Don't use FMA on Plan 9
		c.UseFMA = false
	}

	if ctxt.Flag_shared {
		// LoweredWB is secretly a CALL and CALLs on 386 in
		// shared mode get rewritten by obj6.go to go through
		// the GOT, which clobbers BX.
		opcodeTable[Op386LoweredWB].reg.clobbers |= 1 << 3 // BX
	}

	return c
}

func (c *Config) Ctxt() *obj.Link { return c.ctxt }

func (c *Config) haveByteSwap(size int64) bool {
	switch size {
	case 8:
		return c.haveBswap64
	case 4:
		return c.haveBswap32
	case 2:
		return c.haveBswap16
	default:
		base.Fatalf("bad size %d\n", size)
		return false
	}
}
