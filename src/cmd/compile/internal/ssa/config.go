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
	SoftFloat      bool      //
	Race           bool      // race detector enabled
	BigEndian      bool      //
	unalignedOK    bool      // Unaligned loads/stores are ok
	haveBswap64    bool      // architecture implements Bswap64
	haveBswap32    bool      // architecture implements Bswap32
	haveBswap16    bool      // architecture implements Bswap16
	haveCondSelect bool      // architecture implements CondSelect

	// mulRecipes[x] = function to build v * x from v.
	mulRecipes map[int64]mulRecipe
}

type mulRecipe struct {
	cost  int
	build func(*Value, *Value) *Value // build(m, v) returns v * x built at m.
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
	Logf(string, ...any)

	// Log reports whether logging is not a no-op
	// some logging calls account for more than a few heap allocations.
	Log() bool

	// Fatalf reports a compiler error and exits.
	Fatalf(pos src.XPos, msg string, args ...any)

	// Warnl writes compiler messages in the form expected by "errorcheck" tests
	Warnl(pos src.XPos, fmt_ string, args ...any)

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
		c.haveCondSelect = true
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
		c.haveCondSelect = true
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
		c.haveCondSelect = true
	case "mips64":
		c.BigEndian = true
		fallthrough
	case "mips64le":
		c.PtrSize = 8
		c.RegSize = 8
		c.lowerBlock = rewriteBlockMIPS64
		c.lowerValue = rewriteValueMIPS64
		c.lateLowerBlock = rewriteBlockMIPS64latelower
		c.lateLowerValue = rewriteValueMIPS64latelower
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
		c.lateLowerBlock = rewriteBlockLOONG64latelower
		c.lateLowerValue = rewriteValueLOONG64latelower
		c.registers = registersLOONG64[:]
		c.gpRegMask = gpRegMaskLOONG64
		c.fpRegMask = fpRegMaskLOONG64
		c.intParamRegs = paramIntRegLOONG64
		c.floatParamRegs = paramFloatRegLOONG64
		c.FPReg = framepointerRegLOONG64
		c.LinkReg = linkRegLOONG64
		c.hasGReg = true
		c.unalignedOK = true
		c.haveCondSelect = true
	case "s390x":
		c.PtrSize = 8
		c.RegSize = 8
		c.lowerBlock = rewriteBlockS390X
		c.lowerValue = rewriteValueS390X
		c.registers = registersS390X[:]
		c.gpRegMask = gpRegMaskS390X
		c.fpRegMask = fpRegMaskS390X
		//c.intParamRegs = paramIntRegS390X
		//c.floatParamRegs = paramFloatRegS390X
		c.FPReg = framepointerRegS390X
		c.LinkReg = linkRegS390X
		c.hasGReg = true
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
		c.unalignedOK = true
		c.haveCondSelect = true
	default:
		ctxt.Diag("arch %s not implemented", arch)
	}
	c.ctxt = ctxt
	c.optimize = optimize
	c.SoftFloat = softfloat
	if softfloat {
		c.floatParamRegs = nil // no FP registers in softfloat mode
	}

	c.ABI0 = abi.NewABIConfig(0, 0, ctxt.Arch.FixedFrameSize, 0)
	c.ABI1 = abi.NewABIConfig(len(c.intParamRegs), len(c.floatParamRegs), ctxt.Arch.FixedFrameSize, 1)

	if ctxt.Flag_shared {
		// LoweredWB is secretly a CALL and CALLs on 386 in
		// shared mode get rewritten by obj6.go to go through
		// the GOT, which clobbers BX.
		opcodeTable[Op386LoweredWB].reg.clobbers |= 1 << 3 // BX
	}

	c.buildRecipes(arch)

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

func (c *Config) buildRecipes(arch string) {
	// Information for strength-reducing multiplies.
	type linearCombo struct {
		// we can compute a*x+b*y in one instruction
		a, b int64
		// cost, in arbitrary units (tenths of cycles, usually)
		cost int
		// builds SSA value for a*x+b*y. Use the position
		// information from m.
		build func(m, x, y *Value) *Value
	}

	// List all the linear combination instructions we have.
	var linearCombos []linearCombo
	r := func(a, b int64, cost int, build func(m, x, y *Value) *Value) {
		linearCombos = append(linearCombos, linearCombo{a: a, b: b, cost: cost, build: build})
	}
	var mulCost int
	switch arch {
	case "amd64":
		// Assumes that the following costs from https://gmplib.org/~tege/x86-timing.pdf:
		//    1 - addq, shlq, leaq, negq, subq
		//    3 - imulq
		// These costs limit the rewrites to two instructions.
		// Operations which have to happen in place (and thus
		// may require a reg-reg move) score slightly higher.
		mulCost = 30
		// add
		r(1, 1, 10,
			func(m, x, y *Value) *Value {
				v := m.Block.NewValue2(m.Pos, OpAMD64ADDQ, m.Type, x, y)
				if m.Type.Size() == 4 {
					v.Op = OpAMD64ADDL
				}
				return v
			})
		// neg
		r(-1, 0, 11,
			func(m, x, y *Value) *Value {
				v := m.Block.NewValue1(m.Pos, OpAMD64NEGQ, m.Type, x)
				if m.Type.Size() == 4 {
					v.Op = OpAMD64NEGL
				}
				return v
			})
		// sub
		r(1, -1, 11,
			func(m, x, y *Value) *Value {
				v := m.Block.NewValue2(m.Pos, OpAMD64SUBQ, m.Type, x, y)
				if m.Type.Size() == 4 {
					v.Op = OpAMD64SUBL
				}
				return v
			})
		// lea
		r(1, 2, 10,
			func(m, x, y *Value) *Value {
				v := m.Block.NewValue2(m.Pos, OpAMD64LEAQ2, m.Type, x, y)
				if m.Type.Size() == 4 {
					v.Op = OpAMD64LEAL2
				}
				return v
			})
		r(1, 4, 10,
			func(m, x, y *Value) *Value {
				v := m.Block.NewValue2(m.Pos, OpAMD64LEAQ4, m.Type, x, y)
				if m.Type.Size() == 4 {
					v.Op = OpAMD64LEAL4
				}
				return v
			})
		r(1, 8, 10,
			func(m, x, y *Value) *Value {
				v := m.Block.NewValue2(m.Pos, OpAMD64LEAQ8, m.Type, x, y)
				if m.Type.Size() == 4 {
					v.Op = OpAMD64LEAL8
				}
				return v
			})
		// regular shifts
		for i := 2; i < 64; i++ {
			r(1<<i, 0, 11,
				func(m, x, y *Value) *Value {
					v := m.Block.NewValue1I(m.Pos, OpAMD64SHLQconst, m.Type, int64(i), x)
					if m.Type.Size() == 4 {
						v.Op = OpAMD64SHLLconst
					}
					return v
				})
		}

	case "arm64":
		// Rationale (for M2 ultra):
		// - multiply is 3 cycles.
		// - add/neg/sub/shift are 1 cycle.
		// - add/neg/sub+shiftLL are 2 cycles.
		// We break ties against the multiply because using a
		// multiply also needs to load the constant into a register.
		// (It's 3 cycles and 2 instructions either way, but the
		// linear combo one might use 1 less register.)
		// The multiply constant might get lifted out of a loop though. Hmm....
		// Other arm64 chips have different tradeoffs.
		// Some chip's add+shift instructions are 1 cycle for shifts up to 4
		// and 2 cycles for shifts bigger than 4. So weight the larger shifts
		// a bit more.
		// TODO: figure out a happy medium.
		mulCost = 35
		// add
		r(1, 1, 10,
			func(m, x, y *Value) *Value {
				return m.Block.NewValue2(m.Pos, OpARM64ADD, m.Type, x, y)
			})
		// neg
		r(-1, 0, 10,
			func(m, x, y *Value) *Value {
				return m.Block.NewValue1(m.Pos, OpARM64NEG, m.Type, x)
			})
		// sub
		r(1, -1, 10,
			func(m, x, y *Value) *Value {
				return m.Block.NewValue2(m.Pos, OpARM64SUB, m.Type, x, y)
			})
		// regular shifts
		for i := 1; i < 64; i++ {
			c := 10
			if i == 1 {
				// Prefer x<<1 over x+x.
				// Note that we eventually reverse this decision in ARM64latelower.rules,
				// but this makes shift combining rules in ARM64.rules simpler.
				c--
			}
			r(1<<i, 0, c,
				func(m, x, y *Value) *Value {
					return m.Block.NewValue1I(m.Pos, OpARM64SLLconst, m.Type, int64(i), x)
				})
		}
		// ADDshiftLL
		for i := 1; i < 64; i++ {
			c := 20
			if i > 4 {
				c++
			}
			r(1, 1<<i, c,
				func(m, x, y *Value) *Value {
					return m.Block.NewValue2I(m.Pos, OpARM64ADDshiftLL, m.Type, int64(i), x, y)
				})
		}
		// NEGshiftLL
		for i := 1; i < 64; i++ {
			c := 20
			if i > 4 {
				c++
			}
			r(-1<<i, 0, c,
				func(m, x, y *Value) *Value {
					return m.Block.NewValue1I(m.Pos, OpARM64NEGshiftLL, m.Type, int64(i), x)
				})
		}
		// SUBshiftLL
		for i := 1; i < 64; i++ {
			c := 20
			if i > 4 {
				c++
			}
			r(1, -1<<i, c,
				func(m, x, y *Value) *Value {
					return m.Block.NewValue2I(m.Pos, OpARM64SUBshiftLL, m.Type, int64(i), x, y)
				})
		}
	case "loong64":
		// - multiply is 4 cycles.
		// - add/sub/shift/alsl are 1 cycle.
		// On loong64, using a multiply also needs to load the constant into a register.
		// TODO: figure out a happy medium.
		mulCost = 45

		// add
		r(1, 1, 10,
			func(m, x, y *Value) *Value {
				return m.Block.NewValue2(m.Pos, OpLOONG64ADDV, m.Type, x, y)
			})
		// neg
		r(-1, 0, 10,
			func(m, x, y *Value) *Value {
				return m.Block.NewValue1(m.Pos, OpLOONG64NEGV, m.Type, x)
			})
		// sub
		r(1, -1, 10,
			func(m, x, y *Value) *Value {
				return m.Block.NewValue2(m.Pos, OpLOONG64SUBV, m.Type, x, y)
			})

		// regular shifts
		for i := 1; i < 64; i++ {
			c := 10
			if i == 1 {
				// Prefer x<<1 over x+x.
				// Note that we eventually reverse this decision in LOONG64latelower.rules,
				// but this makes shift combining rules in LOONG64.rules simpler.
				c--
			}
			r(1<<i, 0, c,
				func(m, x, y *Value) *Value {
					return m.Block.NewValue1I(m.Pos, OpLOONG64SLLVconst, m.Type, int64(i), x)
				})
		}

		// ADDshiftLLV
		for i := 1; i < 5; i++ {
			c := 10
			r(1, 1<<i, c,
				func(m, x, y *Value) *Value {
					return m.Block.NewValue2I(m.Pos, OpLOONG64ADDshiftLLV, m.Type, int64(i), x, y)
				})
		}
	}

	c.mulRecipes = map[int64]mulRecipe{}

	// Single-instruction recipes.
	// The only option for the input value(s) is v.
	for _, combo := range linearCombos {
		x := combo.a + combo.b
		cost := combo.cost
		old := c.mulRecipes[x]
		if (old.build == nil || cost < old.cost) && cost < mulCost {
			c.mulRecipes[x] = mulRecipe{cost: cost, build: func(m, v *Value) *Value {
				return combo.build(m, v, v)
			}}
		}
	}
	// Two-instruction recipes.
	// A: Both of the outer's inputs are from the same single-instruction recipe.
	// B: First input is v and the second is from a single-instruction recipe.
	// C: Second input is v and the first is from a single-instruction recipe.
	// A is slightly preferred because it often needs 1 less register, so it
	// goes first.

	// A
	for _, inner := range linearCombos {
		for _, outer := range linearCombos {
			x := (inner.a + inner.b) * (outer.a + outer.b)
			cost := inner.cost + outer.cost
			old := c.mulRecipes[x]
			if (old.build == nil || cost < old.cost) && cost < mulCost {
				c.mulRecipes[x] = mulRecipe{cost: cost, build: func(m, v *Value) *Value {
					v = inner.build(m, v, v)
					return outer.build(m, v, v)
				}}
			}
		}
	}

	// B
	for _, inner := range linearCombos {
		for _, outer := range linearCombos {
			x := outer.a + outer.b*(inner.a+inner.b)
			cost := inner.cost + outer.cost
			old := c.mulRecipes[x]
			if (old.build == nil || cost < old.cost) && cost < mulCost {
				c.mulRecipes[x] = mulRecipe{cost: cost, build: func(m, v *Value) *Value {
					return outer.build(m, v, inner.build(m, v, v))
				}}
			}
		}
	}

	// C
	for _, inner := range linearCombos {
		for _, outer := range linearCombos {
			x := outer.a*(inner.a+inner.b) + outer.b
			cost := inner.cost + outer.cost
			old := c.mulRecipes[x]
			if (old.build == nil || cost < old.cost) && cost < mulCost {
				c.mulRecipes[x] = mulRecipe{cost: cost, build: func(m, v *Value) *Value {
					return outer.build(m, inner.build(m, v, v), v)
				}}
			}
		}
	}

	// Currently we only process 3 linear combination instructions for loong64.
	if arch == "loong64" {
		// Three-instruction recipes.
		// D: The first and the second are all single-instruction recipes, and they are also the third's inputs.
		// E: The first single-instruction is the second's input, and the second is the third's input.

		// D
		for _, first := range linearCombos {
			for _, second := range linearCombos {
				for _, third := range linearCombos {
					x := third.a*(first.a+first.b) + third.b*(second.a+second.b)
					cost := first.cost + second.cost + third.cost
					old := c.mulRecipes[x]
					if (old.build == nil || cost < old.cost) && cost < mulCost {
						c.mulRecipes[x] = mulRecipe{cost: cost, build: func(m, v *Value) *Value {
							v1 := first.build(m, v, v)
							v2 := second.build(m, v, v)
							return third.build(m, v1, v2)
						}}
					}
				}
			}
		}

		// E
		for _, first := range linearCombos {
			for _, second := range linearCombos {
				for _, third := range linearCombos {
					x := third.a*(second.a*(first.a+first.b)+second.b) + third.b
					cost := first.cost + second.cost + third.cost
					old := c.mulRecipes[x]
					if (old.build == nil || cost < old.cost) && cost < mulCost {
						c.mulRecipes[x] = mulRecipe{cost: cost, build: func(m, v *Value) *Value {
							v1 := first.build(m, v, v)
							v2 := second.build(m, v1, v)
							return third.build(m, v2, v)
						}}
					}
				}
			}
		}
	}

	// These cases should be handled specially by rewrite rules.
	// (Otherwise v * 1 == (neg (neg v)))
	delete(c.mulRecipes, 0)
	delete(c.mulRecipes, 1)

	// Currently:
	// len(c.mulRecipes) == 5984 on arm64
	//                       680 on amd64
	//                      9738 on loong64
	// This function takes ~2.5ms on arm64.
	//println(len(c.mulRecipes))
}
