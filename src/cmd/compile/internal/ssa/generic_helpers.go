// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import (
	"fmt"
	"math"
	"math/bits"
	"strings"

	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/reflectdata"
	"cmd/compile/internal/rttype"
	"cmd/compile/internal/typecheck"
	"cmd/compile/internal/types"
	"cmd/internal/obj"
	"cmd/internal/objabi"
)

func addToSub(op Op) Op {
	switch op {
	case OpAdd64:
		return OpSub64
	case OpAdd32:
		return OpSub32
	case OpAdd16:
		return OpSub16
	case OpAdd8:
		return OpSub8
	default:
		panic(fmt.Sprintf("unexpected op %v", op))
	}
}

func bitsAdd64(x, y, carry int64) (r struct{ sum, carry int64 }) {
	s, c := bits.Add64(uint64(x), uint64(y), uint64(carry))
	r.sum, r.carry = int64(s), int64(c)
	return
}

func bitsMulU32(x, y int32) (r struct{ hi, lo int32 }) {
	hi, lo := bits.Mul32(uint32(x), uint32(y))
	r.hi, r.lo = int32(hi), int32(lo)
	return
}

func bitsMulU64(x, y int64) (r struct{ hi, lo int64 }) {
	hi, lo := bits.Mul64(uint64(x), uint64(y))
	r.hi, r.lo = int64(hi), int64(lo)
	return
}

func bitsDiv128u(hi, lo, y int64) (r struct{ quo, rem int64 }) {
	q, rem := bits.Div64(uint64(hi), uint64(lo), uint64(y))
	r.quo, r.rem = int64(q), int64(rem)
	return
}

// bool2int converts bool to int: true to 1, false to 0
func bool2int(x bool) int {
	var b int
	if x {
		b = 1
	}
	return b
}

// canLoadUnaligned reports if the architecture supports unaligned load operations.
func canLoadUnaligned(c *Config) bool {
	return c.Ctxt.Arch.Alignment == 1
}

// canRotate reports whether the architecture supports
// rotates of integer registers with the given number of bits.
func canRotate(c *Config, bits int64) bool {
	if bits > c.PtrSize*8 {
		// Don't rewrite to rotates bigger than the machine word.
		return false
	}
	switch c.Arch {
	case "386", "amd64", "arm64", "loong64", "riscv64":
		return true
	case "arm", "s390x", "ppc64", "ppc64le", "wasm":
		return bits >= 32
	default:
		return false
	}
}

func copyCompatibleType(t1, t2 *types.Type) bool {
	if t1.Size() != t2.Size() {
		return false
	}
	if t1.IsInteger() {
		return t2.IsInteger()
	}
	if IsPtr(t1) {
		return IsPtr(t2)
	}
	return t1.Compare(t2) == types.CMPeq
}

func devirtLECall(v *Value, sym *obj.LSym) *Value {
	v.Op = OpStaticLECall
	auxcall := v.Aux.(*AuxCall)
	auxcall.Fn = sym
	// Remove first arg
	v.Args[0].Uses--
	copy(v.Args[0:], v.Args[1:])
	v.Args[len(v.Args)-1] = nil // aid GC
	v.Args = v.Args[:len(v.Args)-1]
	if f := v.Block.Func; f.Pass.Debug > 0 {
		f.Warnl(v.Pos, "de-virtualizing call")
	}
	return v
}

// hasSmallRotate reports whether the architecture has rotate instructions
// for sizes < 32-bit.  This is used to decide whether to promote some rotations.
func hasSmallRotate(c *Config) bool {
	switch c.Arch {
	case "amd64", "386":
		return true
	default:
		return false
	}
}

func invertibleBool(op Op) bool {
	switch op {
	case OpLess64, OpLess32, OpLess16, OpLess8,
		OpLeq64, OpLeq32, OpLeq16, OpLeq8,
		OpLess64U, OpLess32U, OpLess16U, OpLess8U,
		OpLeq64U, OpLeq32U, OpLeq16U, OpLeq8U,
		OpEq64, OpEq32, OpEq16, OpEq8,
		OpNeq64, OpNeq32, OpNeq16, OpNeq8,
		OpNot:
		return true
	default:
		return false
	}
}

func isConstZero(v *Value) bool {
	switch v.Op {
	case OpConstNil:
		return true
	case OpConst64, OpConst32, OpConst16, OpConst8, OpConstBool, OpConst32F, OpConst64F:
		return v.AuxInt == 0
	case OpStringMake, OpIMake, OpComplexMake:
		return isConstZero(v.Args[0]) && isConstZero(v.Args[1])
	case OpSliceMake:
		return isConstZero(v.Args[0]) && isConstZero(v.Args[1]) && isConstZero(v.Args[2])
	case OpStringPtr, OpStringLen, OpSlicePtr, OpSliceLen, OpSliceCap, OpITab, OpIData, OpComplexReal, OpComplexImag:
		return isConstZero(v.Args[0])
	}
	return false
}

func isDictArgSym(sym Sym) bool {
	return sym.(*ir.Name).Sym().Name == typecheck.LocalDictName
}

// isDirectAndComparableIface reports whether v represents an itab
// (a *runtime._itab) for a type whose value is stored directly
// in an interface (i.e., is pointer or pointer-like) and is comparable.
func isDirectAndComparableIface(v *Value) bool {
	return isDirectAndComparableIface1(v, 9)
}

// v is an itab
func isDirectAndComparableIface1(v *Value, depth int) bool {
	if depth == 0 {
		return false
	}
	switch v.Op {
	case OpITab:
		return isDirectAndComparableIface2(v.Args[0], depth-1)
	case OpAddr:
		lsym := v.Aux.(*obj.LSym)
		if ii := lsym.ItabInfo(); ii != nil {
			t := ii.Type.(*types.Type)
			return types.IsDirectIface(t) && types.IsComparable(t)
		}
	case OpConstNil:
		// We can treat this as direct, because if the itab is
		// nil, the data field must be nil also.
		return true
	}
	return false
}

// v is an interface
func isDirectAndComparableIface2(v *Value, depth int) bool {
	if depth == 0 {
		return false
	}
	switch v.Op {
	case OpIMake:
		return isDirectAndComparableIface1(v.Args[0], depth-1)
	case OpPhi:
		for _, a := range v.Args {
			if !isDirectAndComparableIface2(a, depth-1) {
				return false
			}
		}
		return true
	}
	return false
}

// isDirectAndComparableType reports whether v represents a type
// (a *runtime._type) whose value is stored directly in an
// interface (i.e., is pointer or pointer-like) and is comparable.
func isDirectAndComparableType(v *Value) bool {
	return isDirectAndComparableType1(v)
}

// v is a type
func isDirectAndComparableType1(v *Value) bool {
	switch v.Op {
	case OpITab:
		return isDirectAndComparableType2(v.Args[0])
	case OpAddr:
		lsym := v.Aux.(*obj.LSym)
		if ti := lsym.TypeInfo(); ti != nil {
			t := ti.Type.(*types.Type)
			return types.IsDirectIface(t) && types.IsComparable(t)
		}
	}
	return false
}

// v is an empty interface
func isDirectAndComparableType2(v *Value) bool {
	switch v.Op {
	case OpIMake:
		return isDirectAndComparableType1(v.Args[0])
	}
	return false
}

// isFixedLoad returns true if the load can be resolved to fixed address or constant,
// and can be rewritten by rewriteFixedLoad.
func isFixedLoad(v *Value, sym Sym, off int64) bool {
	lsym := sym.(*obj.LSym)
	if (v.Type.IsPtrShaped() || v.Type.IsUintptr()) && lsym.Type == objabi.SRODATA {
		for _, r := range lsym.R {
			if (r.Type == objabi.R_ADDR || r.Type == objabi.R_WEAKADDR) && int64(r.Off) == off && r.Add == 0 {
				return true
			}
		}
		return false
	}

	if ti := lsym.TypeInfo(); ti != nil {
		// Type symbols do not contain information about their fields, unlike the cases above.
		// Hand-implement field accesses.
		// TODO: can this be replaced with reflectdata.writeType and just use the code above?

		t := ti.Type.(*types.Type)

		for _, f := range rttype.Type.Fields() {
			if f.Offset == off && copyCompatibleType(v.Type, f.Type) {
				switch f.Sym.Name {
				case "Size_", "PtrBytes", "Hash", "Kind_", "GCData", "TFlag":
					return true
				default:
					// fmt.Println("unknown field", f.Sym.Name)
					return false
				}
			}
		}

		if t.IsPtr() && off == rttype.PtrType.OffsetOf("Elem") {
			return true
		}

		return false
	}

	return false
}

func isInlinableMemclr(c *Config, sz int64) bool {
	if sz < 0 {
		return false
	}
	// TODO: expand this check to allow other architectures
	// see CL 454255 and issue 56997
	switch c.Arch {
	case "amd64", "arm64":
		return true
	case "ppc64le", "ppc64", "loong64":
		return sz < 512
	}
	return false
}

func isMalloc(aux Aux) bool {
	return IsNewObjectCall(aux) || IsSpecializedMalloc(aux)
}

// isNonNegative reports whether v is known to be greater or equal to zero.
// Note that this is pretty simplistic. The prove pass generates more detailed
// nonnegative information about values.
func isNonNegative(v *Value) bool {
	if !v.Type.IsInteger() {
		v.Fatalf("isNonNegative bad type: %v", v.Type)
	}
	// TODO: return true if !v.Type.IsSigned()
	// SSA isn't type-safe enough to do that now (issue 37753).
	// The checks below depend only on the pattern of bits.

	switch v.Op {
	case OpConst64:
		return v.AuxInt >= 0

	case OpConst32:
		return int32(v.AuxInt) >= 0

	case OpConst16:
		return int16(v.AuxInt) >= 0

	case OpConst8:
		return int8(v.AuxInt) >= 0

	case OpStringLen, OpSliceLen, OpSliceCap,
		OpZeroExt8to64, OpZeroExt16to64, OpZeroExt32to64,
		OpZeroExt8to32, OpZeroExt16to32, OpZeroExt8to16,
		OpCtz64, OpCtz32, OpCtz16, OpCtz8,
		OpCtz64NonZero, OpCtz32NonZero, OpCtz16NonZero, OpCtz8NonZero,
		OpBitLen64, OpBitLen32, OpBitLen16, OpBitLen8:
		return true

	case OpRsh64Ux64, OpRsh32Ux64:
		by := v.Args[1]
		return by.Op == OpConst64 && by.AuxInt > 0

	case OpRsh64x64, OpRsh32x64, OpRsh8x64, OpRsh16x64, OpRsh32x32, OpRsh64x32,
		OpSignExt32to64, OpSignExt16to64, OpSignExt8to64, OpSignExt16to32, OpSignExt8to32:
		return isNonNegative(v.Args[0])

	case OpAnd64, OpAnd32, OpAnd16, OpAnd8:
		return isNonNegative(v.Args[0]) || isNonNegative(v.Args[1])

	case OpMod64, OpMod32, OpMod16, OpMod8,
		OpDiv64, OpDiv32, OpDiv16, OpDiv8,
		OpOr64, OpOr32, OpOr16, OpOr8,
		OpXor64, OpXor32, OpXor16, OpXor8:
		return isNonNegative(v.Args[0]) && isNonNegative(v.Args[1])

		// We could handle OpPhi here, but the improvements from doing
		// so are very minor, and it is neither simple nor cheap.
	}
	return false
}

func isStackPtr(v *Value) bool {
	for v.Op == OpOffPtr || v.Op == OpAddPtr {
		v = v.Args[0]
	}
	return v.Op == OpSP || v.Op == OpLocalAddr
}

// needRaceCleanup reports whether this call to racefuncenter/exit isn't needed.
func needRaceCleanup(sym *AuxCall, v *Value) bool {
	f := v.Block.Func
	if !f.Config.Race {
		return false
	}
	if !IsSameCall(sym, "runtime.racefuncenter") && !IsSameCall(sym, "runtime.racefuncexit") {
		return false
	}
	for _, b := range f.Blocks {
		for _, v := range b.Values {
			switch v.Op {
			case OpStaticCall, OpStaticLECall:
				// Check for racefuncenter will encounter racefuncexit and vice versa.
				// Allow calls to panic*
				s := v.Aux.(*AuxCall).Fn.String()
				switch s {
				case "runtime.racefuncenter", "runtime.racefuncexit",
					"runtime.panicdivide", "runtime.panicwrap",
					"runtime.panicshift":
					continue
				}
				// If we encountered any call, we need to keep racefunc*,
				// for accurate stacktraces.
				return false
			case OpPanicBounds, OpPanicExtend:
				// Note: these are panic generators that are ok (like the static calls above).
			case OpClosureCall, OpInterCall, OpClosureLECall, OpInterLECall:
				// We must keep the race functions if there are any other call types.
				return false
			}
		}
	}
	if IsSameCall(sym, "runtime.racefuncenter") {
		// TODO REGISTER ABI this needs to be cleaned up.
		// If we're removing racefuncenter, remove its argument as well.
		if v.Args[0].Op != OpStore {
			if v.Op == OpStaticLECall {
				// there is no store, yet.
				return true
			}
			return false
		}
		mem := v.Args[0].Args[2]
		v.Args[0].Reset(OpCopy)
		v.Args[0].AddArg(mem)
	}
	return true
}

func nlz16(x int16) int { return bits.LeadingZeros16(uint16(x)) }

func nlz32(x int32) int { return bits.LeadingZeros32(uint32(x)) }

// nlzX returns the number of leading zeros.
func nlz64(x int64) int { return bits.LeadingZeros64(uint64(x)) }

func nlz8(x int8) int { return bits.LeadingZeros8(uint8(x)) }

func ntz16(x int16) int { return bits.TrailingZeros16(uint16(x)) }

func ntz32(x int32) int { return bits.TrailingZeros32(uint32(x)) }

func ntz8(x int8) int { return bits.TrailingZeros8(uint8(x)) }

// reciprocalExact32 reports whether 1/c is exactly representable.
func reciprocalExact32(c float32) bool {
	b := math.Float32bits(c)
	man := b & (1<<23 - 1)
	if man != 0 {
		return false // not a power of 2, denormal, or NaN
	}
	exp := b >> 23 & (1<<8 - 1)
	// exponent bias is 0x7f.  So taking the reciprocal of a number
	// changes the exponent to 0xfe-exp.
	switch exp {
	case 0:
		return false // ±0
	case 0xff:
		return false // ±inf
	case 0xfe:
		return false // exponent is not representable
	default:
		return true
	}
}

// reciprocalExact64 reports whether 1/c is exactly representable.
func reciprocalExact64(c float64) bool {
	b := math.Float64bits(c)
	man := b & (1<<52 - 1)
	if man != 0 {
		return false // not a power of 2, denormal, or NaN
	}
	exp := b >> 52 & (1<<11 - 1)
	// exponent bias is 0x3ff.  So taking the reciprocal of a number
	// changes the exponent to 0x7fe-exp.
	switch exp {
	case 0:
		return false // ±0
	case 0x7ff:
		return false // ±inf
	case 0x7fe:
		return false // exponent is not representable
	default:
		return true
	}
}

// registerizable reports whether t is a primitive type that fits in
// a register. It assumes float64 values will always fit into registers
// even if that isn't strictly true.
func registerizable(b *Block, typ *types.Type) bool {
	if typ.IsPtrShaped() || typ.IsFloat() || typ.IsBoolean() {
		return true
	}
	if typ.IsInteger() {
		return typ.Size() <= b.Func.Config.RegSize
	}
	return false
}

// resetCopy resets v to be a copy of arg.
// Always returns true.
func resetCopy(v *Value, arg *Value) bool {
	v.Reset(OpCopy)
	v.AddArg(arg)
	return true
}

// rewriteCondSelectIntoMath reports whether x OP (y * constant) should be used instead of a CondSelect.
// x arbitrary, y in [0,1]
func rewriteCondSelectIntoMath(config *Config, op Op, constant int64) bool {
	switch config.Arch {
	case "amd64":
		// constant=1 becomes zext, add 2/4/8 becomes lea, rest becomes shl.
		// shl has asymmetric latency (1:3 vs 2:2) but performs better in accumulation chains.
		return IsPowerOfTwo(uint64(constant))
	case "arm64":
		switch op {
		case OpAdd64, OpAdd32, OpAdd16, OpAdd8:
			if constant == 1 {
				return false // better done as CSINC
			}
			fallthrough
		case OpSub64, OpSub32, OpSub16, OpSub8,
			OpAnd64, OpAnd32, OpAnd16, OpAnd8,
			OpOr64, OpOr32, OpOr16, OpOr8,
			OpXor64, OpXor32, OpXor16, OpXor8:
			// Implemented using an inline LSL
			return IsPowerOfTwo(uint64(constant))
		default:
			if constant == 1 {
				return true
			}
		}
	default:
		// TODO: fine tune for other architectures.
		return constant == 1
	}
	return false
}

// rewriteFixedLoad rewrites a load to a fixed address or constant, if isFixedLoad returns true.
func rewriteFixedLoad(v *Value, sym Sym, sb *Value, off int64) *Value {
	b := v.Block
	f := b.Func

	lsym := sym.(*obj.LSym)
	if (v.Type.IsPtrShaped() || v.Type.IsUintptr()) && lsym.Type == objabi.SRODATA {
		for _, r := range lsym.R {
			if (r.Type == objabi.R_ADDR || r.Type == objabi.R_WEAKADDR) && int64(r.Off) == off && r.Add == 0 {
				if strings.HasPrefix(r.Sym.Name, "type:") {
					// In case we're loading a type out of a dictionary, we need to record
					// that the containing function might put that type in an interface.
					// That information is currently recorded in relocations in the dictionary,
					// but if we perform this load at compile time then the dictionary
					// might be dead.
					reflectdata.MarkTypeSymUsedInInterface(r.Sym, f.Fe.Func().Linksym())
				} else if strings.HasPrefix(r.Sym.Name, "go:itab") {
					// Same, but if we're using an itab we need to record that the
					// itab._type might be put in an interface.
					reflectdata.MarkTypeSymUsedInInterface(r.Sym, f.Fe.Func().Linksym())
				}
				v.Reset(OpAddr)
				v.Aux = SymToAux(r.Sym)
				v.AddArg(sb)
				return v
			}
		}
		base.Fatalf("fixedLoad data not known for %s:%d", sym, off)
	}

	if ti := lsym.TypeInfo(); ti != nil {
		// Type symbols do not contain information about their fields, unlike the cases above.
		// Hand-implement field accesses.
		// TODO: can this be replaced with reflectdata.writeType and just use the code above?

		t := ti.Type.(*types.Type)

		ptrSizedOpConst := OpConst64
		if f.Config.PtrSize == 4 {
			ptrSizedOpConst = OpConst32
		}

		for _, f := range rttype.Type.Fields() {
			if f.Offset == off && copyCompatibleType(v.Type, f.Type) {
				switch f.Sym.Name {
				case "Size_":
					v.Reset(ptrSizedOpConst)
					v.AuxInt = t.Size()
					return v
				case "PtrBytes":
					v.Reset(ptrSizedOpConst)
					v.AuxInt = types.PtrDataSize(t)
					return v
				case "Hash":
					v.Reset(OpConst32)
					v.AuxInt = int64(int32(types.TypeHash(t)))
					return v
				case "TFlag":
					v.Reset(OpConst8)
					v.AuxInt = int64(t.TFlag())
					return v
				case "Kind_":
					v.Reset(OpConst8)
					v.AuxInt = int64(int8(reflectdata.ABIKindOfType(t)))
					return v
				case "GCData":
					gcdata, _ := reflectdata.GCSym(t, true)
					v.Reset(OpAddr)
					v.Aux = SymToAux(gcdata)
					v.AddArg(sb)
					return v
				default:
					base.Fatalf("unknown field %s for fixedLoad of %s at offset %d", f.Sym.Name, lsym.Name, off)
				}
			}
		}

		if t.IsPtr() && off == rttype.PtrType.OffsetOf("Elem") {
			elemSym := reflectdata.TypeLinksym(t.Elem())
			reflectdata.MarkTypeSymUsedInInterface(elemSym, f.Fe.Func().Linksym())
			v.Reset(OpAddr)
			v.Aux = SymToAux(elemSym)
			v.AddArg(sb)
			return v
		}

		base.Fatalf("fixedLoad data not known for %s:%d", sym, off)
	}

	base.Fatalf("fixedLoad data not known for %s:%d", sym, off)
	return nil
}

func rewriteStructLoad(v *Value) *Value {
	b := v.Block
	ptr := v.Args[0]
	mem := v.Args[1]

	t := v.Type
	args := make([]*Value, t.NumFields())
	for i := range args {
		ft := t.FieldType(i)
		addr := b.NewValue1I(v.Pos, OpOffPtr, ft.PtrTo(), t.FieldOff(i), ptr)
		args[i] = b.NewValue2(v.Pos, OpLoad, ft, addr, mem)
	}

	v.Reset(OpStructMake)
	v.AddArgs(args...)
	return v
}

// symIsROZero reports whether sym is a read-only global whose data contains all zeros.
func symIsROZero(sym Sym) bool {
	lsym := sym.(*obj.LSym)
	if lsym.Type != objabi.SRODATA || len(lsym.R) != 0 {
		return false
	}
	for _, b := range lsym.P {
		if b != 0 {
			return false
		}
	}
	return true
}

// uaddOvf reports whether unsigned a+b would overflow.
func uaddOvf(a, b int64) bool {
	return uint64(a)+uint64(b) < uint64(a)
}

// warnRule generates compiler debug output with string s when
// v is not in autogenerated code, cond is true and the rule has fired.
func warnRule(cond bool, v *Value, s string) bool {
	if pos := v.Pos; pos.Line() > 1 && cond {
		v.Block.Func.Warnl(pos, s)
	}
	return true
}

func bitsSub64(x, y, borrow int64) (r struct{ diff, borrow int64 }) {
	d, b := bits.Sub64(uint64(x), uint64(y), uint64(borrow))
	r.diff, r.borrow = int64(d), int64(b)
	return
}
