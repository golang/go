// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rewritegeneric

import (
	"fmt"
	"math"
	"math/bits"
	"strings"

	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/reflectdata"
	"cmd/compile/internal/rttype"
	"cmd/compile/internal/ssa"
	"cmd/compile/internal/ssa/ssaop"
	"cmd/compile/internal/typecheck"
	"cmd/compile/internal/types"
	"cmd/internal/obj"
	"cmd/internal/objabi"
)

func addToSub(op ssaop.Op) ssaop.Op {
	switch op {
	case ssaop.OpAdd64:
		return ssaop.OpSub64
	case ssaop.OpAdd32:
		return ssaop.OpSub32
	case ssaop.OpAdd16:
		return ssaop.OpSub16
	case ssaop.OpAdd8:
		return ssaop.OpSub8
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
func canLoadUnaligned(c *ssa.Config) bool {
	return c.Ctxt.Arch.Alignment == 1
}

// canRotate reports whether the architecture supports
// rotates of integer registers with the given number of bits.
func canRotate(c *ssa.Config, bits int64) bool {
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
	if ssa.IsPtr(t1) {
		return ssa.IsPtr(t2)
	}
	return t1.Compare(t2) == types.CMPeq
}

func devirtLECall(v *ssa.Value, sym *obj.LSym) *ssa.Value {
	v.Op = ssaop.OpStaticLECall
	auxcall := v.Aux.(*ssa.AuxCall)
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
func hasSmallRotate(c *ssa.Config) bool {
	switch c.Arch {
	case "amd64", "386":
		return true
	default:
		return false
	}
}

func invertibleBool(op ssaop.Op) bool {
	switch op {
	case ssaop.OpLess64, ssaop.OpLess32, ssaop.OpLess16, ssaop.OpLess8,
		ssaop.OpLeq64, ssaop.OpLeq32, ssaop.OpLeq16, ssaop.OpLeq8,
		ssaop.OpLess64U, ssaop.OpLess32U, ssaop.OpLess16U, ssaop.OpLess8U,
		ssaop.OpLeq64U, ssaop.OpLeq32U, ssaop.OpLeq16U, ssaop.OpLeq8U,
		ssaop.OpEq64, ssaop.OpEq32, ssaop.OpEq16, ssaop.OpEq8,
		ssaop.OpNeq64, ssaop.OpNeq32, ssaop.OpNeq16, ssaop.OpNeq8,
		ssaop.OpNot:
		return true
	default:
		return false
	}
}

func isDictArgSym(sym ssa.Sym) bool {
	return sym.(*ir.Name).Sym().Name == typecheck.LocalDictName
}

// isDirectAndComparableIface reports whether v represents an itab
// (a *runtime._itab) for a type whose value is stored directly
// in an interface (i.e., is pointer or pointer-like) and is comparable.
func isDirectAndComparableIface(v *ssa.Value) bool {
	return isDirectAndComparableIface1(v, 9)
}

// v is an itab
func isDirectAndComparableIface1(v *ssa.Value, depth int) bool {
	if depth == 0 {
		return false
	}
	switch v.Op {
	case ssaop.OpITab:
		return isDirectAndComparableIface2(v.Args[0], depth-1)
	case ssaop.OpAddr:
		lsym := v.Aux.(*obj.LSym)
		if ii := lsym.ItabInfo(); ii != nil {
			t := ii.Type.(*types.Type)
			return types.IsDirectIface(t) && types.IsComparable(t)
		}
	case ssaop.OpConstNil:
		// We can treat this as direct, because if the itab is
		// nil, the data field must be nil also.
		return true
	}
	return false
}

// v is an interface
func isDirectAndComparableIface2(v *ssa.Value, depth int) bool {
	if depth == 0 {
		return false
	}
	switch v.Op {
	case ssaop.OpIMake:
		return isDirectAndComparableIface1(v.Args[0], depth-1)
	case ssaop.OpPhi:
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
func isDirectAndComparableType(v *ssa.Value) bool {
	return isDirectAndComparableType1(v)
}

// v is a type
func isDirectAndComparableType1(v *ssa.Value) bool {
	switch v.Op {
	case ssaop.OpITab:
		return isDirectAndComparableType2(v.Args[0])
	case ssaop.OpAddr:
		lsym := v.Aux.(*obj.LSym)
		if ti := lsym.TypeInfo(); ti != nil {
			t := ti.Type.(*types.Type)
			return types.IsDirectIface(t) && types.IsComparable(t)
		}
	}
	return false
}

// v is an empty interface
func isDirectAndComparableType2(v *ssa.Value) bool {
	switch v.Op {
	case ssaop.OpIMake:
		return isDirectAndComparableType1(v.Args[0])
	}
	return false
}

// isFixedLoad returns true if the load can be resolved to fixed address or constant,
// and can be rewritten by rewriteFixedLoad.
func isFixedLoad(v *ssa.Value, sym ssa.Sym, off int64) bool {
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

func isInlinableMemclr(c *ssa.Config, sz int64) bool {
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

func isMalloc(aux ssa.Aux) bool {
	return ssa.IsNewObjectCall(aux) || ssa.IsSpecializedMalloc(aux)
}

// isNonNegative reports whether v is known to be greater or equal to zero.
// Note that this is pretty simplistic. The prove pass generates more detailed
// nonnegative information about values.
func isNonNegative(v *ssa.Value) bool {
	if !v.Type.IsInteger() {
		v.Fatalf("isNonNegative bad type: %v", v.Type)
	}
	// TODO: return true if !v.Type.IsSigned()
	// SSA isn't type-safe enough to do that now (issue 37753).
	// The checks below depend only on the pattern of bits.

	switch v.Op {
	case ssaop.OpConst64:
		return v.AuxInt >= 0

	case ssaop.OpConst32:
		return int32(v.AuxInt) >= 0

	case ssaop.OpConst16:
		return int16(v.AuxInt) >= 0

	case ssaop.OpConst8:
		return int8(v.AuxInt) >= 0

	case ssaop.OpStringLen, ssaop.OpSliceLen, ssaop.OpSliceCap,
		ssaop.OpZeroExt8to64, ssaop.OpZeroExt16to64, ssaop.OpZeroExt32to64,
		ssaop.OpZeroExt8to32, ssaop.OpZeroExt16to32, ssaop.OpZeroExt8to16,
		ssaop.OpCtz64, ssaop.OpCtz32, ssaop.OpCtz16, ssaop.OpCtz8,
		ssaop.OpCtz64NonZero, ssaop.OpCtz32NonZero, ssaop.OpCtz16NonZero, ssaop.OpCtz8NonZero,
		ssaop.OpBitLen64, ssaop.OpBitLen32, ssaop.OpBitLen16, ssaop.OpBitLen8:
		return true

	case ssaop.OpRsh64Ux64, ssaop.OpRsh32Ux64:
		by := v.Args[1]
		return by.Op == ssaop.OpConst64 && by.AuxInt > 0

	case ssaop.OpRsh64x64, ssaop.OpRsh32x64, ssaop.OpRsh8x64, ssaop.OpRsh16x64, ssaop.OpRsh32x32, ssaop.OpRsh64x32,
		ssaop.OpSignExt32to64, ssaop.OpSignExt16to64, ssaop.OpSignExt8to64, ssaop.OpSignExt16to32, ssaop.OpSignExt8to32:
		return isNonNegative(v.Args[0])

	case ssaop.OpAnd64, ssaop.OpAnd32, ssaop.OpAnd16, ssaop.OpAnd8:
		return isNonNegative(v.Args[0]) || isNonNegative(v.Args[1])

	case ssaop.OpMod64, ssaop.OpMod32, ssaop.OpMod16, ssaop.OpMod8,
		ssaop.OpDiv64, ssaop.OpDiv32, ssaop.OpDiv16, ssaop.OpDiv8,
		ssaop.OpOr64, ssaop.OpOr32, ssaop.OpOr16, ssaop.OpOr8,
		ssaop.OpXor64, ssaop.OpXor32, ssaop.OpXor16, ssaop.OpXor8:
		return isNonNegative(v.Args[0]) && isNonNegative(v.Args[1])

		// We could handle OpPhi here, but the improvements from doing
		// so are very minor, and it is neither simple nor cheap.
	}
	return false
}

func isStackPtr(v *ssa.Value) bool {
	for v.Op == ssaop.OpOffPtr || v.Op == ssaop.OpAddPtr {
		v = v.Args[0]
	}
	return v.Op == ssaop.OpSP || v.Op == ssaop.OpLocalAddr
}

// needRaceCleanup reports whether this call to racefuncenter/exit isn't needed.
func needRaceCleanup(sym *ssa.AuxCall, v *ssa.Value) bool {
	f := v.Block.Func
	if !f.Config.Race {
		return false
	}
	if !ssa.IsSameCall(sym, "runtime.racefuncenter") && !ssa.IsSameCall(sym, "runtime.racefuncexit") {
		return false
	}
	for _, b := range f.Blocks {
		for _, v := range b.Values {
			switch v.Op {
			case ssaop.OpStaticCall, ssaop.OpStaticLECall:
				// Check for racefuncenter will encounter racefuncexit and vice versa.
				// Allow calls to panic*
				s := v.Aux.(*ssa.AuxCall).Fn.String()
				switch s {
				case "runtime.racefuncenter", "runtime.racefuncexit",
					"runtime.panicdivide", "runtime.panicwrap",
					"runtime.panicshift":
					continue
				}
				// If we encountered any call, we need to keep racefunc*,
				// for accurate stacktraces.
				return false
			case ssaop.OpPanicBounds, ssaop.OpPanicExtend:
				// Note: these are panic generators that are ok (like the static calls above).
			case ssaop.OpClosureCall, ssaop.OpInterCall, ssaop.OpClosureLECall, ssaop.OpInterLECall:
				// We must keep the race functions if there are any other call types.
				return false
			}
		}
	}
	if ssa.IsSameCall(sym, "runtime.racefuncenter") {
		// TODO REGISTER ABI this needs to be cleaned up.
		// If we're removing racefuncenter, remove its argument as well.
		if v.Args[0].Op != ssaop.OpStore {
			if v.Op == ssaop.OpStaticLECall {
				// there is no store, yet.
				return true
			}
			return false
		}
		mem := v.Args[0].Args[2]
		v.Args[0].Reset(ssaop.OpCopy)
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
func registerizable(b *ssa.Block, typ *types.Type) bool {
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
func resetCopy(v *ssa.Value, arg *ssa.Value) bool {
	v.Reset(ssaop.OpCopy)
	v.AddArg(arg)
	return true
}

// rewriteCondSelectIntoMath reports whether x OP (y * constant) should be used instead of a CondSelect.
// x arbitrary, y in [0,1]
func rewriteCondSelectIntoMath(config *ssa.Config, op ssaop.Op, constant int64) bool {
	switch config.Arch {
	case "amd64":
		// constant=1 becomes zext, add 2/4/8 becomes lea, rest becomes shl.
		// shl has asymmetric latency (1:3 vs 2:2) but performs better in accumulation chains.
		return ssa.IsPowerOfTwo(uint64(constant))
	case "arm64":
		switch op {
		case ssaop.OpAdd64, ssaop.OpAdd32, ssaop.OpAdd16, ssaop.OpAdd8:
			if constant == 1 {
				return false // better done as CSINC
			}
			fallthrough
		case ssaop.OpSub64, ssaop.OpSub32, ssaop.OpSub16, ssaop.OpSub8,
			ssaop.OpAnd64, ssaop.OpAnd32, ssaop.OpAnd16, ssaop.OpAnd8,
			ssaop.OpOr64, ssaop.OpOr32, ssaop.OpOr16, ssaop.OpOr8,
			ssaop.OpXor64, ssaop.OpXor32, ssaop.OpXor16, ssaop.OpXor8:
			// Implemented using an inline LSL
			return ssa.IsPowerOfTwo(uint64(constant))
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
func rewriteFixedLoad(v *ssa.Value, sym ssa.Sym, sb *ssa.Value, off int64) *ssa.Value {
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
				v.Reset(ssaop.OpAddr)
				v.Aux = ssa.SymToAux(r.Sym)
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

		ptrSizedOpConst := ssaop.OpConst64
		if f.Config.PtrSize == 4 {
			ptrSizedOpConst = ssaop.OpConst32
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
					v.Reset(ssaop.OpConst32)
					v.AuxInt = int64(int32(types.TypeHash(t)))
					return v
				case "TFlag":
					v.Reset(ssaop.OpConst8)
					v.AuxInt = int64(t.TFlag())
					return v
				case "Kind_":
					v.Reset(ssaop.OpConst8)
					v.AuxInt = int64(int8(reflectdata.ABIKindOfType(t)))
					return v
				case "GCData":
					gcdata, _ := reflectdata.GCSym(t, true)
					v.Reset(ssaop.OpAddr)
					v.Aux = ssa.SymToAux(gcdata)
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
			v.Reset(ssaop.OpAddr)
			v.Aux = ssa.SymToAux(elemSym)
			v.AddArg(sb)
			return v
		}

		base.Fatalf("fixedLoad data not known for %s:%d", sym, off)
	}

	base.Fatalf("fixedLoad data not known for %s:%d", sym, off)
	return nil
}

func rewriteStructLoad(v *ssa.Value) *ssa.Value {
	b := v.Block
	ptr := v.Args[0]
	mem := v.Args[1]

	t := v.Type
	args := make([]*ssa.Value, t.NumFields())
	for i := range args {
		ft := t.FieldType(i)
		addr := b.NewValue1I(v.Pos, ssaop.OpOffPtr, ft.PtrTo(), t.FieldOff(i), ptr)
		args[i] = b.NewValue2(v.Pos, ssaop.OpLoad, ft, addr, mem)
	}

	v.Reset(ssaop.OpStructMake)
	v.AddArgs(args...)
	return v
}

// symIsROZero reports whether sym is a read-only global whose data contains all zeros.
func symIsROZero(sym ssa.Sym) bool {
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
func warnRule(cond bool, v *ssa.Value, s string) bool {
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

func modularMultiplicativeInverse(x uint64) (y uint64) {
	if x%2 != 1 {
		panic("even numbers in a power-of-two modulus do not have a multiplicative inverse")
	}
	// we start with 3 bits of precision because each odd number is its own multiplicative inverse mod 8
	y = x // 3 bits

	// now use the Newton-Raphson method to double the number of correct bits in each iteration.
	y *= 2 - x*y // 6 bits
	y *= 2 - x*y // 12 bits
	y *= 2 - x*y // 24 bits
	y *= 2 - x*y // 48 bits
	y *= 2 - x*y // 96 bits; good enough
	return
}
