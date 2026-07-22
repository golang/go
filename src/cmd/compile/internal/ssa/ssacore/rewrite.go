// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssacore

import (
	"fmt"
	"strings"

	"cmd/compile/internal/logopt"
	"cmd/compile/internal/ssa/ssaop"
	"cmd/compile/internal/types"
	"cmd/internal/src"
)

// Aux is an interface to hold miscellaneous data in Blocks and Values.
type Aux interface {
	CanBeAnSSAAux()
}

func BoolToAuxInt(b bool) int64 {
	if b {
		return 1
	}
	return 0
}

// FlagConstant represents the result of a compile-time comparison.
// The sense of these flags does not necessarily represent the hardware's notion
// of a flags register - these are just a compile-time construct.
// We happen to match the semantics to those of arm/arm64.
// Note that these semantics differ from x86: the carry flag has the opposite
// sense on a subtraction!
//
//	On amd64, C=1 represents a borrow, e.g. SBB on amd64 does x - y - C.
//	On arm64, C=0 represents a borrow, e.g. SBC on arm64 does x - y - ^C.
//	 (because it does x + ^y + C).
//
// See https://en.wikipedia.org/wiki/Carry_flag#Vs._borrow_flag
type FlagConstant uint8

func IsNewObjectCall(aux Aux) bool {
	fn := aux.(*AuxCall).Fn
	return fn != nil && fn.String() == "runtime.newobject"
}

func IsSpecializedMalloc(aux Aux) bool {
	fn := aux.(*AuxCall).Fn
	if fn == nil {
		return false
	}
	name := fn.String()
	return strings.HasPrefix(name, "runtime.mallocgcSmallNoScanSC") ||
		strings.HasPrefix(name, "runtime.mallocgcSmallScanNoHeaderSC") ||
		strings.HasPrefix(name, "runtime.mallocgcTinySC")
}

// StringAux wraps string values for use in Aux.
type StringAux string

func StringToAux(s string) Aux {
	return StringAux(s)
}

func AuxIntToArm64ConditionalParams(i int64) Arm64ConditionalParams {
	var params Arm64ConditionalParams
	params.Cond = ssaop.Op(i & 0xffff)
	i >>= 16
	params.NzcvVal = uint8(i & 0x0f)
	i >>= 4
	params.ConstVal = uint8(i & 0x1f)
	i >>= 5
	params.Ind = i == 1
	return params
}

func LogLargeCopy(funcName string, pos src.XPos, s int64) {
	if s < 128 {
		return
	}
	if logopt.Enabled() {
		logopt.LogOpt(pos, "copy", "lower", funcName, fmt.Sprintf("%d bytes", s))
	}
}

// for now only used to mark moves that need to avoid clobbering flags
type auxMark bool

var AuxMark auxMark

// PanicBoundsC contains a constant for a bounds failure.
type PanicBoundsC struct {
	C int64
}

// PanicBoundsCC contains 2 constants for a bounds failure.
type PanicBoundsCC struct {
	Cx int64
	Cy int64
}

func GetPPC64Shiftsh(auxint int64) int64 {
	return int64(int8(auxint >> 16))
}

func GetPPC64Shiftmb(auxint int64) int64 {
	return int64(int8(auxint >> 8))
}

// DecodePPC64RotateMask is the inverse operation of encodePPC64RotateMask.  The values returned as
// mb and me satisfy the POWER ISA definition of MASK(x,y) where MASK(mb,me) = mask.
func DecodePPC64RotateMask(sauxint int64) (rotate, mb, me int64, mask uint64) {
	auxint := uint64(sauxint)
	rotate = int64((auxint >> 16) & 0xFF)
	mb = int64((auxint >> 8) & 0xFF)
	me = int64((auxint >> 0) & 0xFF)
	nbits := int64((auxint >> 24) & 0xFF)
	mask = ((1 << uint(nbits-mb)) - 1) ^ ((1 << uint(nbits-me)) - 1)
	if mb > me {
		mask = ^mask
	}
	if nbits == 32 {
		mask = uint64(uint32(mask))
	}

	// Fixup ME to match ISA definition.  The second argument to MASK(..,me)
	// is inclusive.
	me = (me - 1) & (nbits - 1)
	return
}

// DivisionNeedsFixUp reports whether the division needs fix-up code.
func DivisionNeedsFixUp(v *Value) bool {
	return v.AuxInt == 0
}

// ZeroUpper32Bits checks if value zeroes out upper 32-bit of 64-bit register.
// depth limits recursion depth. In AMD64.rules 3 is used as limit,
// because it catches same amount of cases as 4.
func ZeroUpper32Bits(x *Value) bool { return zeroUpperBits(x, 32, 3) }

// zeroUpperBits reports whether the 64-bit register holding x provably has
// its upper `bits` bits zero, i.e. the value is below 2^(64-bits).
//
// Which ops guarantee this is declared per op in the _gen op definitions
// (the zeroUpperBits attribute); only the value-dependent cases live here.
func zeroUpperBits(x *Value, bits int64, depth int) bool {
	if x.Type.IsSigned() && 8*x.Type.Size() <= 64-bits {
		// A spill/restore sign-extends from the type's width (issue 68227).
		// A signed type no wider than the claimed value width may have its
		// sign bit set, so a restore can write ones into the upper bits.
		// Wider signed types are safe: their value is below the type's
		// sign bit, so a restore zero-extends.
		return false
	}
	if int64(ssaop.OpcodeTable[x.Op].ZeroUpperBits) >= bits {
		return true
	}
	switch x.Op {
	case ssaop.OpAMD64MOVQconst, ssaop.OpAMD64MOVLconst:
		// A constant qualifies whenever its value fits the claimed width.
		// (MOVLconst always zeroes the upper 32 bits, so for bits==32 it
		// is already handled by its zeroUpperBits attribute.)
		return uint64(x.AuxInt)>>(64-bits) == 0
	case ssaop.OpArg: // note: but not ArgIntReg
		// amd64 always loads args from the stack unsigned.
		// most other architectures load them sign/zero extended based on the type.
		return 8*x.Type.Size() == 64-bits && x.Block.Func.Config.Arch == "amd64"
	case ssaop.OpSelect0, ssaop.OpSelect1:
		// A Select names one register result of a tuple-producing op, so
		// the question is what that op's write does. The op's attribute
		// covers every integer result; a Select of a non-covered result
		// (flags, memory) never appears as an operand of the rules that
		// ask about upper bits.
		return int64(ssaop.OpcodeTable[x.Args[0].Op].ZeroUpperBits) >= bits
	case ssaop.OpPhi:
		// Phis can use each-other as an arguments, instead of tracking visited values,
		// just limit recursion depth.
		if depth <= 0 {
			return false
		}
		for i := range x.Args {
			if !zeroUpperBits(x.Args[i], bits, depth-1) {
				return false
			}
		}
		return true
	}
	return false
}

// ZeroUpper48Bits is similar to ZeroUpper32Bits, but for upper 48 bits.
func ZeroUpper48Bits(x *Value) bool { return zeroUpperBits(x, 48, 3) }

// ZeroUpper56Bits is similar to ZeroUpper32Bits, but for upper 56 bits.
func ZeroUpper56Bits(x *Value) bool { return zeroUpperBits(x, 56, 3) }

// IsSamePtr reports whether p1 and p2 point to the same address.
func IsSamePtr(p1, p2 *Value) bool {
	if p1 == p2 {
		return true
	}
	if p1.Op != p2.Op {
		for p1.Op == ssaop.OpOffPtr && p1.AuxInt == 0 {
			p1 = p1.Args[0]
		}
		for p2.Op == ssaop.OpOffPtr && p2.AuxInt == 0 {
			p2 = p2.Args[0]
		}
		if p1 == p2 {
			return true
		}
		if p1.Op != p2.Op {
			return false
		}
	}
	switch p1.Op {
	case ssaop.OpOffPtr:
		return p1.AuxInt == p2.AuxInt && IsSamePtr(p1.Args[0], p2.Args[0])
	case ssaop.OpAddr, ssaop.OpLocalAddr:
		return p1.Aux == p2.Aux
	case ssaop.OpAddPtr:
		return p1.Args[1] == p2.Args[1] && IsSamePtr(p1.Args[0], p2.Args[0])
	}
	return false
}

// Disjoint reports whether the memory region specified by [p1:p1+t1.Size())
// does not overlap with [p2:p2+t2.Size()).
// A return value of false does not imply the regions overlap.
func Disjoint(p1 *Value, t1 *types.Type, p2 *Value, t2 *types.Type) bool {
	return Disjoint1(p1, t1.Size(), p2, t2.Size())
}

// Disjoint1 reports whether the memory region specified by [p1:p1+n1)
// does not overlap with [p2:p2+n2).
// A return value of false does not imply the regions overlap.
func Disjoint1(p1 *Value, n1 int64, p2 *Value, n2 int64) bool {
	if n1 == 0 || n2 == 0 {
		return true
	}
	if p1 == p2 {
		return false
	}
	baseAndOffset := func(ptr *Value) (base *Value, offset int64) {
		base, offset = ptr, 0
		for base.Op == ssaop.OpOffPtr {
			offset += base.AuxInt
			base = base.Args[0]
		}
		if ssaop.OpcodeTable[base.Op].NilCheck {
			base = base.Args[0]
		}
		return base, offset
	}

	// Run types-based analysis
	if DisjointTypes(p1.Type, p2.Type) {
		return true
	}

	p1, off1 := baseAndOffset(p1)
	p2, off2 := baseAndOffset(p2)
	if IsSamePtr(p1, p2) {
		return !Overlap(off1, n1, off2, n2)
	}
	// p1 and p2 are not the same, so if they are both OpAddrs then
	// they point to different variables.
	// If one pointer is on the stack and the other is an argument
	// then they can't overlap.
	switch p1.Op {
	case ssaop.OpAddr, ssaop.OpLocalAddr:
		if p2.Op == ssaop.OpAddr || p2.Op == ssaop.OpLocalAddr || p2.Op == ssaop.OpSP {
			return true
		}
		return (p2.Op == ssaop.OpArg || p2.Op == ssaop.OpArgIntReg) && p1.Args[0].Op == ssaop.OpSP
	case ssaop.OpArg, ssaop.OpArgIntReg:
		if p2.Op == ssaop.OpSP || p2.Op == ssaop.OpLocalAddr {
			return true
		}
	case ssaop.OpSP:
		return p2.Op == ssaop.OpAddr || p2.Op == ssaop.OpLocalAddr || p2.Op == ssaop.OpArg || p2.Op == ssaop.OpArgIntReg || p2.Op == ssaop.OpSP
	}
	return false
}

// DisjointTypes reports whether a memory region pointed to by a pointer of type
// t1 does not overlap with a memory region pointed to by a pointer of type t2 --
// based on type aliasing rules.
func DisjointTypes(t1 *types.Type, t2 *types.Type) bool {
	// Unsafe pointer can alias with anything.
	if t1.IsUnsafePtr() || t2.IsUnsafePtr() {
		return false
	}

	if !t1.IsPtr() || !t2.IsPtr() {
		// Treat non-pointer types (such as TFUNC, TMAP, uintptr) conservatively.
		return false
	}

	t1 = t1.Elem()
	t2 = t2.Elem()

	// Not-in-heap types are not supported -- they are rare and non-important; also,
	// type.HasPointers check doesn't work for them correctly.
	if t1.NotInHeap() || t2.NotInHeap() {
		return false
	}

	isPtrShaped := func(t *types.Type) bool { return int(t.Size()) == types.PtrSize && t.HasPointers() }

	// Pointers and non-pointers are disjoint (https://pkg.go.dev/unsafe#Pointer).
	if (isPtrShaped(t1) && !t2.HasPointers()) ||
		(isPtrShaped(t2) && !t1.HasPointers()) {
		return true
	}

	return false
}

// Overlap reports whether the ranges given by the given offset and
// size pairs Overlap.
func Overlap(offset1, size1, offset2, size2 int64) bool {
	if offset1 >= offset2 && offset2+size2 > offset1 {
		return true
	}
	if offset2 >= offset1 && offset1+size1 > offset2 {
		return true
	}
	return false
}

// isInlinableMemmove reports whether the given arch performs a Move of the given size
// faster than memmove. It will only return true if replacing the memmove with a Move is
// safe, either because Move will do all of its loads before any of its stores, or
// because the arguments are known to be disjoint.
// This is used as a check for replacing memmove with Move ops.
func isInlinableMemmove(dst, src *Value, sz int64, c *Config) bool {
	// It is always safe to convert memmove into Move when its arguments are disjoint.
	// Move ops may or may not be faster for large sizes depending on how the platform
	// lowers them, so we only perform this optimization on platforms that we know to
	// have fast Move ops.
	switch c.Arch {
	case "amd64":
		return sz <= 16 || (sz < 1024 && Disjoint1(dst, sz, src, sz))
	case "arm64":
		return sz <= 64 || (sz <= 1024 && Disjoint1(dst, sz, src, sz))
	case "loong64":
		return sz <= 16 || (sz <= 64 && Disjoint1(dst, sz, src, sz))
	case "386":
		return sz <= 8
	case "s390x", "ppc64", "ppc64le":
		return sz <= 8 || Disjoint1(dst, sz, src, sz)
	case "arm", "mips", "mips64", "mipsle", "mips64le":
		return sz <= 4
	}
	return false
}

func IsInlinableMemmove(dst, src *Value, sz int64, c *Config) bool {
	return isInlinableMemmove(dst, src, sz, c)
}

func (auxMark) CanBeAnSSAAux() {}

func (StringAux) CanBeAnSSAAux() {}

// returns the Lsb part of the auxInt field of arm64 bitfield ops.
func (bfc Arm64BitField) Lsb() int64 {
	return int64(uint64(bfc) >> 8)
}

// returns the Width part of the auxInt field of arm64 bitfield ops.
func (bfc Arm64BitField) Width() int64 {
	return int64(bfc) & 0xff
}

// extracts NZCV flags from auxint.
func (condParams Arm64ConditionalParams) Nzcv() int64 {
	return int64(condParams.NzcvVal)
}

// extracts constant value from auxint if present.
func (condParams Arm64ConditionalParams) ConstValue() (int64, bool) {
	return int64(condParams.ConstVal), condParams.Ind
}

// N reports whether the result of an operation is negative (high bit set).
func (fc FlagConstant) N() bool {
	return fc&1 != 0
}

// Z reports whether the result of an operation is 0.
func (fc FlagConstant) Z() bool {
	return fc&2 != 0
}

// C reports whether an unsigned add overflowed (carry), or an
// unsigned subtract did not underflow (borrow).
func (fc FlagConstant) C() bool {
	return fc&4 != 0
}

// V reports whether a signed operation overflowed or underflowed.
func (fc FlagConstant) V() bool {
	return fc&8 != 0
}

func (fc FlagConstant) Eq() bool {
	return fc.Z()
}

func (fc FlagConstant) Ne() bool {
	return !fc.Z()
}

func (fc FlagConstant) Lt() bool {
	return fc.N() != fc.V()
}

func (fc FlagConstant) Le() bool {
	return fc.Z() || fc.Lt()
}

func (fc FlagConstant) Gt() bool {
	return !fc.Z() && fc.Ge()
}

func (fc FlagConstant) Ge() bool {
	return fc.N() == fc.V()
}

func (fc FlagConstant) Ult() bool {
	return !fc.C()
}

func (fc FlagConstant) Ule() bool {
	return fc.Z() || fc.Ult()
}

func (fc FlagConstant) Ugt() bool {
	return !fc.Z() && fc.Uge()
}

func (fc FlagConstant) Uge() bool {
	return fc.C()
}

func (fc FlagConstant) LtNoov() bool {
	return fc.Lt() && !fc.V()
}

func (fc FlagConstant) LeNoov() bool {
	return fc.Le() && !fc.V()
}

func (fc FlagConstant) GtNoov() bool {
	return fc.Gt() && !fc.V()
}

func (fc FlagConstant) GeNoov() bool {
	return fc.Ge() && !fc.V()
}

func (fc FlagConstant) String() string {
	return fmt.Sprintf("N=%v,Z=%v,C=%v,V=%v", fc.N(), fc.Z(), fc.C(), fc.V())
}

func (p PanicBoundsC) CanBeAnSSAAux() {
}

func (p PanicBoundsCC) CanBeAnSSAAux() {
}
