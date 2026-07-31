// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import (
	"encoding/binary"
	"fmt"
	"internal/buildcfg"
	"io"
	"math"
	"math/bits"
	"os"
	"path/filepath"
	"strings"

	"cmd/compile/internal/base"
	"cmd/compile/internal/logopt"
	"cmd/compile/internal/ssa/ssaop"
	"cmd/compile/internal/types"
	"cmd/internal/obj"
	"cmd/internal/obj/s390x"
	"cmd/internal/objabi"
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

// AddFlags32 returns the flags that would be set from computing x+y.
func AddFlags32(x, y int32) FlagConstant {
	var fcb FlagConstantBuilder
	fcb.Z = x+y == 0
	fcb.N = x+y < 0
	fcb.C = uint32(x+y) < uint32(x)
	fcb.V = x >= 0 && y >= 0 && x+y < 0 || x < 0 && y < 0 && x+y >= 0
	return fcb.Encode()
}

// Note: addFlags(x,y) != subFlags(x,-y) in some situations:
//  - the results of the C flag are different
//  - the results of the V flag when y==minint are different

// AddFlags64 returns the flags that would be set from computing x+y.
func AddFlags64(x, y int64) FlagConstant {
	var fcb FlagConstantBuilder
	fcb.Z = x+y == 0
	fcb.N = x+y < 0
	fcb.C = uint64(x+y) < uint64(x)
	fcb.V = x >= 0 && y >= 0 && x+y < 0 || x < 0 && y < 0 && x+y >= 0
	return fcb.Encode()
}

func Arm64BitFieldToAuxInt(v Arm64BitField) int64 {
	return int64(v)
}

func Arm64ConditionalParamsToAuxInt(v Arm64ConditionalParams) int64 {
	if v.Cond&^0xffff != 0 {
		panic("condition value exceeds 16 bits")
	}

	var i int64
	if v.Ind {
		i = 1 << 25
	}
	i |= int64(v.ConstVal) << 20
	i |= int64(v.NzcvVal) << 16
	i |= int64(v.Cond)
	return i
}

// encodes the lsb and width for arm(64) bitfield ops into the expected auxInt format.
func ArmBFAuxInt(lsb, width int64) Arm64BitField {
	if lsb < 0 || lsb > 63 {
		panic("ARM(64) bit field lsb constant out of range")
	}
	if width < 1 || lsb+width > 64 {
		panic("ARM(64) bit field width constant out of range")
	}
	return Arm64BitField(width | lsb<<8)
}

func AuxIntToArm64BitField(i int64) Arm64BitField {
	return Arm64BitField(i)
}

func AuxIntToBool(i int64) bool {
	if i == 0 {
		return false
	}
	return true
}

func AuxIntToFlagConstant(x int64) FlagConstant {
	return FlagConstant(x)
}

func AuxIntToFloat32(i int64) float32 {
	return float32(math.Float64frombits(uint64(i)))
}

func AuxIntToFloat64(i int64) float64 {
	return math.Float64frombits(uint64(i))
}

func AuxIntToInt16(i int64) int16 {
	return int16(i)
}

func AuxIntToInt32(i int64) int32 {
	return int32(i)
}

func AuxIntToInt64(i int64) int64 {
	return i
}

func AuxIntToInt8(i int64) int8 {
	return int8(i)
}

func AuxIntToOp(cc int64) ssaop.Op {
	return ssaop.Op(cc)
}

func AuxIntToUint64(i int64) uint64 {
	return uint64(i)
}

func AuxIntToUint8(i int64) uint8 {
	return uint8(i)
}

func AuxIntToValAndOff(i int64) ValAndOff {
	return ValAndOff(i)
}

func AuxToCall(i Aux) *AuxCall {
	return i.(*AuxCall)
}

func AuxToPanicBoundsC(i Aux) PanicBoundsC {
	return i.(PanicBoundsC)
}

func AuxToPanicBoundsCC(i Aux) PanicBoundsCC {
	return i.(PanicBoundsCC)
}

func AuxToS390xCCMask(i Aux) s390x.CCMask {
	return i.(s390x.CCMask)
}

func AuxToS390xRotateParams(i Aux) s390x.RotateParams {
	return i.(s390x.RotateParams)
}

func AuxToString(i Aux) string {
	return string(i.(StringAux))
}

func AuxToSym(i Aux) Sym {
	// TODO: kind of a hack - allows nil interface through
	s, _ := i.(Sym)
	return s
}

func AuxToType(i Aux) *types.Type {
	return i.(*types.Type)
}

// B2i translates a boolean value to 0 or 1 for assigning to auxInt.
func B2i(b bool) int64 {
	if b {
		return 1
	}
	return 0
}

// B2i32 translates a boolean value to 0 or 1.
func B2i32(b bool) int32 {
	if b {
		return 1
	}
	return 0
}

func CallToAux(s *AuxCall) Aux {
	return s
}

// CanMergeLoad reports whether the load can be merged into target without
// invalidating the schedule.
func CanMergeLoad(target, load *Value) bool {
	if target.Block.ID != load.Block.ID {
		// If the load is in a different block do not merge it.
		return false
	}

	// We can't merge the load into the target if the load
	// has more than one use.
	if load.Uses != 1 {
		return false
	}

	mem := load.MemoryArg()

	// We need the load's memory arg to still be alive at target. That
	// can't be the case if one of target's args depends on a memory
	// state that is a successor of load's memory arg.
	//
	// For example, it would be invalid to merge load into target in
	// the following situation because newmem has killed oldmem
	// before target is reached:
	//     load = read ... oldmem
	//   newmem = write ... oldmem
	//     arg0 = read ... newmem
	//   target = add arg0 load
	//
	// If the argument comes from a different block then we can exclude
	// it immediately because it must dominate load (which is in the
	// same block as target).
	var args []*Value
	for _, a := range target.Args {
		if a != load && a.Block.ID == target.Block.ID {
			args = append(args, a)
		}
	}

	f := target.Block.Func
	visited := f.NewSparseSet(f.NumValues())
	defer f.RetSparseSet(visited)

	// memPreds contains memory states known to be predecessors of load's
	// memory state. It is lazily initialized.
	var memPreds map[*Value]bool
	for len(args) > 0 {
		const limit = 2048 // enough to comfortably cover unrolled crypto blocks
		if visited.Size() >= limit {
			// Give up if we have visited a lot of values.
			return false
		}
		v := args[len(args)-1]
		args = args[:len(args)-1]
		if visited.Contains(v.ID) {
			continue
		}
		visited.Add(v.ID)
		if target.Block.ID != v.Block.ID {
			// Since target and load are in the same block
			// we can stop searching when we leave the block.
			continue
		}
		if v.Op == ssaop.OpPhi {
			// A Phi implies we have reached the top of the block.
			// The memory phi, if it exists, is always
			// the first logical store in the block.
			continue
		}
		if v.Type.IsTuple() && v.Type.FieldType(1).IsMemory() {
			// We could handle this situation however it is likely
			// to be very rare.
			return false
		}
		if v.Op.SymEffect()&ssaop.SymAddr != 0 {
			// This case prevents an operation that calculates the
			// address of a local variable from being forced to schedule
			// before its corresponding VarDef.
			// See issue 28445.
			//   v1 = LOAD ...
			//   v2 = VARDEF
			//   v3 = LEAQ
			//   v4 = CMPQ v1 v3
			// We don't want to combine the CMPQ with the load, because
			// that would force the CMPQ to schedule before the VARDEF, which
			// in turn requires the LEAQ to schedule before the VARDEF.
			return false
		}
		if v.Type.IsMemory() {
			if memPreds == nil {
				// Initialise a map containing memory states
				// known to be predecessors of load's memory
				// state.
				memPreds = make(map[*Value]bool)
				m := mem
				const limit = 50
				for i := 0; i < limit; i++ {
					if m.Op == ssaop.OpPhi {
						// The memory phi, if it exists, is always
						// the first logical store in the block.
						break
					}
					if m.Block.ID != target.Block.ID {
						break
					}
					if !m.Type.IsMemory() {
						break
					}
					memPreds[m] = true
					if len(m.Args) == 0 {
						break
					}
					m = m.MemoryArg()
				}
			}

			// We can merge if v is a predecessor of mem.
			//
			// For example, we can merge load into target in the
			// following scenario:
			//      x = read ... v
			//    mem = write ... v
			//   load = read ... mem
			// target = add x load
			if memPreds[v] {
				continue
			}
			return false
		}
		if len(v.Args) > 0 && v.Args[len(v.Args)-1] == mem {
			// If v takes mem as an input then we know mem
			// is valid at this point.
			continue
		}
		for _, a := range v.Args {
			if target.Block.ID == a.Block.ID {
				args = append(args, a)
			}
		}
	}

	return true
}

// CanMergeLoadClobber reports whether the load can be merged into target without
// invalidating the schedule.
// It also checks that the other non-load argument x is something we
// are ok with clobbering.
func CanMergeLoadClobber(target, load, x *Value) bool {
	// The register containing x is going to get clobbered.
	// Don't merge if we still need the value of x.
	// We don't have liveness information here, but we can
	// approximate x dying with:
	//  1) target is x's only use.
	//  2) target is not in a deeper loop than x.
	switch {
	case x.Uses == 2 && x.Op == ssaop.OpPhi && len(x.Args) == 2 && (x.Args[0] == target || x.Args[1] == target) && target.Uses == 1:
		// This is a simple detector to determine that x is probably
		// not live after target. (It does not need to be perfect,
		// regalloc will issue a reg-reg move to save it if we are wrong.)
		// We have:
		//   x = Phi(?, target)
		//   target = Op(load, x)
		// Because target has only one use as a Phi argument, we can schedule it
		// very late. Hopefully, later than the other use of x. (The other use died
		// between x and target, or exists on another branch entirely).
	case x.Uses > 1:
		return false
	}
	loopnest := x.Block.Func.Loopnest()
	if loopnest.Depth(target.Block.ID) > loopnest.Depth(x.Block.ID) {
		return false
	}
	return CanMergeLoad(target, load)
}

func CanMergeSym(x, y Sym) bool {
	return x == nil || y == nil
}

func CanMulStrengthReduce(config *Config, x int64) bool {
	_, ok := config.MulRecipes[x]
	return ok
}

func CanMulStrengthReduce32(config *Config, x int32) bool {
	_, ok := config.MulRecipes[int64(x)]
	return ok
}

// CanonLessThan returns whether x is "ordered" less than y, for purposes of normalizing
// generated code as much as possible.
func CanonLessThan(x, y *Value) bool {
	if x.Op != y.Op {
		return x.Op < y.Op
	}
	if !x.Pos.SameFileAndLine(y.Pos) {
		return x.Pos.Before(y.Pos)
	}
	return x.ID < y.ID
}

// Clobber invalidates values. Returns true.
// Clobber is used by rewrite rules to:
//
//	A) make sure the values are really dead and never used again.
//	B) decrement use counts of the values' args.
func Clobber(vv ...*Value) bool {
	for _, v := range vv {
		v.Reset(ssaop.OpInvalid)
		// Note: leave v.Block intact.  The Block field is used after clobber.
	}
	return true
}

// ClobberIfDead resets v when use count is 1. Returns true.
// ClobberIfDead is used by rewrite rules to decrement
// use counts of v's args when v is dead and never used.
func ClobberIfDead(v *Value) bool {
	if v.Uses == 1 {
		v.Reset(ssaop.OpInvalid)
	}
	// Note: leave v.Block intact.  The Block field is used after clobberIfDead.
	return true
}

// CountRule increments Func.ruleMatches[key].
// If Func.ruleMatches is non-nil at the end
// of compilation, it will be printed to stdout.
// This is intended to make it easier to find which functions
// which contain lots of rules matches when developing new rules.
func CountRule(v *Value, key string) bool {
	f := v.Block.Func
	if f.RuleMatches == nil {
		f.RuleMatches = make(map[string]int)
	}
	f.RuleMatches[key]++
	return true
}

type DeadValueChoice bool

// Compress mask and shift into single value of the form
// me | mb<<8 | rotate<<16 | nbits<<24 where me and mb can
// be used to regenerate the input mask.
func EncodePPC64RotateMask(rotate, mask, nbits int64) int64 {
	var mb, me, mbn, men int

	// Determine boundaries and then decode them
	if mask == 0 || ^mask == 0 || rotate >= nbits {
		panic(fmt.Sprintf("invalid PPC64 rotate mask: %x %d %d", uint64(mask), rotate, nbits))
	} else if nbits == 32 {
		mb = bits.LeadingZeros32(uint32(mask))
		me = 32 - bits.TrailingZeros32(uint32(mask))
		mbn = bits.LeadingZeros32(^uint32(mask))
		men = 32 - bits.TrailingZeros32(^uint32(mask))
	} else {
		mb = bits.LeadingZeros64(uint64(mask))
		me = 64 - bits.TrailingZeros64(uint64(mask))
		mbn = bits.LeadingZeros64(^uint64(mask))
		men = 64 - bits.TrailingZeros64(^uint64(mask))
	}
	// Check for a wrapping mask (e.g bits at 0 and 63)
	if mb == 0 && me == int(nbits) {
		// swap the inverted values
		mb, me = men, mbn
	}

	return int64(me) | int64(mb<<8) | rotate<<16 | nbits<<24
}

// for a pseudo-op like (LessThan x), extract x.
func FlagArg(v *Value) *Value {
	if len(v.Args) != 1 || !v.Args[0].Type.IsFlags() {
		return nil
	}
	return v.Args[0]
}

type FlagConstantBuilder struct {
	N bool
	Z bool
	C bool
	V bool
}

func FlagConstantToAuxInt(x FlagConstant) int64 {
	return int64(x)
}

func Float32ToAuxInt(f float32) int64 {
	return int64(math.Float64bits(float64(f)))
}

func Float64ToAuxInt(f float64) int64 {
	return int64(math.Float64bits(f))
}

// When v is (IMake typ (StructMake ...)), convert to
// (IMake typ arg) where arg is the pointer-y argument to
// the StructMake (there must be exactly one).
func ImakeOfStructMake(v *Value) *Value {
	var arg *Value
	for _, a := range v.Args[1].Args {
		if a.Type.Size() > 0 {
			arg = a
			break
		}
	}
	return v.Block.NewValue2(v.Pos, ssaop.OpIMake, v.Type, v.Args[0], arg)
}

func Int16ToAuxInt(i int16) int64 {
	return int64(i)
}

func Int32ToAuxInt(i int32) int64 {
	return int64(i)
}

func Int64ToAuxInt(i int64) int64 {
	return i
}

func Int8ToAuxInt(i int8) int64 {
	return int64(i)
}

// Is12Bit reports whether n can be represented as a signed 12 bit integer.
func Is12Bit(n int64) bool {
	return -(1<<11) <= n && n < (1<<11)
}

// Is16Bit reports whether n can be represented as a signed 16 bit integer.
func Is16Bit(n int64) bool {
	return n == int64(int16(n))
}

func Is16BitInt(t *types.Type) bool {
	return t.Size() == 2 && t.IsInteger()
}

// Is20Bit reports whether n can be represented as a signed 20 bit integer.
func Is20Bit(n int64) bool {
	return -(1<<19) <= n && n < (1<<19)
}

// Is32Bit reports whether n can be represented as a signed 32 bit integer.
func Is32Bit(n int64) bool {
	return n == int64(int32(n))
}

func Is32BitFloat(t *types.Type) bool {
	return t.Size() == 4 && t.IsFloat()
}

func Is32BitInt(t *types.Type) bool {
	return t.Size() == 4 && t.IsInteger()
}

// Common functions called from rewriting rules

func Is64BitFloat(t *types.Type) bool {
	return t.Size() == 8 && t.IsFloat()
}

func Is64BitInt(t *types.Type) bool {
	return t.Size() == 8 && t.IsInteger()
}

func Is8BitInt(t *types.Type) bool {
	return t.Size() == 1 && t.IsInteger()
}

// isPowerOfTwoX functions report whether n is a power of 2.
func IsPowerOfTwo[T int8 | int16 | int32 | int64 | uint8 | uint16 | uint32 | uint64](n T) bool {
	return n > 0 && n&(n-1) == 0
}

// This verifies that the mask is a set of
// consecutive bits including the least
// significant bit.
func IsPPC64ValidShiftMask(v int64) bool {
	if (v != 0) && ((v+1)&v) == 0 {
		return true
	}
	return false
}

// Test if this value can encoded as a mask for a rlwinm like
// operation.  Masks can also extend from the msb and wrap to
// the lsb too.  That is, the valid masks are 32 bit strings
// of the form: 0..01..10..0 or 1..10..01..1 or 1...1
//
// Note: This ignores the upper 32 bits of the input. When a
// zero extended result is desired (e.g a 64 bit result), the
// user must verify the upper 32 bits are 0 and the mask is
// contiguous (that is, non-wrapping).
func IsPPC64WordRotateMask(v64 int64) bool {
	// Isolate rightmost 1 (if none 0) and add.
	v := uint32(v64)
	vp := (v & -v) + v
	// Likewise, for the wrapping case.
	vn := ^v
	vpn := (vn & -vn) + vn
	return (v&vp == 0 || vn&vpn == 0) && v != 0
}

func IsPtr(t *types.Type) bool {
	return t.IsPtrShaped()
}

// IsSameCall reports whether aux is the same as the given named symbol.
func IsSameCall(aux Aux, name string) bool {
	fn := aux.(*AuxCall).Fn
	return fn != nil && fn.String() == name
}

// IsU32Bit reports whether n can be represented as an unsigned 32 bit integer.
func IsU32Bit(n int64) bool {
	return n == int64(uint32(n))
}

// IsVolatile reports whether v is a pointer to argument region on stack which
// will be clobbered by a function call.
func IsVolatile(v *Value) bool {
	for v.Op == ssaop.OpOffPtr || v.Op == ssaop.OpAddPtr || v.Op == ssaop.OpPtrIndex || v.Op == ssaop.OpCopy || v.Op == ssaop.OpSelectNAddr {
		v = v.Args[0]
	}
	return v.Op == ssaop.OpSP
}

const (
	LeaveDeadValues  DeadValueChoice = false
	RemoveDeadValues                 = true

	RepZeroThreshold = 1408 // size beyond which we use REP STOS for zeroing
	RepMoveThreshold = 1408 // size beyond which we use REP MOVS for copying
)

func Log16(n int16) int64 { return Log16u(uint16(n)) }

func Log16u(n uint16) int64 { return int64(bits.Len16(n)) - 1 }

func Log32(n int32) int64 { return Log32u(uint32(n)) }

func Log32u(n uint32) int64 { return int64(bits.Len32(n)) - 1 }

func Log64(n int64) int64 { return Log64u(uint64(n)) }

func Log64u(n uint64) int64 { return int64(bits.Len64(n)) - 1 }

// logXu returns the logarithm of n base 2.
// n must be a power of 2 (isPowerOfTwo returns true)
func Log8u(n uint8) int64 { return int64(bits.Len8(n)) - 1 }

// LogicFlags32 returns flags set to the sign/zeroness of x.
// C and V are set to false.
func LogicFlags32(x int32) FlagConstant {
	var fcb FlagConstantBuilder
	fcb.Z = x == 0
	fcb.N = x < 0
	return fcb.Encode()
}

// LogicFlags64 returns flags set to the sign/zeroness of x.
// C and V are set to false.
func LogicFlags64(x int64) FlagConstant {
	var fcb FlagConstantBuilder
	fcb.Z = x == 0
	fcb.N = x < 0
	return fcb.Encode()
}

// LogLargeCopyValue logs the occurrence of a large copy.
// The best place to do this is in the rewrite rules where the size of the move is easy to find.
// "Large" is arbitrarily chosen to be 128 bytes; this may change.
func LogLargeCopyValue(v *Value, s int64) bool {
	if s < 128 {
		return true
	}
	if logopt.Enabled() {
		logopt.LogOpt(v.Pos, "copy", "lower", v.Block.Func.Name, fmt.Sprintf("%d bytes", s))
	}
	return true
}

// LogRule logs the use of the rule s. This will only be enabled if
// rewrite rules were generated with the -log option, see _gen/rulegen.go.
func LogRule(s string) {
	if ruleFile == nil {
		// Open a log file to write log to. We open in append
		// mode because all.bash runs the compiler lots of times,
		// and we want the concatenation of all of those logs.
		// This means, of course, that users need to rm the old log
		// to get fresh data.
		// TODO: all.bash runs compilers in parallel. Need to synchronize logging somehow?
		w, err := os.OpenFile(filepath.Join(os.Getenv("GOROOT"), "src", "rulelog"),
			os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0666)
		if err != nil {
			panic(err)
		}
		ruleFile = w
	}
	// Ignore errors in case of multiple processes fighting over the file.
	fmt.Fprintln(ruleFile, s)
}

func MakeJumpTableSym(b *Block) *obj.LSym {
	s := base.Ctxt.Lookup(fmt.Sprintf("%s.jump%d", b.Func.Fe.Func().LSym.Name, b.ID))
	// The jump table symbol is accessed only from the function symbol.
	s.Set(obj.AttrStatic, true)
	return s
}

// Combine (ANDconst [m] (SRWconst [s])) into (RLWINM [y]) or return 0
func MergePPC64AndSrwi(m, s int64) int64 {
	mask := MergePPC64RShiftMask(m, s, 32)
	if !IsPPC64WordRotateMask(mask) {
		return 0
	}
	return EncodePPC64RotateMask((32-s)&31, mask, 32)
}

// Test if a RLWINM feeding into a CLRLSLDI can be merged into RLWINM.  Return
// the encoded RLWINM constant, or 0 if they cannot be merged.
func MergePPC64ClrlsldiRlwinm(sld int32, rlw int64) int64 {
	r_1, _, _, mask_1 := DecodePPC64RotateMask(rlw)
	// for CLRLSLDI, it's more convenient to think of it as a mask left bits then rotate left.
	mask_2 := uint64(0xFFFFFFFFFFFFFFFF) >> uint(GetPPC64Shiftmb(int64(sld)))

	// combine the masks, and adjust for the final left shift.
	mask_3 := (mask_1 & mask_2) << uint(GetPPC64Shiftsh(int64(sld)))
	r_2 := GetPPC64Shiftsh(int64(sld))
	r_3 := (r_1 + r_2) & 31 // This can wrap.

	// Verify the result is still a valid bitmask of <= 32 bits.
	if !IsPPC64WordRotateMask(int64(mask_3)) || uint64(uint32(mask_3)) != mask_3 {
		return 0
	}
	return EncodePPC64RotateMask(r_3, int64(mask_3), 32)
}

// Test if a word shift right feeding into a CLRLSLDI can be merged into RLWINM.
// Return the encoded RLWINM constant, or 0 if they cannot be merged.
func MergePPC64ClrlsldiSrw(sld, srw int64) int64 {
	mask_1 := uint64(0xFFFFFFFF >> uint(srw))
	// for CLRLSLDI, it's more convenient to think of it as a mask left bits then rotate left.
	mask_2 := uint64(0xFFFFFFFFFFFFFFFF) >> uint(GetPPC64Shiftmb(sld))

	// Rewrite mask to apply after the final left shift.
	mask_3 := (mask_1 & mask_2) << uint(GetPPC64Shiftsh(sld))

	r_1 := 32 - srw
	r_2 := GetPPC64Shiftsh(sld)
	r_3 := (r_1 + r_2) & 31 // This can wrap.

	if uint64(uint32(mask_3)) != mask_3 || mask_3 == 0 {
		return 0
	}
	return EncodePPC64RotateMask(r_3, int64(mask_3), 32)
}

// Decompose a shift right into an equivalent rotate/mask,
// and return mask & m.
func MergePPC64RShiftMask(m, s, nbits int64) int64 {
	smask := uint64((1<<uint(nbits))-1) >> uint(s)
	return m & int64(smask)
}

// Compute the encoded RLWINM constant from combining (SLDconst [sld] (SRWconst [srw] x)),
// or return 0 if they cannot be combined.
func MergePPC64SldiSrw(sld, srw int64) int64 {
	if sld > srw || srw >= 32 {
		return 0
	}
	mask_r := uint32(0xFFFFFFFF) >> uint(srw)
	mask_l := uint32(0xFFFFFFFF) >> uint(sld)
	mask := (mask_r & mask_l) << uint(sld)
	return EncodePPC64RotateMask((32-srw+sld)&31, int64(mask), 32)
}

// MergeSym merges two symbolic offsets. There is no real merging of
// offsets, we just pick the non-nil one.
func MergeSym(x, y Sym) Sym {
	if x == nil {
		return y
	}
	if y == nil {
		return x
	}
	panic(fmt.Sprintf("mergeSym with two non-nil syms %v %v", x, y))
}

func ModularMultiplicativeInverse(x uint64) (y uint64) {
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

// MoveSize returns the number of bytes an aligned MOV instruction moves.
func MoveSize(align int64, c *Config) int64 {
	switch {
	case align%8 == 0 && c.PtrSize == 8:
		return 8
	case align%4 == 0:
		return 4
	case align%2 == 0:
		return 2
	}
	return 1
}

// MulStrengthReduce returns v*x evaluated at the location
// (block and source position) of m.
// canMulStrengthReduce must have returned true.
func MulStrengthReduce(m *Value, v *Value, x int64) *Value {
	return v.Block.Func.Config.MulRecipes[x].Build(m, v)
}

// MulStrengthReduce32 returns v*x evaluated at the location
// (block and source position) of m.
// canMulStrengthReduce32 must have returned true.
// The upper 32 bits of m might be set to junk.
func MulStrengthReduce32(m *Value, v *Value, x int32) *Value {
	return v.Block.Func.Config.MulRecipes[int64(x)].Build(m, v)
}

func NewPPC64ShiftAuxInt(sh, mb, me, sz int64) int32 {
	if sh < 0 || sh >= sz {
		panic("PPC64 shift arg sh out of range")
	}
	if mb < 0 || mb >= sz {
		panic("PPC64 shift arg mb out of range")
	}
	if me < 0 || me >= sz {
		panic("PPC64 shift arg me out of range")
	}
	return int32(sh<<16 | mb<<8 | me)
}

// NoteRule is an easy way to track if a rule is matched when writing
// new ones.  Make the rule of interest also conditional on
//
//	NoteRule("note to self: rule of interest matched")
//
// and that message will print when the rule matches.
func NoteRule(s string) bool {
	fmt.Println(s)
	return true
}

// ntzX returns the number of trailing zeros.
func Ntz64(x int64) int { return bits.TrailingZeros64(uint64(x)) }

// OneBit reports whether x contains exactly one set bit.
func OneBit[T int8 | int16 | int32 | int64](x T) bool {
	return x&(x-1) == 0 && x != 0
}

func OpToAuxInt(o ssaop.Op) int64 {
	return int64(o)
}

func PanicBoundsCCToAux(p PanicBoundsCC) Aux {
	return p
}

func PanicBoundsCToAux(p PanicBoundsC) Aux {
	return p
}

// Read16 reads two bytes from the read-only global sym at offset off.
func Read16(sym Sym, off int64, byteorder binary.ByteOrder) uint16 {
	lsym := sym.(*obj.LSym)
	// lsym.P is written lazily.
	// Bytes requested after the end of lsym.P are 0.
	var src []byte
	if 0 <= off && off < int64(len(lsym.P)) {
		src = lsym.P[off:]
	}
	buf := make([]byte, 2)
	copy(buf, src)
	return byteorder.Uint16(buf)
}

// Read32 reads four bytes from the read-only global sym at offset off.
func Read32(sym Sym, off int64, byteorder binary.ByteOrder) uint32 {
	lsym := sym.(*obj.LSym)
	var src []byte
	if 0 <= off && off < int64(len(lsym.P)) {
		src = lsym.P[off:]
	}
	buf := make([]byte, 4)
	copy(buf, src)
	return byteorder.Uint32(buf)
}

// Read64 reads eight bytes from the read-only global sym at offset off.
func Read64(sym Sym, off int64, byteorder binary.ByteOrder) uint64 {
	lsym := sym.(*obj.LSym)
	var src []byte
	if 0 <= off && off < int64(len(lsym.P)) {
		src = lsym.P[off:]
	}
	buf := make([]byte, 8)
	copy(buf, src)
	return byteorder.Uint64(buf)
}

// Read8 reads one byte from the read-only global sym at offset off.
func Read8(sym Sym, off int64) uint8 {
	lsym := sym.(*obj.LSym)
	if off >= int64(len(lsym.P)) || off < 0 {
		// Invalid index into the global sym.
		// This can happen in dead code, so we don't want to panic.
		// Just return any value, it will eventually get ignored.
		// See issue 29215.
		return 0
	}
	return lsym.P[off]
}

func RewriteStructStore(v *Value) *Value {
	b := v.Block
	dst := v.Args[0]
	x := v.Args[1]
	if x.Op != ssaop.OpStructMake {
		base.Fatalf("invalid struct store: %v", x)
	}
	mem := v.Args[2]

	t := x.Type
	for i, arg := range x.Args {
		ft := t.FieldType(i)

		addr := b.NewValue1I(v.Pos, ssaop.OpOffPtr, ft.PtrTo(), t.FieldOff(i), dst)
		mem = b.NewValue3A(v.Pos, ssaop.OpStore, types.TypeMem, TypeToAux(ft), addr, arg, mem)
	}

	return mem
}

func S390xCCMaskToAux(c s390x.CCMask) Aux {
	return c
}

func S390xRotateParamsToAux(r s390x.RotateParams) Aux {
	return r
}

// SetPos sets the position of v to pos, then returns true.
// Useful for setting the result of a rewrite's position to
// something other than the default.
func SetPos(v *Value, pos src.XPos) bool {
	v.Pos = pos
	return true
}

// ShiftIsBounded reports whether (left/right) shift Value v is known to be bounded.
// A shift is bounded if it is shifting by less than the width of the shifted value.
func ShiftIsBounded(v *Value) bool {
	return v.AuxInt != 0
}

// SubFlags32 returns the flags that would be set from computing x-y.
func SubFlags32(x, y int32) FlagConstant {
	var fcb FlagConstantBuilder
	fcb.Z = x-y == 0
	fcb.N = x-y < 0
	fcb.C = uint32(y) <= uint32(x) // This code follows the arm carry flag model.
	fcb.V = x >= 0 && y < 0 && x-y < 0 || x < 0 && y >= 0 && x-y >= 0
	return fcb.Encode()
}

// SubFlags64 returns the flags that would be set from computing x-y.
func SubFlags64(x, y int64) FlagConstant {
	var fcb FlagConstantBuilder
	fcb.Z = x-y == 0
	fcb.N = x-y < 0
	fcb.C = uint64(y) <= uint64(x) // This code follows the arm carry flag model.
	fcb.V = x >= 0 && y < 0 && x-y < 0 || x < 0 && y >= 0 && x-y >= 0
	return fcb.Encode()
}

func SupportsPPC64PCRel() bool {
	// PCRel is currently supported for >= power10, linux only
	// Internal and external linking supports this on ppc64le; internal linking on ppc64.
	return buildcfg.GOPPC64 >= 10 && buildcfg.GOOS == "linux"
}

// SymIsRO reports whether sym is a read-only global.
func SymIsRO(sym Sym) bool {
	lsym := sym.(*obj.LSym)
	return lsym.Type == objabi.SRODATA && len(lsym.R) == 0
}

func SymToAux(s Sym) Aux {
	return s
}

func TypeToAux(t *types.Type) Aux {
	return t
}

func Uint64ToAuxInt(i uint64) int64 {
	return int64(i)
}

func Uint8ToAuxInt(i uint8) int64 {
	return int64(int8(i))
}

func ValAndOffToAuxInt(v ValAndOff) int64 {
	return int64(v)
}

var ruleFile io.Writer

func (fcs FlagConstantBuilder) Encode() FlagConstant {
	var fc FlagConstant
	if fcs.N {
		fc |= 1
	}
	if fcs.Z {
		fc |= 2
	}
	if fcs.C {
		fc |= 4
	}
	if fcs.V {
		fc |= 8
	}
	return fc
}
