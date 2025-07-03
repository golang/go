// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package asmgen

import (
	"fmt"
	"strings"
)

// Note: Exported fields and methods are expected to be used
// by function generators (like the ones in add.go and so on).
// Unexported fields and methods should not be.

// An Arch defines how to generate assembly for a specific architecture.
type Arch struct {
	Name          string // name of architecture
	Build         string // build tag
	WordBits      int    // length of word in bits (32 or 64)
	WordBytes     int    // length of word in bytes (4 or 8)
	CarrySafeLoop bool   // whether loops preserve carry flag across iterations

	// Registers.
	regs        []string // usable general registers, in allocation order
	reg0        string   // dedicated zero register
	regCarry    string   // dedicated carry register, for systems with no hardware carry bits
	regAltCarry string   // dedicated secondary carry register, for systems with no hardware carry bits
	regTmp      string   // dedicated temporary register

	// regShift indicates that the architecture supports
	// using REG1>>REG2 and REG1<<REG2 as the first source
	// operand in an arithmetic instruction. (32-bit ARM does this.)
	regShift bool

	// setup is called to emit any per-architecture function prologue,
	// immediately after the TEXT line has been emitted.
	// If setup is nil, it is taken to be a no-op.
	setup func(*Func)

	// hint returns the register to use for a given hint.
	// Returning an empty string indicates no preference.
	// If hint is nil, it is considered to return an empty string.
	hint func(*Asm, Hint) string

	// op3 reports whether the named opcode accepts 3 operands
	// (true on most instructions on most systems, but not true of x86 instructions).
	// The assembler unconditionally turns op x,z,z into op x,z.
	// If op3 returns false, then the assembler will turn op x,y,z into mov y,z; op x,z.
	// If op3 is nil, then all opcodes are assumed to accept 3 operands.
	op3 func(name string) bool

	// memOK indicates that arithmetic instructions can use memory references (like on x86)
	memOK bool

	// maxColumns is the default maximum number of vector columns
	// to process in a single [Pipe.Loop] block.
	// 0 means unlimited.
	// [Pipe.SetMaxColumns] overrides this.
	maxColumns int

	// Instruction names.
	mov   string // move (word-sized)
	add   string // add with no carry involvement
	adds  string // add, setting but not using carry
	adc   string // add, using but not setting carry
	adcs  string // add, setting and using carry
	sub   string // sub with no carry involvement
	subs  string // sub, setting but not using carry
	sbc   string // sub, using but not setting carry
	sbcs  string // sub, setting and using carry
	mul   string // multiply
	mulhi string // multiply producing high bits
	lsh   string // left shift
	lshd  string // double-width left shift
	rsh   string // right shift
	rshd  string // double-width right shift
	and   string // bitwise and
	or    string // bitwise or
	xor   string // bitwise xor
	neg   string // negate
	rsb   string // reverse subtract
	sltu  string // set less-than unsigned (dst = src2 < src1), for carry-less systems
	sgtu  string // set greater-than unsigned (dst = src2 > src1), for carry-less systems
	lea   string // load effective address

	// addF and subF implement a.Add and a.Sub
	// on systems where the situation is more complicated than
	// the six basic instructions (add, adds, adcs, sub, subs, sbcs).
	// They return a boolean indicating whether the operation was handled.
	addF func(a *Asm, src1, src2, dst Reg, carry Carry) bool
	subF func(a *Asm, src1, src2, dst Reg, carry Carry) bool

	// mulF and mulWideF implement Mul and MulWide.
	// They call Fatalf if the operation is unsupported.
	// An architecture can set the mul field instead of mulF.
	// mulWide is optional, but otherwise mulhi should be set.
	mulWideF func(a *Asm, src1, src2, dstlo, dsthi Reg)

	// addWords is a printf format taking src1, src2, dst
	// and sets dst = WordBytes*src1+src2.
	// It may modify the carry flag.
	addWords string

	// subCarryIsBorrow is true when the actual processor carry bit used in subtraction
	// is really a “borrow” bit, meaning 1 means borrow and 0 means no borrow.
	// In contrast, most systems (except x86) use a carry bit with the opposite
	// meaning: 0 means a borrow happened, and 1 means it didn't.
	subCarryIsBorrow bool

	// Jump instruction printf formats.
	// jmpZero and jmpNonZero are printf formats taking src, label
	// and jump to label if src is zero / non-zero.
	jmpZero    string
	jmpNonZero string

	// loopTop is a printf format taking src, label that should
	// jump to label if src is zero, or else set up for a loop.
	// If loopTop is not set, jmpZero is used.
	loopTop string

	// loopBottom is a printf format taking dst, label that should
	// decrement dst and then jump to label if src is non-zero.
	// If loopBottom is not set, a subtraction is used followed by
	// use of jmpNonZero.
	loopBottom string

	// loopBottomNeg is like loopBottom but used in negative-index
	// loops, which only happen memIndex is also set (only on 386).
	// It increments dst instead of decrementing it.
	loopBottomNeg string

	// Indexed memory access.
	// If set, memIndex returns a memory reference for a mov instruction
	// addressing off(ptr)(ix*WordBytes).
	// Using memIndex costs an extra register but allows the end-of-loop
	// to do a single increment/decrement instead of advancing two or three pointers.
	// This is particularly important on 386.
	memIndex func(a *Asm, off int, ix Reg, ptr RegPtr) Reg

	// Incrementing/decrementing memory access.
	// loadIncN loads memory at ptr into regs, incrementing ptr by WordBytes after each reg.
	// loadDecN loads memory at ptr into regs, decrementing ptr by WordBytes before each reg.
	// storeIncN and storeDecN are the same, but storing from regs instead of loading into regs.
	// If missing, the assembler accesses memory and advances pointers using separate instructions.
	loadIncN  func(a *Asm, ptr RegPtr, regs []Reg)
	loadDecN  func(a *Asm, ptr RegPtr, regs []Reg)
	storeIncN func(a *Asm, ptr RegPtr, regs []Reg)
	storeDecN func(a *Asm, ptr RegPtr, regs []Reg)

	// options is a map from optional CPU features to functions that test for them.
	// The test function should jump to label if the feature is available.
	options map[Option]func(a *Asm, label string)
}

// HasShiftWide reports whether the Arch has working LshWide/RshWide instructions.
// If not, calling them will panic.
func (a *Arch) HasShiftWide() bool {
	return a.lshd != ""
}

// A Hint is a hint about what a register will be used for,
// so that an appropriate one can be selected.
type Hint uint

const (
	HintNone       Hint = iota
	HintShiftCount      // shift count (CX on x86)
	HintMulSrc          // mul source operand (AX on x86)
	HintMulHi           // wide mul high output (DX on x86)
	HintMemOK           // a memory reference is okay
	HintCarry           // carry flag
	HintAltCarry        // secondary carry flag
)

// A Reg is an allocated register or other assembly operand.
// (For example, a constant might have name "$123"
// and a memory reference might have name "0(R8)".)
type Reg struct{ name string }

// IsImm reports whether r is an immediate value.
func (r Reg) IsImm() bool { return strings.HasPrefix(r.name, "$") }

// IsMem reports whether r is a memory value.
func (r Reg) IsMem() bool { return strings.HasSuffix(r.name, ")") }

// String returns the assembly syntax for r.
func (r Reg) String() string { return r.name }

// Valid reports whether is valid, meaning r is not the zero value of Reg (a register with no name).
func (r Reg) Valid() bool { return r.name != "" }

// A RegPtr is like a Reg but expected to hold a pointer.
// The separate Go type helps keeps pointers and scalars separate and avoid mistakes;
// it is okay to convert to Reg as needed to use specific routines.
type RegPtr struct{ name string }

// String returns the assembly syntax for r.
func (r RegPtr) String() string { return r.name }

// Valid reports whether is valid, meaning r is not the zero value of RegPtr (a register with no name).
func (r RegPtr) Valid() bool { return r.name != "" }

// mem returns a memory reference to off bytes from the pointer r.
func (r *RegPtr) mem(off int) Reg { return Reg{fmt.Sprintf("%d(%s)", off, r)} }

// A Carry is a flag field explaining how an instruction sets and uses the carry flags.
// Different operations expect different sets of bits.
// Add and Sub expect: UseCarry or 0, SetCarry, KeepCarry, or SmashCarry; and AltCarry or 0.
// ClearCarry, SaveCarry, and ConvertCarry expect: AddCarry or SubCarry; and AltCarry or 0.
type Carry uint

const (
	SetCarry   Carry = 1 << iota // sets carry
	UseCarry                     // uses carry
	KeepCarry                    // must preserve carry
	SmashCarry                   // can modify carry or not, whatever is easiest

	AltCarry // use the secondary carry flag
	AddCarry // use add carry flag semantics (for ClearCarry, ConvertCarry)
	SubCarry // use sub carry flag semantics (for ClearCarry, ConvertCarry)
)

// An Option denotes an optional CPU feature that can be tested at runtime.
type Option int

const (
	_ Option = iota

	// OptionAltCarry checks whether there is an add instruction
	// that uses a secondary carry flag, so that two different sums
	// can be accumulated in parallel with independent carry flags.
	// Some architectures (MIPS, Loong64, RISC-V) provide this
	// functionality natively, indicated by asm.Carry().Valid() being true.
	OptionAltCarry
)
