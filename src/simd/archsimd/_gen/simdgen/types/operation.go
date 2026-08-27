// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types

import (
	"fmt"
	"simd/archsimd/_gen/specgen/specexpr"
	"simd/archsimd/_gen/unify"
)

// RawOperation is the unifier representation of an [Operation]. It is
// translated into a more parsed form after unifier decoding.
type RawOperation struct {
	Go string // Base Go method name

	GOARCH       string  // GOARCH for this definition
	Asm          string  // Assembly mnemonic
	Arrangement  *string // optional Arrangement for ARM64 SIMD operations (e.g., "4S", "2D")
	OperandOrder *string // optional Operand order for better Go declarations
	// Optional tag to indicate this operation is paired with special generic->machine ssa lowering rules.
	// Should be paired with special templates in gen_simdrules.go
	SpecialLower *string
	// HiHalfAsm is the assembly mnemonic for the hi-half "2" variant of this operation,
	// specified in go_arm64.yaml (e.g., "VSHRN2", "VUMULL2").
	// When non-nil, simdgen generates the "2" variant machine op and folding rules.
	HiHalfAsm *string

	In              []Operand // Parameters
	InVariant       []Operand // Optional parameters
	Out             []Operand // Results
	MemFeatures     *string   // The memory operand feature this operation supports
	MemFeaturesData *string   // Additional data associated with MemFeatures
	Commutative     bool      // Commutativity
	CPUFeature      string    // CPUID/Has* feature name
	Zeroing         *bool     // nil => use asm suffix ".Z"; false => do not use asm suffix ".Z"
	Documentation   *string   // Documentation will be appended to the stubs comments.
	AddDoc          *string   // Additional doc to be appended.
	// ConstMask is a hack to reduce the size of defs the user writes for const-immediate
	// If present, it will be copied to [In[0].Const].
	ConstImm *string
	// NameAndSizeCheck is used to check [BWDQ] maps to (8|16|32|64) elemBits.
	NameAndSizeCheck *bool
	// If non-nil, all generation in gen_simdTypes.go and gen_intrinsics will be skipped.
	NoTypes *string
	// If non-nil, all generation in gen_simdGenericOps and gen_simdrules will be skipped.
	NoGenericOps *string
	// If non-nil, this string will be attached to the machine ssa op name.  E.g. "const"
	SSAVariant *string
	// If true, do not emit method declarations, generic ops, or intrinsics for masked variants
	// DO emit the architecture-specific opcodes and optimizations.
	HideMaskMethods *bool
	// WidthAgnostic marks an SVE bitwise operation whose unpredicated encoding
	// is written .D but computes the same bits under any element view. The
	// loader emits one def per element width so every Go type gets the API;
	// the unpredicated machine op collapses back to the single .D instruction
	// (see machineOpName), while the per-<T> predicated forms stay per width.
	WidthAgnostic *bool
}

// MaxVectorBits is the maximum vector length in bits Go currently supports (256
// bits / 32 bytes). It is used where a concrete upper bound is required for
// scalable SVE vectors (e.g., SSA vector types and buffer allocations).
const MaxVectorBits = 256

type Operand struct {
	Class string // One of "mask", "immediate", "vreg", "greg", and "mem"

	Go     *string // Go type of this operand
	AsmPos int     // Position of this operand in the assembly instruction

	Base     *string    // Base Go type ("int", "uint", "float")
	ElemBits *int       // Element bit width (omitted for greg)
	Bits     VectorSize // Total bit width, or scalable

	Const *string // Optional constant value for immediates.
	// Optional immediate arg offsets. If this field is non-nil,
	// This operand will be an immediate operand:
	// The compiler will right-shift the user-passed value by ImmOffset and set it as the AuxInt
	// field of the operation.
	ImmOffset *string
	ImmMax    *int    // optional maximum immediate, also highest case in immediate jump table
	Name      *string // optional name in the Go intrinsic declaration
	Lanes     *int    // Omitted for scalable
	// TreatLikeAScalarOfSize means only the lower $TreatLikeAScalarOfSize bits of the vector
	// is used, so at the API level we can make it just a scalar value of this size; Then we
	// can overwrite it to a vector of the right size during intrinsics stage.
	TreatLikeAScalarOfSize *int
	// If non-nil, it means the [Class] field is overwritten here, right now this is used to
	// overwrite the results of AVX2 compares to masks.
	OverwriteClass *string
	// If non-nil, it means the [Base] field is overwritten here. This field exist solely
	// because Intel's XED data is inconsistent. e.g. VANDNP[SD] marks its operand int.
	OverwriteBase *string
	// If non-nil, it means the [ElementBits] field is overwritten. This field exist solely
	// because Intel's XED data is inconsistent. e.g. AVX512 VPMADDUBSW marks its operand
	// elemBits 16, which should be 8.
	OverwriteElementBits *int
	// For greg only, specifically VPEXTR[BW], their results are specified by Intel as 32 bits,
	// but they really are 8/16 bits.
	OverwriteBits *int
	// FixedReg is the name of the fixed registers
	FixedReg *string
	// If non-nil, marks this vreg as a register list operand (for TBL/TBX).
	// Currently only list number 0 is supported (we might need to teach regalloc handle register lists
	// to support more than one register in the list).
	ListNumber *int
	// RegName is the assembly template's register symbol for this operand, e.g.
	// "Zdn", "Zn", "Pg" (SVE only). Comparing it across operands is how the
	// shape of an instruction is recognised: an input naming the same register
	// as the destination is written in place.
	RegName *string
	// PredRegName is the symbol this operand has in each of the operation's
	// predicated encodings, indexed to match InVariant (SVE only). It is nil
	// for an operation with no predicated encoding, and for every other target.
	PredRegName *[]string
	// Predication is the SVE governing-predicate qualifier, "M" (merging) or
	// "Z" (zeroing). It is set on mask operands of predicated encodings and
	// decides whether the generated machine op is the merging or the zeroing
	// form (see sveMaskSuffix).
	Predication *string
	// Governing marks the SVE governing predicate among an instruction's
	// operands — the one that selects which lanes the instruction acts on, as
	// opposed to a predicate it merely reads as data (SEL's <Pv>, the <Pn>/<Pm>
	// of a predicate-logical op).
	//
	// It is set only where the instruction has no unpredicated encoding, since
	// otherwise that encoding carries the operation and its predicated sibling's
	// predicate becomes an InVariant instead. So a governing predicate here is
	// always one the Go API hides: the generated method, generic op and
	// intrinsic omit it, and the lowering synthesizes an all-true predicate in
	// its place, which is how predicated-only instructions (ZCMPGT) expose an
	// unpredicated API. See #79781.
	//
	// This is independent of [Operand.Predication]: a governing predicate need
	// not carry a qualifier (SADDV <Dd>, <Pg>, <Zn>.<T> has no lanes to merge
	// into), and a qualified predicate need not be governing in this sense (the
	// InVariant of a paired operation is a real operand a peephole supplies).
	Governing *bool
}

// VectorSize is a unifier value that is either a number or the string "scalable".
type VectorSize struct {
	Scalable bool
	NRaw     int // Only meaningful if !Scalable
}

// N returns vs.NRaw, or panics if vs.Scalable.
func (vs VectorSize) N() int {
	if vs.Scalable {
		panic("cannot get bit width of scalable type")
	}
	return vs.NRaw
}

func (vs VectorSize) Num() specexpr.Num {
	if vs.Scalable {
		return specexpr.VW()
	}
	return specexpr.Int(vs.NRaw)
}

func (vs VectorSize) String() string {
	if vs.Scalable {
		return "scalable"
	}
	return fmt.Sprint(vs.N())
}

// IsGoverning reports whether this operand is the SVE governing predicate, and
// so is dropped from the API and filled with an all-true predicate at lowering.
func (o *Operand) IsGoverning() bool {
	return o.Governing != nil && *o.Governing
}

func (o Operand) OpName(s string) string {
	if n := o.Name; n != nil {
		return *n
	}
	if o.Class == "mask" {
		return "mask"
	}
	return s
}

func (o Operand) OpNameAndType(s string) string {
	return o.OpName(s) + " " + *o.Go
}

func (vs *VectorSize) DecodeUnified(v *unify.Value) error {
	var n int
	if err := v.Decode(&n); err == nil {
		*vs = VectorSize{false, n}
		return nil
	}

	var s string
	if err := v.Decode(&s); err == nil && s == "scalable" {
		*vs = VectorSize{true, -1}
		return nil
	}

	return fmt.Errorf("bits must be an integer or \"scalable\"")
}
