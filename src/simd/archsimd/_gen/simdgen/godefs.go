// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"log"
	"regexp"
	"slices"
	"strconv"
	"strings"
	"unicode"

	"simd/archsimd/_gen/unify"
)

type Operation struct {
	rawOperation

	// Go is the Go method name of this operation.
	//
	// It is derived from the raw Go method name by adding optional suffixes.
	// Currently, "Masked" is the only suffix.
	Go string

	// Documentation is the doc string for this API.
	//
	// It is computed from the raw documentation:
	//
	// - "NAME" is replaced by the Go method name.
	//
	// - For masked operation, a sentence about masking is added.
	Documentation string

	// In is the sequence of parameters to the Go method.
	//
	// For masked operations, this will have the mask operand appended.
	In []Operand
}

// rawOperation is the unifier representation of an [Operation]. It is
// translated into a more parsed form after unifier decoding.
type rawOperation struct {
	Go string // Base Go method name

	GoArch       string  // GOARCH for this definition
	Asm          string  // Assembly mnemonic
	OperandOrder *string // optional Operand order for better Go declarations
	// Optional tag to indicate this operation is paired with special generic->machine ssa lowering rules.
	// Should be paired with special templates in gen_simdrules.go
	SpecialLower *string

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
}

func (o *Operation) IsMasked() bool {
	if len(o.InVariant) == 0 {
		return false
	}
	if len(o.InVariant) == 1 && o.InVariant[0].Class == "mask" {
		return true
	}
	panic(fmt.Errorf("unknown inVariant"))
}

func (o *Operation) SkipMaskedMethod() bool {
	if o.HideMaskMethods == nil {
		return false
	}
	if *o.HideMaskMethods && o.IsMasked() {
		return true
	}
	return false
}

var reForName = regexp.MustCompile(`\bNAME\b`)

func (o *Operation) DecodeUnified(v *unify.Value) error {
	if err := v.Decode(&o.rawOperation); err != nil {
		return err
	}

	isMasked := o.IsMasked()

	// Compute full Go method name.
	o.Go = o.rawOperation.Go
	if isMasked {
		o.Go += "Masked"
	}

	// Compute doc string.
	if o.rawOperation.Documentation != nil {
		o.Documentation = *o.rawOperation.Documentation
	} else {
		o.Documentation = "// UNDOCUMENTED"
	}
	o.Documentation = reForName.ReplaceAllString(o.Documentation, o.Go)
	if isMasked {
		o.Documentation += "\n//\n// This operation is applied selectively under a write mask."
		// Suppress generic op and method declaration for exported methods, if a mask is present.
		if unicode.IsUpper([]rune(o.Go)[0]) {
			trueVal := "true"
			o.NoGenericOps = &trueVal
			o.NoTypes = &trueVal
		}
	}
	if o.rawOperation.AddDoc != nil {
		o.Documentation += "\n" + reForName.ReplaceAllString(*o.rawOperation.AddDoc, o.Go)
	}

	o.In = append(o.rawOperation.In, o.rawOperation.InVariant...)

	// For down conversions, the high elements are zeroed if the result has more elements.
	// TODO: we should encode this logic in the YAML file, instead of hardcoding it here.
	if len(o.In) > 0 && len(o.Out) > 0 {
		inLanes := o.In[0].Lanes
		outLanes := o.Out[0].Lanes
		if inLanes != nil && outLanes != nil && *inLanes < *outLanes {
			if (strings.Contains(o.Go, "Saturate") || strings.Contains(o.Go, "Truncate")) &&
				!strings.HasSuffix(o.Go, "Concat") {
				o.Documentation += "\n// Results are packed to low elements in the returned vector, its upper elements are zeroed."
			}
		}
	}

	return nil
}

func (o *Operation) VectorWidth() int {
	out := o.Out[0]
	if out.Class == "vreg" {
		return *out.Bits
	} else if out.Class == "greg" || out.Class == "mask" {
		for i := range o.In {
			if o.In[i].Class == "vreg" {
				return *o.In[i].Bits
			}
		}
	}
	panic(fmt.Errorf("Figure out what the vector width is for %v and implement it", *o))
}

// Right now simdgen computes the machine op name for most instructions
// as $Name$OutputSize, by this denotation, these instructions are "overloaded".
// for example:
// (Uint16x8) ConvertToInt8
// (Uint16x16) ConvertToInt8
// are both VPMOVWB128.
// To make them distinguishable we need to append the input size to them as well.
// TODO: document them well in the generated code.
var demotingConvertOps = map[string]bool{
	"VPMOVQD128": true, "VPMOVSQD128": true, "VPMOVUSQD128": true, "VPMOVQW128": true, "VPMOVSQW128": true,
	"VPMOVUSQW128": true, "VPMOVDW128": true, "VPMOVSDW128": true, "VPMOVUSDW128": true, "VPMOVQB128": true,
	"VPMOVSQB128": true, "VPMOVUSQB128": true, "VPMOVDB128": true, "VPMOVSDB128": true, "VPMOVUSDB128": true,
	"VPMOVWB128": true, "VPMOVSWB128": true, "VPMOVUSWB128": true,
	"VPMOVQDMasked128": true, "VPMOVSQDMasked128": true, "VPMOVUSQDMasked128": true, "VPMOVQWMasked128": true, "VPMOVSQWMasked128": true,
	"VPMOVUSQWMasked128": true, "VPMOVDWMasked128": true, "VPMOVSDWMasked128": true, "VPMOVUSDWMasked128": true, "VPMOVQBMasked128": true,
	"VPMOVSQBMasked128": true, "VPMOVUSQBMasked128": true, "VPMOVDBMasked128": true, "VPMOVSDBMasked128": true, "VPMOVUSDBMasked128": true,
	"VPMOVWBMasked128": true, "VPMOVSWBMasked128": true, "VPMOVUSWBMasked128": true,
}

func machineOpName(maskType maskShape, gOp Operation) string {
	asm := gOp.Asm
	if maskType == OneMask {
		asm += "Masked"
	}
	asm = fmt.Sprintf("%s%d", asm, gOp.VectorWidth())
	if gOp.SSAVariant != nil {
		asm += *gOp.SSAVariant
	}
	if demotingConvertOps[asm] {
		// Need to append the size of the source as well.
		// TODO: should be "%sto%d".
		asm = fmt.Sprintf("%s_%d", asm, *gOp.In[0].Bits)
	}
	return asm
}

func compareStringPointers(x, y *string) int {
	if x != nil && y != nil {
		return compareNatural(*x, *y)
	}
	if x == nil && y == nil {
		return 0
	}
	if x == nil {
		return -1
	}
	return 1
}

func compareIntPointers(x, y *int) int {
	if x != nil && y != nil {
		return *x - *y
	}
	if x == nil && y == nil {
		return 0
	}
	if x == nil {
		return -1
	}
	return 1
}

func compareOperations(x, y Operation) int {
	if c := compareNatural(x.Go, y.Go); c != 0 {
		return c
	}
	xIn, yIn := x.In, y.In

	if len(xIn) > len(yIn) && xIn[len(xIn)-1].Class == "mask" {
		xIn = xIn[:len(xIn)-1]
	} else if len(xIn) < len(yIn) && yIn[len(yIn)-1].Class == "mask" {
		yIn = yIn[:len(yIn)-1]
	}

	if len(xIn) < len(yIn) {
		return -1
	}
	if len(xIn) > len(yIn) {
		return 1
	}
	if len(x.Out) < len(y.Out) {
		return -1
	}
	if len(x.Out) > len(y.Out) {
		return 1
	}
	for i := range xIn {
		ox, oy := &xIn[i], &yIn[i]
		if c := compareOperands(ox, oy); c != 0 {
			return c
		}
	}
	return 0
}

func compareOperands(x, y *Operand) int {
	if c := compareNatural(x.Class, y.Class); c != 0 {
		return c
	}
	if x.Class == "immediate" {
		return compareStringPointers(x.ImmOffset, y.ImmOffset)
	} else {
		if c := compareStringPointers(x.Base, y.Base); c != 0 {
			return c
		}
		if c := compareIntPointers(x.ElemBits, y.ElemBits); c != 0 {
			return c
		}
		if c := compareIntPointers(x.Bits, y.Bits); c != 0 {
			return c
		}
		return 0
	}
}

type Operand struct {
	Class string // One of "mask", "immediate", "vreg", "greg", and "mem"

	Go     *string // Go type of this operand
	AsmPos int     // Position of this operand in the assembly instruction

	Base     *string // Base Go type ("int", "uint", "float")
	ElemBits *int    // Element bit width
	Bits     *int    // Total vector bit width

	Const *string // Optional constant value for immediates.
	// Optional immediate arg offsets. If this field is non-nil,
	// This operand will be an immediate operand:
	// The compiler will right-shift the user-passed value by ImmOffset and set it as the AuxInt
	// field of the operation.
	ImmOffset *string
	Name      *string // optional name in the Go intrinsic declaration
	Lanes     *int    // *Lanes equals Bits/ElemBits except for scalars, when *Lanes == 1
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
	// FixedReg is the name of the fixed registers
	FixedReg *string
}

// isDigit returns true if the byte is an ASCII digit.
func isDigit(b byte) bool {
	return b >= '0' && b <= '9'
}

// compareNatural performs a "natural sort" comparison of two strings.
// It compares non-digit sections lexicographically and digit sections
// numerically.  In the case of string-unequal "equal" strings like
// "a01b" and "a1b", strings.Compare breaks the tie.
//
// It returns:
//
//	-1 if s1 < s2
//	 0 if s1 == s2
//	+1 if s1 > s2
func compareNatural(s1, s2 string) int {
	i, j := 0, 0
	len1, len2 := len(s1), len(s2)

	for i < len1 && j < len2 {
		// Find a non-digit segment or a number segment in both strings.
		if isDigit(s1[i]) && isDigit(s2[j]) {
			// Number segment comparison.
			numStart1 := i
			for i < len1 && isDigit(s1[i]) {
				i++
			}
			num1, _ := strconv.Atoi(s1[numStart1:i])

			numStart2 := j
			for j < len2 && isDigit(s2[j]) {
				j++
			}
			num2, _ := strconv.Atoi(s2[numStart2:j])

			if num1 < num2 {
				return -1
			}
			if num1 > num2 {
				return 1
			}
			// If numbers are equal, continue to the next segment.
		} else {
			// Non-digit comparison.
			if s1[i] < s2[j] {
				return -1
			}
			if s1[i] > s2[j] {
				return 1
			}
			i++
			j++
		}
	}

	// deal with a01b vs a1b; there needs to be an order.
	return strings.Compare(s1, s2)
}

const generatedHeader = `// Code generated by x/arch/internal/simdgen using 'go run . -xedPath $XED_PATH -o godefs -goroot $GOROOT go.yaml types.yaml categories.yaml'; DO NOT EDIT.
`

func writeGoDefs(path string, cl unify.Closure) error {
	// TODO: Merge operations with the same signature but multiple
	// implementations (e.g., SSE vs AVX)
	var ops []Operation
	for def := range cl.All() {
		var op Operation
		if !def.Exact() {
			continue
		}
		if err := def.Decode(&op); err != nil {
			log.Println(err.Error())
			log.Println(def)
			continue
		}
		// TODO: verify that this is safe.
		op.sortOperand()
		op.adjustAsm()
		ops = append(ops, op)
	}
	slices.SortFunc(ops, compareOperations)
	// The parsed XED data might contain duplicates, like
	// 512 bits VPADDP.
	deduped := dedup(ops)
	slices.SortFunc(deduped, compareOperations)

	if *Verbose {
		log.Printf("dedup len: %d\n", len(ops))
	}
	var err error
	if err = overwrite(deduped); err != nil {
		return err
	}
	if *Verbose {
		log.Printf("dedup len: %d\n", len(deduped))
	}
	if !*FlagNoDedup {
		// TODO: This can hide mistakes in the API definitions, especially when
		// multiple patterns result in the same API unintentionally. Make it stricter.
		if deduped, err = dedupGodef(deduped); err != nil {
			return err
		}
	}
	if *Verbose {
		log.Printf("dedup len: %d\n", len(deduped))
	}
	if !*FlagNoConstImmPorting {
		if err = copyConstImm(deduped); err != nil {
			return err
		}
	}
	if *Verbose {
		log.Printf("dedup len: %d\n", len(deduped))
	}
	reportXEDInconsistency(deduped)
	typeMap := parseSIMDTypes(deduped)

	formatWriteAndClose(writeSIMDTypes(typeMap), path, "src/"+simdPackage+"/types_amd64.go")
	formatWriteAndClose(writeSIMDFeatures(deduped), path, "src/"+simdPackage+"/cpu.go")
	f, fI := writeSIMDStubs(deduped, typeMap)
	formatWriteAndClose(f, path, "src/"+simdPackage+"/ops_amd64.go")
	formatWriteAndClose(fI, path, "src/"+simdPackage+"/ops_internal_amd64.go")
	formatWriteAndClose(writeSIMDIntrinsics(deduped, typeMap), path, "src/cmd/compile/internal/ssagen/simdintrinsics.go")
	formatWriteAndClose(writeSIMDGenericOps(deduped), path, "src/cmd/compile/internal/ssa/_gen/simdgenericOps.go")
	formatWriteAndClose(writeSIMDMachineOps(deduped), path, "src/cmd/compile/internal/ssa/_gen/simdAMD64ops.go")
	formatWriteAndClose(writeSIMDSSA(deduped), path, "src/cmd/compile/internal/amd64/simdssa.go")
	writeAndClose(writeSIMDRules(deduped).Bytes(), path, "src/cmd/compile/internal/ssa/_gen/simdAMD64.rules")

	return nil
}
