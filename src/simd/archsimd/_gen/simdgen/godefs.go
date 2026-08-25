// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"log"
	"math/rand/v2"
	"regexp"
	"slices"
	"strconv"
	"strings"
	"unicode"

	"simd/archsimd/_gen/gentools"
	"simd/archsimd/_gen/simdgen/types"
	"simd/archsimd/_gen/unify"
)

type rawOperation = types.RawOperation

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
	In []types.Operand
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

// hiHalfKind returns "narrow" or "long" based on whether the operation narrows or widens its elements.
// Returns "" if HiHalfAsm is nil or classification is ambiguous.
func (o *Operation) hiHalfKind() string {
	if o.HiHalfAsm == nil {
		return ""
	}
	// Find the first vreg input and the first vreg output to compare elemBits.
	var inElemBits, outElemBits *int
	for i := range o.In {
		if o.In[i].Class == "vreg" && o.In[i].ElemBits != nil {
			inElemBits = o.In[i].ElemBits
			break
		}
	}
	for i := range o.Out {
		if o.Out[i].Class == "vreg" && o.Out[i].ElemBits != nil {
			outElemBits = o.Out[i].ElemBits
			break
		}
	}
	if inElemBits == nil || outElemBits == nil {
		return ""
	}
	if *outElemBits < *inElemBits {
		return "narrow"
	}
	if *outElemBits > *inElemBits {
		return "long"
	}
	return ""
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

	// For operations that read only the lower half of input registers (indicated by hiHalfAsm),
	// add a doc note showing the compositional pattern for the upper half.
	if o.rawOperation.HiHalfAsm != nil && o.hiHalfKind() == "long" {
		// Count vector-register inputs (exclude immediates/scalars).
		vregIns := 0
		for _, in := range o.In {
			if in.Class == "vreg" {
				vregIns++
			}
		}
		// note this is arm64-specific
		switch vregIns {
		case 2:
			// Binary: MulLong, AddLong, SubLong, etc.
			o.Documentation += "\n// For the high-indexed elements, use HiToLo:\n//\n//\tx.HiToLo()." + o.Go + "(y.HiToLo())"
		case 1:
			// Unary: ShiftLeftLongConst, etc.
			o.Documentation += "\n// For the high-indexed elements, use HiToLo:\n//\n//\tx.HiToLo()." + o.Go + "(...)"
		}
	}

	// For down conversions, the high elements are zeroed if the result has more elements.
	// TODO: we should encode this logic in the YAML file, instead of hardcoding it here.
	if len(o.In) > 0 && len(o.Out) > 0 {
		inLanes := o.In[0].Lanes
		outLanes := o.Out[0].Lanes
		if inLanes != nil && outLanes != nil && *inLanes < *outLanes {
			if (strings.Contains(o.Go, "Saturate") || strings.Contains(o.Go, "TruncTo")) &&
				!strings.Contains(o.Go, "Concat") {
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

// sveArrangementLetter returns the SVE element-size arrangement letter
// (B=8, H=16, S=32, D=64) that names an SVE machine op, or "" when the target
// is not SVE. The letter comes from the operation's governing element width:
// the output vreg's elemBits, else the first vreg/mask operand's elemBits.
func sveArrangementLetter(gOp Operation) string {
	if !CurrentArch().isSVE() {
		return ""
	}
	elemBits := 0
	pick := func(ops []types.Operand) {
		if elemBits != 0 {
			return
		}
		for i := range ops {
			if c := ops[i].Class; (c == "vreg" || c == "mask") && ops[i].ElemBits != nil {
				elemBits = *ops[i].ElemBits
				return
			}
		}
	}
	pick(gOp.Out)
	pick(gOp.In)
	switch elemBits {
	case 8:
		return "B"
	case 16:
		return "H"
	case 32:
		return "S"
	case 64:
		return "D"
	}
	panic(fmt.Errorf("SVE op %s has no B/H/S/D element width (elemBits=%d)", gOp.Asm, elemBits))
}

func machineOpName(maskType maskShape, gOp Operation) string {
	asm := gOp.Asm
	if maskType == OneMask {
		asm += "Masked"
	}
	// For ARM64, use arrangement to create distinct SSA op names
	if letter := sveArrangementLetter(gOp); letter != "" {
		// SVE: scalable vectors have no fixed width, so distinguish machine ops
		// by element-size arrangement letter (B/H/S/D), e.g. ZADD -> ZADDB.
		asm += letter
	} else if gOp.Arrangement != nil && *gOp.Arrangement != "" {
		asm = fmt.Sprintf("%s%s", asm, *gOp.Arrangement)
	} else {
		asm = fmt.Sprintf("%s%d", asm, gOp.VectorWidth())
	}
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

func compareOperands(x, y *types.Operand) int {
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
		if c := compareIntPointers(x.ListNumber, y.ListNumber); c != 0 {
			return c
		}
		return 0
	}
}

// implicitPredCount reports whether the op has an implicit-all-true governing
// predicate input, as a count (0 or 1). An instruction has at most one governing
// predicate — the single mask input carrying a /Z or /M qualifier (see the
// role=="mask" operand in sve.buildOperandList) — which is a real machine-op
// input the lowering synthesizes as all-true but which is invisible in the Go
// API. So the generic op, intrinsic and stub size themselves by len(In) minus
// this. Source predicates (e.g. Pn, Pm in a predicate-logical op) are ordinary
// numbered inputs, not governing predicates, and are never counted.
func (op Operation) implicitPredCount() int {
	n := 0
	for i := range op.In {
		if op.In[i].IsImplicitAllTrue() {
			n++
		}
	}
	return n
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
			// "1" < "01".  Don't expect it in simdgen, but just in case.
			if ln1, ln2 := i-numStart1, j-numStart2; ln1 != ln2 {
				return ln1 - ln2
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

// generatedHeader returns the architecture-specific header for generated files.
func generatedHeader() string {
	return CurrentArch().GeneratedHeader
}

func writeGoDefs(cl unify.Closure) error {
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
		op.adjustAsm()
		ops = append(ops, op)
	}

	rand.Shuffle(len(ops), func(i, j int) {
		ops[i], ops[j] = ops[j], ops[i]
	})

	slices.SortFunc(ops, compareOperations)
	// The parsed XED data might contain duplicates, like
	// 512 bits VPADDP.
	deduped := dedup(ops)
	slices.SortFunc(deduped, compareOperations)

	if *Verbose {
		log.Printf("dedup len: %d, ops len: %d\n", len(deduped), len(ops))
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

	// Sorting again, just in case.
	slices.SortFunc(deduped, compareOperations)

	typeMap := parseSIMDTypes(deduped)

	archInfo := CurrentArch()
	// Generated files are named by GoTypeArch: the Go API files directly, the
	// backend files by SIMDTag. For amd64/arm64 these match the
	// GOARCH, so those filenames are unchanged; only SVE diverges (sve/SVE) so its
	// output sits alongside the NEON arm64 files instead of overwriting them.
	simdTag := archInfo.SIMDTag
	goTypeArch := archInfo.GoTypeArch
	archLower := archInfo.Arch

	var files gentools.Files
	defer files.FlushOrExit()

	writeSIMDTypes(files.NewGoFile(simdPackage+"/types_"+goTypeArch+".go"), typeMap)
	// TODO: Enable CPU feature generation for non-x86 architectures.
	if archLower == "amd64" {
		writeSIMDFeatures(files.NewGoFile(simdPackage+"/cpu.go"), deduped)
	}
	writeSIMDStubs(
		files.NewGoFile(simdPackage+"/ops_"+goTypeArch+".go"),
		files.NewGoFile(simdPackage+"/ops_internal_"+goTypeArch+".go"),
		deduped, typeMap, archLower == "amd64",
	)
	writeSIMDIntrinsics(files.NewGoFile("cmd/compile/internal/ssagen/simd"+simdTag+"intrinsics.go"), deduped, typeMap)
	const simdGenericOpsFile = "cmd/compile/internal/ssa/_gen/simdgenericOps.go"
	writeSIMDGenericOps(files.NewGoFile(simdGenericOpsFile), deduped, genFlags.InputPath(simdGenericOpsFile))
	writeSIMDMachineOps(files.NewGoFile("cmd/compile/internal/ssa/_gen/simd"+simdTag+"ops.go"), deduped)
	writeSIMDSSA(files.NewGoFile("cmd/compile/internal/"+archLower+"/"+archInfo.ssaGenFile()), deduped)
	writeSIMDRules(files.NewRawFile("cmd/compile/internal/ssa/_gen/simd"+simdTag+".rules"), deduped)

	return nil
}
