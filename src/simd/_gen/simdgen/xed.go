// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"cmp"
	"fmt"
	"log"
	"maps"
	"regexp"
	"slices"
	"strconv"
	"strings"

	"golang.org/x/arch/x86/xeddata"
	"gopkg.in/yaml.v3"
	"simd/_gen/unify"
)

const (
	NOT_REG_CLASS = 0 // not a register
	VREG_CLASS    = 1 // classify as a vector register; see
	GREG_CLASS    = 2 // classify as a general register
)

// instVariant is a bitmap indicating a variant of an instruction that has
// optional parameters.
type instVariant uint8

const (
	instVariantNone instVariant = 0

	// instVariantMasked indicates that this is the masked variant of an
	// optionally-masked instruction.
	instVariantMasked instVariant = 1 << iota
)

var operandRemarks int

// TODO: Doc. Returns Values with Def domains.
func loadXED(xedPath string) []*unify.Value {
	// TODO: Obviously a bunch more to do here.

	db, err := xeddata.NewDatabase(xedPath)
	if err != nil {
		log.Fatalf("open database: %v", err)
	}

	var defs []*unify.Value
	err = xeddata.WalkInsts(xedPath, func(inst *xeddata.Inst) {
		inst.Pattern = xeddata.ExpandStates(db, inst.Pattern)

		switch {
		case inst.RealOpcode == "N":
			return // Skip unstable instructions
		case !strings.HasPrefix(inst.Extension, "AVX"):
			// We're only interested in AVX instructions.
			return
		}

		if *flagDebugXED {
			fmt.Printf("%s:\n%+v\n", inst.Pos, inst)
		}

		ops, err := decodeOperands(db, strings.Fields(inst.Operands))
		if err != nil {
			operandRemarks++
			if *Verbose {
				log.Printf("%s: [%s] %s", inst.Pos, inst.Opcode(), err)
			}
			return
		}

		applyQuirks(inst, ops)

		defsPos := len(defs)
		defs = append(defs, instToUVal(inst, ops)...)

		if *flagDebugXED {
			for i := defsPos; i < len(defs); i++ {
				y, _ := yaml.Marshal(defs[i])
				fmt.Printf("==>\n%s\n", y)
			}
		}
	})
	if err != nil {
		log.Fatalf("walk insts: %v", err)
	}

	if len(unknownFeatures) > 0 {
		if !*Verbose {
			nInst := 0
			for _, insts := range unknownFeatures {
				nInst += len(insts)
			}
			log.Printf("%d unhandled CPU features for %d instructions (use -v for details)", len(unknownFeatures), nInst)
		} else {
			keys := slices.SortedFunc(maps.Keys(unknownFeatures), func(a, b cpuFeatureKey) int {
				return cmp.Or(cmp.Compare(a.Extension, b.Extension),
					cmp.Compare(a.ISASet, b.ISASet))
			})
			for _, key := range keys {
				if key.ISASet == "" || key.ISASet == key.Extension {
					log.Printf("unhandled Extension %s", key.Extension)
				} else {
					log.Printf("unhandled Extension %s and ISASet %s", key.Extension, key.ISASet)
				}
				log.Printf("  opcodes: %s", slices.Sorted(maps.Keys(unknownFeatures[key])))
			}
		}
	}

	return defs
}

var (
	maskRequiredRe = regexp.MustCompile(`VPCOMPRESS[BWDQ]|VCOMPRESSP[SD]|VPEXPAND[BWDQ]|VEXPANDP[SD]`)
	maskOptionalRe = regexp.MustCompile(`VPCMP(EQ|GT|U)?[BWDQ]|VCMPP[SD]`)
)

func applyQuirks(inst *xeddata.Inst, ops []operand) {
	opc := inst.Opcode()
	switch {
	case maskRequiredRe.MatchString(opc):
		// The mask on these instructions is marked optional, but the
		// instruction is pointless without the mask.
		for i, op := range ops {
			if op, ok := op.(operandMask); ok {
				op.optional = false
				ops[i] = op
			}
		}

	case maskOptionalRe.MatchString(opc):
		// Conversely, these masks should be marked optional and aren't.
		for i, op := range ops {
			if op, ok := op.(operandMask); ok && op.action.r {
				op.optional = true
				ops[i] = op
			}
		}
	}
}

type operandCommon struct {
	action operandAction
}

// operandAction defines whether this operand is read and/or written.
//
// TODO: Should this live in [xeddata.Operand]?
type operandAction struct {
	r  bool // Read
	w  bool // Written
	cr bool // Read is conditional (implies r==true)
	cw bool // Write is conditional (implies w==true)
}

type operandMem struct {
	operandCommon
	// TODO
}

type vecShape struct {
	elemBits int // Element size in bits
	bits     int // Register width in bits (total vector bits)
}

type operandVReg struct { // Vector register
	operandCommon
	vecShape
	elemBaseType scalarBaseType
}

type operandGReg struct { // Vector register
	operandCommon
	vecShape
	elemBaseType scalarBaseType
}

// operandMask is a vector mask.
//
// Regardless of the actual mask representation, the [vecShape] of this operand
// corresponds to the "bit for bit" type of mask. That is, elemBits gives the
// element width covered by each mask element, and bits/elemBits gives the total
// number of mask elements. (bits gives the total number of bits as if this were
// a bit-for-bit mask, which may be meaningless on its own.)
type operandMask struct {
	operandCommon
	vecShape
	// Bits in the mask is w/bits.

	allMasks bool // If set, size cannot be inferred because all operands are masks.

	// Mask can be omitted, in which case it defaults to K0/"no mask"
	optional bool
}

type operandImm struct {
	operandCommon
	bits int // Immediate size in bits
}

type operand interface {
	common() operandCommon
	addToDef(b *unify.DefBuilder)
}

func strVal(s any) *unify.Value {
	return unify.NewValue(unify.NewStringExact(fmt.Sprint(s)))
}

func (o operandCommon) common() operandCommon {
	return o
}

func (o operandMem) addToDef(b *unify.DefBuilder) {
	// TODO: w, base
	b.Add("class", strVal("memory"))
}

func (o operandVReg) addToDef(b *unify.DefBuilder) {
	baseDomain, err := unify.NewStringRegex(o.elemBaseType.regex())
	if err != nil {
		panic("parsing baseRe: " + err.Error())
	}
	b.Add("class", strVal("vreg"))
	b.Add("bits", strVal(o.bits))
	b.Add("base", unify.NewValue(baseDomain))
	// If elemBits == bits, then the vector can be ANY shape. This happens with,
	// for example, logical ops.
	if o.elemBits != o.bits {
		b.Add("elemBits", strVal(o.elemBits))
	}
}

func (o operandGReg) addToDef(b *unify.DefBuilder) {
	baseDomain, err := unify.NewStringRegex(o.elemBaseType.regex())
	if err != nil {
		panic("parsing baseRe: " + err.Error())
	}
	b.Add("class", strVal("greg"))
	b.Add("bits", strVal(o.bits))
	b.Add("base", unify.NewValue(baseDomain))
	if o.elemBits != o.bits {
		b.Add("elemBits", strVal(o.elemBits))
	}
}

func (o operandMask) addToDef(b *unify.DefBuilder) {
	b.Add("class", strVal("mask"))
	if o.allMasks {
		// If all operands are masks, omit sizes and let unification determine mask sizes.
		return
	}
	b.Add("elemBits", strVal(o.elemBits))
	b.Add("bits", strVal(o.bits))
}

func (o operandImm) addToDef(b *unify.DefBuilder) {
	b.Add("class", strVal("immediate"))
	b.Add("bits", strVal(o.bits))
}

var actionEncoding = map[string]operandAction{
	"r":   {r: true},
	"cr":  {r: true, cr: true},
	"w":   {w: true},
	"cw":  {w: true, cw: true},
	"rw":  {r: true, w: true},
	"crw": {r: true, w: true, cr: true},
	"rcw": {r: true, w: true, cw: true},
}

func decodeOperand(db *xeddata.Database, operand string) (operand, error) {
	op, err := xeddata.NewOperand(db, operand)
	if err != nil {
		log.Fatalf("parsing operand %q: %v", operand, err)
	}
	if *flagDebugXED {
		fmt.Printf("  %+v\n", op)
	}

	if strings.HasPrefix(op.Name, "EMX_BROADCAST") {
		// This refers to a set of macros defined in all-state.txt that set a
		// BCAST operand to various fixed values. But the BCAST operand is
		// itself suppressed and "internal", so I think we can just ignore this
		// operand.
		return nil, nil
	}

	// TODO: See xed_decoded_inst_operand_action. This might need to be more
	// complicated.
	action, ok := actionEncoding[op.Action]
	if !ok {
		return nil, fmt.Errorf("unknown action %q", op.Action)
	}
	common := operandCommon{action: action}

	lhs := op.NameLHS()
	if strings.HasPrefix(lhs, "MEM") {
		// TODO: Width, base type
		return operandMem{
			operandCommon: common,
		}, nil
	} else if strings.HasPrefix(lhs, "REG") {
		if op.Width == "mskw" {
			// The mask operand doesn't specify a width. We have to infer it.
			//
			// XED uses the marker ZEROSTR to indicate that a mask operand is
			// optional and, if omitted, implies K0, aka "no mask".
			return operandMask{
				operandCommon: common,
				optional:      op.Attributes["TXT=ZEROSTR"],
			}, nil
		} else {
			class, regBits := decodeReg(op)
			if class == NOT_REG_CLASS {
				return nil, fmt.Errorf("failed to decode register %q", operand)
			}
			baseType, elemBits, ok := decodeType(op)
			if !ok {
				return nil, fmt.Errorf("failed to decode register width %q", operand)
			}
			shape := vecShape{elemBits: elemBits, bits: regBits}
			if class == VREG_CLASS {
				return operandVReg{
					operandCommon: common,
					vecShape:      shape,
					elemBaseType:  baseType,
				}, nil
			}
			// general register
			m := min(shape.bits, shape.elemBits)
			shape.bits, shape.elemBits = m, m
			return operandGReg{
				operandCommon: common,
				vecShape:      shape,
				elemBaseType:  baseType,
			}, nil

		}
	} else if strings.HasPrefix(lhs, "IMM") {
		_, bits, ok := decodeType(op)
		if !ok {
			return nil, fmt.Errorf("failed to decode register width %q", operand)
		}
		return operandImm{
			operandCommon: common,
			bits:          bits,
		}, nil
	}

	// TODO: BASE and SEG
	return nil, fmt.Errorf("unknown operand LHS %q in %q", lhs, operand)
}

func decodeOperands(db *xeddata.Database, operands []string) (ops []operand, err error) {
	// Decode the XED operand descriptions.
	for _, o := range operands {
		op, err := decodeOperand(db, o)
		if err != nil {
			return nil, err
		}
		if op != nil {
			ops = append(ops, op)
		}
	}

	// XED doesn't encode the size of mask operands. If there are mask operands,
	// try to infer their sizes from other operands.
	if err := inferMaskSizes(ops); err != nil {
		return nil, fmt.Errorf("%w in operands %+v", err, operands)
	}

	return ops, nil
}

func inferMaskSizes(ops []operand) error {
	// This is a heuristic and it falls apart in some cases:
	//
	// - Mask operations like KAND[BWDQ] have *nothing* in the XED to indicate
	// mask size.
	//
	// - VINSERT*, VPSLL*, VPSRA*, and VPSRL* and some others naturally have
	// mixed input sizes and the XED doesn't indicate which operands the mask
	// applies to.
	//
	// - VPDP* and VP4DP* have really complex mixed operand patterns.
	//
	// I think for these we may just have to hand-write a table of which
	// operands each mask applies to.
	inferMask := func(r, w bool) error {
		var masks []int
		var rSizes, wSizes, sizes []vecShape
		allMasks := true
		hasWMask := false
		for i, op := range ops {
			action := op.common().action
			if _, ok := op.(operandMask); ok {
				if action.r && action.w {
					return fmt.Errorf("unexpected rw mask")
				}
				if action.r == r || action.w == w {
					masks = append(masks, i)
				}
				if action.w {
					hasWMask = true
				}
			} else {
				allMasks = false
				if reg, ok := op.(operandVReg); ok {
					if action.r {
						rSizes = append(rSizes, reg.vecShape)
					}
					if action.w {
						wSizes = append(wSizes, reg.vecShape)
					}
				}
			}
		}
		if len(masks) == 0 {
			return nil
		}

		if r {
			sizes = rSizes
			if len(sizes) == 0 {
				sizes = wSizes
			}
		}
		if w {
			sizes = wSizes
			if len(sizes) == 0 {
				sizes = rSizes
			}
		}

		if len(sizes) == 0 {
			// If all operands are masks, leave the mask inferrence to the users.
			if allMasks {
				for _, i := range masks {
					m := ops[i].(operandMask)
					m.allMasks = true
					ops[i] = m
				}
				return nil
			}
			return fmt.Errorf("cannot infer mask size: no register operands")
		}
		shape, ok := singular(sizes)
		if !ok {
			if !hasWMask && len(wSizes) == 1 && len(masks) == 1 {
				// This pattern looks like predicate mask, so its shape should align with the
				// output. TODO: verify this is a safe assumption.
				shape = wSizes[0]
			} else {
				return fmt.Errorf("cannot infer mask size: multiple register sizes %v", sizes)
			}
		}
		for _, i := range masks {
			m := ops[i].(operandMask)
			m.vecShape = shape
			ops[i] = m
		}
		return nil
	}
	if err := inferMask(true, false); err != nil {
		return err
	}
	if err := inferMask(false, true); err != nil {
		return err
	}
	return nil
}

// addOperandstoDef adds "in", "inVariant", and "out" to an instruction Def.
//
// Optional mask input operands are added to the inVariant field if
// variant&instVariantMasked, and omitted otherwise.
func addOperandsToDef(ops []operand, instDB *unify.DefBuilder, variant instVariant) {
	var inVals, inVar, outVals []*unify.Value
	asmPos := 0
	for _, op := range ops {
		var db unify.DefBuilder
		op.addToDef(&db)
		db.Add("asmPos", unify.NewValue(unify.NewStringExact(fmt.Sprint(asmPos))))

		action := op.common().action
		asmCount := 1 // # of assembly operands; 0 or 1
		if action.r {
			inVal := unify.NewValue(db.Build())
			// If this is an optional mask, put it in the input variant tuple.
			if mask, ok := op.(operandMask); ok && mask.optional {
				if variant&instVariantMasked != 0 {
					inVar = append(inVar, inVal)
				} else {
					// This operand doesn't appear in the assembly at all.
					asmCount = 0
				}
			} else {
				// Just a regular input operand.
				inVals = append(inVals, inVal)
			}
		}
		if action.w {
			outVal := unify.NewValue(db.Build())
			outVals = append(outVals, outVal)
		}

		asmPos += asmCount
	}

	instDB.Add("in", unify.NewValue(unify.NewTuple(inVals...)))
	instDB.Add("inVariant", unify.NewValue(unify.NewTuple(inVar...)))
	instDB.Add("out", unify.NewValue(unify.NewTuple(outVals...)))
}

func instToUVal(inst *xeddata.Inst, ops []operand) []*unify.Value {
	feature, ok := decodeCPUFeature(inst)
	if !ok {
		return nil
	}

	var vals []*unify.Value
	vals = append(vals, instToUVal1(inst, ops, feature, instVariantNone))
	if hasOptionalMask(ops) {
		vals = append(vals, instToUVal1(inst, ops, feature, instVariantMasked))
	}
	return vals
}

func instToUVal1(inst *xeddata.Inst, ops []operand, feature string, variant instVariant) *unify.Value {
	var db unify.DefBuilder
	db.Add("goarch", unify.NewValue(unify.NewStringExact("amd64")))
	db.Add("asm", unify.NewValue(unify.NewStringExact(inst.Opcode())))
	addOperandsToDef(ops, &db, variant)
	db.Add("cpuFeature", unify.NewValue(unify.NewStringExact(feature)))

	if strings.Contains(inst.Pattern, "ZEROING=0") {
		// This is an EVEX instruction, but the ".Z" (zero-merging)
		// instruction flag is NOT valid. EVEX.z must be zero.
		//
		// This can mean a few things:
		//
		// - The output of an instruction is a mask, so merging modes don't
		// make any sense. E.g., VCMPPS.
		//
		// - There are no masks involved anywhere. (Maybe MASK=0 is also set
		// in this case?) E.g., VINSERTPS.
		//
		// - The operation inherently performs merging. E.g., VCOMPRESSPS
		// with a mem operand.
		//
		// There may be other reasons.
		db.Add("zeroing", unify.NewValue(unify.NewStringExact("false")))
	}
	pos := unify.Pos{Path: inst.Pos.Path, Line: inst.Pos.Line}
	return unify.NewValuePos(db.Build(), pos)
}

// decodeCPUFeature returns the CPU feature name required by inst. These match
// the names of the "Has*" feature checks in the simd package.
func decodeCPUFeature(inst *xeddata.Inst) (string, bool) {
	key := cpuFeatureKey{
		Extension: inst.Extension,
		ISASet:    isaSetStrip.ReplaceAllLiteralString(inst.ISASet, ""),
	}
	feat, ok := cpuFeatureMap[key]
	if !ok {
		imap := unknownFeatures[key]
		if imap == nil {
			imap = make(map[string]struct{})
			unknownFeatures[key] = imap
		}
		imap[inst.Opcode()] = struct{}{}
		return "", false
	}
	if feat == "ignore" {
		return "", false
	}
	return feat, true
}

var isaSetStrip = regexp.MustCompile("_(128N?|256N?|512)$")

type cpuFeatureKey struct {
	Extension, ISASet string
}

// cpuFeatureMap maps from XED's "EXTENSION" and "ISA_SET" to a CPU feature name
// that can be used in the SIMD API.
var cpuFeatureMap = map[cpuFeatureKey]string{
	{"AVX", ""}:              "AVX",
	{"AVX_VNNI", "AVX_VNNI"}: "AVXVNNI",
	{"AVX2", ""}:             "AVX2",

	// AVX-512 foundational features. We combine all of these into one "AVX512" feature.
	{"AVX512EVEX", "AVX512F"}:  "AVX512",
	{"AVX512EVEX", "AVX512CD"}: "AVX512",
	{"AVX512EVEX", "AVX512BW"}: "AVX512",
	{"AVX512EVEX", "AVX512DQ"}: "AVX512",
	// AVX512VL doesn't appear explicitly in the ISASet. I guess it's implied by
	// the vector length suffix.

	// AVX-512 extension features
	{"AVX512EVEX", "AVX512_BITALG"}:    "AVX512BITALG",
	{"AVX512EVEX", "AVX512_GFNI"}:      "AVX512GFNI",
	{"AVX512EVEX", "AVX512_VBMI2"}:     "AVX512VBMI2",
	{"AVX512EVEX", "AVX512_VBMI"}:      "AVX512VBMI",
	{"AVX512EVEX", "AVX512_VNNI"}:      "AVX512VNNI",
	{"AVX512EVEX", "AVX512_VPOPCNTDQ"}: "AVX512VPOPCNTDQ",

	// AVX 10.2 (not yet supported)
	{"AVX512EVEX", "AVX10_2_RC"}: "ignore",
}

var unknownFeatures = map[cpuFeatureKey]map[string]struct{}{}

// hasOptionalMask returns whether there is an optional mask operand in ops.
func hasOptionalMask(ops []operand) bool {
	for _, op := range ops {
		if op, ok := op.(operandMask); ok && op.optional {
			return true
		}
	}
	return false
}

func singular[T comparable](xs []T) (T, bool) {
	if len(xs) == 0 {
		return *new(T), false
	}
	for _, x := range xs[1:] {
		if x != xs[0] {
			return *new(T), false
		}
	}
	return xs[0], true
}

// decodeReg returns class (NOT_REG_CLASS, VREG_CLASS, GREG_CLASS),
// and width in bits.  If the operand cannot be decided as a register,
// then the clas is NOT_REG_CLASS.
func decodeReg(op *xeddata.Operand) (class, width int) {
	// op.Width tells us the total width, e.g.,:
	//
	//    dq => 128 bits (XMM)
	//    qq => 256 bits (YMM)
	//    mskw => K
	//    z[iuf?](8|16|32|...) => 512 bits (ZMM)
	//
	// But the encoding is really weird and it's not clear if these *always*
	// mean XMM/YMM/ZMM or if other irregular things can use these large widths.
	// Hence, we dig into the register sets themselves.

	if !strings.HasPrefix(op.NameLHS(), "REG") {
		return NOT_REG_CLASS, 0
	}
	// TODO: We shouldn't be relying on the macro naming conventions. We should
	// use all-dec-patterns.txt, but xeddata doesn't support that table right now.
	rhs := op.NameRHS()
	if !strings.HasSuffix(rhs, "()") {
		return NOT_REG_CLASS, 0
	}
	switch {
	case strings.HasPrefix(rhs, "XMM_"):
		return VREG_CLASS, 128
	case strings.HasPrefix(rhs, "YMM_"):
		return VREG_CLASS, 256
	case strings.HasPrefix(rhs, "ZMM_"):
		return VREG_CLASS, 512
	case strings.HasPrefix(rhs, "GPR64_"), strings.HasPrefix(rhs, "VGPR64_"):
		return GREG_CLASS, 64
	case strings.HasPrefix(rhs, "GPR32_"), strings.HasPrefix(rhs, "VGPR32_"):
		return GREG_CLASS, 32
	}
	return NOT_REG_CLASS, 0
}

var xtypeRe = regexp.MustCompile(`^([iuf])([0-9]+)$`)

// scalarBaseType describes the base type of a scalar element. This is a Go
// type, but without the bit width suffix (with the exception of
// scalarBaseIntOrUint).
type scalarBaseType int

const (
	scalarBaseInt scalarBaseType = iota
	scalarBaseUint
	scalarBaseIntOrUint // Signed or unsigned is unspecified
	scalarBaseFloat
	scalarBaseComplex
	scalarBaseBFloat
	scalarBaseHFloat
)

func (s scalarBaseType) regex() string {
	switch s {
	case scalarBaseInt:
		return "int"
	case scalarBaseUint:
		return "uint"
	case scalarBaseIntOrUint:
		return "int|uint"
	case scalarBaseFloat:
		return "float"
	case scalarBaseComplex:
		return "complex"
	case scalarBaseBFloat:
		return "BFloat"
	case scalarBaseHFloat:
		return "HFloat"
	}
	panic(fmt.Sprintf("unknown scalar base type %d", s))
}

func decodeType(op *xeddata.Operand) (base scalarBaseType, bits int, ok bool) {
	// The xtype tells you the element type. i8, i16, i32, i64, f32, etc.
	//
	// TODO: Things like AVX2 VPAND have an xtype of u256 because they're
	// element-width agnostic. Do I map that to all widths, or just omit the
	// element width and let unification flesh it out? There's no u512
	// (presumably those are all masked, so elem width matters). These are all
	// Category: LOGICAL, so maybe we could use that info?

	// Handle some weird ones.
	switch op.Xtype {
	// 8-bit float formats as defined by Open Compute Project "OCP 8-bit
	// Floating Point Specification (OFP8)".
	case "bf8": // E5M2 float
		return scalarBaseBFloat, 8, true
	case "hf8": // E4M3 float
		return scalarBaseHFloat, 8, true
	case "bf16": // bfloat16 float
		return scalarBaseBFloat, 16, true
	case "2f16":
		// Complex consisting of 2 float16s. Doesn't exist in Go, but we can say
		// what it would be.
		return scalarBaseComplex, 32, true
	case "2i8", "2I8":
		// These just use the lower INT8 in each 16 bit field.
		// As far as I can tell, "2I8" is a typo.
		return scalarBaseInt, 8, true
	case "2u16", "2U16":
		// some VPDP* has it
		// TODO: does "z" means it has zeroing?
		return scalarBaseUint, 16, true
	case "2i16", "2I16":
		// some VPDP* has it
		return scalarBaseInt, 16, true
	case "4u8", "4U8":
		// some VPDP* has it
		return scalarBaseUint, 8, true
	case "4i8", "4I8":
		// some VPDP* has it
		return scalarBaseInt, 8, true
	}

	// The rest follow a simple pattern.
	m := xtypeRe.FindStringSubmatch(op.Xtype)
	if m == nil {
		// TODO: Report unrecognized xtype
		return 0, 0, false
	}
	bits, _ = strconv.Atoi(m[2])
	switch m[1] {
	case "i", "u":
		// XED is rather inconsistent about what's signed, unsigned, or doesn't
		// matter, so merge them together and let the Go definitions narrow as
		// appropriate. Maybe there's a better way to do this.
		return scalarBaseIntOrUint, bits, true
	case "f":
		return scalarBaseFloat, bits, true
	default:
		panic("unreachable")
	}
}
