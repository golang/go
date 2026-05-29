// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bytes"
	"fmt"
	"slices"
	"strings"
	"text/template"
	"unicode"

	"_gen/sgutil"
)

type tplRuleData struct {
	TplName        string // e.g. "sftimm"
	GoOp           string // e.g. "ShiftAllLeft"
	GoType         string // e.g. "Uint32x8"
	Args           string // e.g. "x y"
	Asm            string // e.g. "VPSLLD256"
	ArgsOut        string // e.g. "x y"
	MaskInConvert  string // e.g. "VPMOVVec32x8ToM"
	MaskOutConvert string // e.g. "VPMOVMToVec32x8"
	ElementSize    int    // e.g. 32
	Size           int    // e.g. 128
	ArgsLoadAddr   string // [Args] with its last vreg arg being a concrete "(VMOVDQUload* ptr mem)", and might contain mask.
	ArgsAddr       string // [Args] with its last vreg arg being replaced by "ptr", and might contain mask, and with a "mem" at the end.
	FeatCheck      string // e.g. "v.Block.CPUfeatures.hasFeature(CPUavx512)" -- for a ssa/_gen rules file.
	RuleCond       string // e.g. "a==0" -- condition for asmRule or argsMatchRule.
	RuleOut        string // e.g. "y" -- output of an asmRule or argsMatchRule.
	RuleArgs       string // custom args pattern for argsMatchRule.
}

// Helper type to make template map initialization less repetitive
// (and also remove a chance for errors.)
type ruleTemplateMap struct {
	sgutil.InsertMap[string, *template.Template]
}

// Add creates a template named "name" after appending "// {{.TmplName}}\n" to the
// template, and returns the input so that additions may be chained.
// This helps make template initialization easy to order and easy to read.
func (rtm *ruleTemplateMap) Add(name string, templ string) *ruleTemplateMap {
	// Append debugging comment AND end of line
	templ += " // {{.TplName}}\n"
	ct := sgutil.TemplateNamed(name, templ)
	rtm.InsertMap.Put(name, ct)
	return rtm
}

var (
	// ORDER MATTERS.  These should appear in most-to-least-specific order
	// TODO: vregMemFeatCheck is not necessarily in the right place; there was an order,
	// this was copied from it, but vregMemFeatCheck was not in it.  It's not clear
	// it was ever used.
	// It's also not clear that this is strictly most-to-least-specific?
	ruleTemplates = new(ruleTemplateMap).
		Add("masksftimm", `({{.Asm}} x (MOVQconst [c]) mask) => ({{.Asm}}const [amd64CapAVXShift(c)] x mask)`).
		Add("sftimm", `({{.Asm}} x (MOVQconst [c])) => ({{.Asm}}const [amd64CapAVXShift(c)] x)`).
		Add("maskInMaskOut", `({{.GoOp}}{{.GoType}} {{.Args}} mask) => ({{.MaskOutConvert}} ({{.Asm}} {{.ArgsOut}} ({{.MaskInConvert}} <types.TypeMask> mask)))`).
		Add("maskOut", `({{.GoOp}}{{.GoType}} {{.Args}}) => ({{.MaskOutConvert}} ({{.Asm}} {{.ArgsOut}}))`).
		Add("maskIn", `({{.GoOp}}{{.GoType}} {{.Args}} mask) => ({{.Asm}} {{.ArgsOut}} ({{.MaskInConvert}} <types.TypeMask> mask))`).
		Add("pureVreg", `({{.GoOp}}{{.GoType}} {{.Args}}) => ({{.Asm}} {{.ArgsOut}})`).
		Add("vregMem", `({{.Asm}} {{.ArgsLoadAddr}}) && canMergeLoad(v, l) && clobber(l) => ({{.Asm}}load {{.ArgsAddr}})`).
		Add("vregMemFeatCheck", `({{.Asm}} {{.ArgsLoadAddr}}) && {{.FeatCheck}} && canMergeLoad(v, l) && clobber(l) => ({{.Asm}}load {{.ArgsAddr}})`).
		Add("asmRule", `({{.Asm}} {{.Args}}) {{.RuleCond}} => {{.RuleOut}}`).
		Add("argsMatchRule", `({{.Asm}} {{.RuleArgs}}) {{.RuleCond}} => {{.RuleOut}}`).
		Add("earlyMatchRule", `({{.GoOp}}{{.GoType}} {{.RuleArgs}}) {{.RuleCond}} => {{.RuleOut}}`)
)

func (d tplRuleData) MaskOptimization(asmCheck map[string]bool) string {
	asmNoMask := d.Asm
	if i := strings.Index(asmNoMask, "Masked"); i == -1 {
		return ""
	}
	asmNoMask = strings.ReplaceAll(asmNoMask, "Masked", "")
	if asmCheck[asmNoMask] == false {
		return ""
	}

	for _, nope := range []string{"VMOVDQU", "VPCOMPRESS", "VCOMPRESS", "VPEXPAND", "VEXPAND", "VPBLENDM", "VMOVUP"} {
		if strings.HasPrefix(asmNoMask, nope) {
			return ""
		}
	}

	size := asmNoMask[len(asmNoMask)-3:]
	if strings.HasSuffix(asmNoMask, "const") {
		sufLen := len("128const")
		size = asmNoMask[len(asmNoMask)-sufLen:][:3]
	}
	switch size {
	case "128", "256", "512":
	default:
		panic("Unexpected operation size on " + d.Asm)
	}

	switch d.ElementSize {
	case 8, 16, 32, 64:
	default:
		panic(fmt.Errorf("Unexpected operation width %d on %v", d.ElementSize, d.Asm))
	}

	return fmt.Sprintf("(VMOVDQU%dMasked%s (%s %s) mask) => (%s %s mask)\n", d.ElementSize, size, asmNoMask, d.Args, d.Asm, d.Args)
}

func compareTplRuleData(x, y tplRuleData) int {
	if c := compareNatural(x.GoOp, y.GoOp); c != 0 {
		return c
	}
	if c := compareNatural(x.GoType, y.GoType); c != 0 {
		return c
	}
	if c := compareNatural(x.Args, y.Args); c != 0 {
		return c
	}
	if x.TplName == y.TplName {
		return 0
	}
	return ruleTemplates.Compare(x.TplName, y.TplName)
}

// parseAsmRule tries to parse given string as it would be asmRule:
// if <cond> => <out>
// Return false, "", "" if not matched, otherwise true, condition, rule output.
// For example:
// rule:"if a==0 => (VADD4S x y)" can be used to provide addional
// lowering for an instruction which doesn't support zero immediate encoding:
// (VUSRA4S [a] x y) && a==0 => (VADD4S x y)
func parseAsmRule(rule string) (bool, string, string) {
	arrowIndex := strings.Index(rule, "=>")
	if arrowIndex == -1 {
		return false, "", ""
	}

	condPart := rule[:arrowIndex]
	outPart := rule[arrowIndex+len("=>"):]

	// Check if condPart starts with "if" followed by at least one space.
	cond := strings.TrimPrefix(condPart, "if")
	if cond == condPart || len(cond) == 0 || !unicode.IsSpace(rune(cond[0])) {
		return false, "", ""
	}

	// Trim any spaces around <cond> and <out>.
	cond = strings.TrimSpace(cond)
	out := strings.TrimSpace(outPart)
	if cond == "" || out == "" {
		return false, "", ""
	}

	return true, cond, out
}

// parseArgsMatchRule tries to parse given string as it would be asmRule with custom arguments to match:
// match <args> [&& <cond>] => <out>
// earlymatch <args> [&& <cond>] => <out>
// Return false, "", "", "", false if not matched, otherwise true, args, cond, rule output, isEarly.
// For example:
// rule:"match [0] (VMOV%sins [0] _ (MOVDconst [c])) && uint64(c)<= 255 => (VMOVI%a [uint8(c)])" can be used to provide addional
// lowering for a broadcast to use immediate source:
// (VDUPSbcast [0] (VMOVSins [0] _ (MOVDconst [c]))) && uint64(c)<= 255 => (VMOVI4S [uint8(c)])
// The specifiers currently supported are only arm64-specific, it may be generalized in the expandFormatSpecifiers function in future.
// The "earlymatch" variant uses GoOp instead of Asm on the left-hand side, replacing the default lowering rule.
func parseArgsMatchRule(rule string) (bool, string, string, string, bool) {
	isEarly := false
	rest := ""
	if strings.HasPrefix(rule, "earlymatch ") {
		isEarly = true
		rest = rule[len("earlymatch "):]
	} else if strings.HasPrefix(rule, "match ") {
		rest = rule[len("match "):]
	} else {
		return false, "", "", "", false
	}

	arrowIndex := strings.Index(rest, "=>")
	if arrowIndex == -1 {
		return false, "", "", "", false
	}

	condPart := rest[:arrowIndex]
	outPart := rest[arrowIndex+len("=>"):]

	// Split args and optional condition on "&&".
	args := condPart
	cond := ""
	if andIndex := strings.Index(condPart, "&&"); andIndex != -1 {
		args = strings.TrimSpace(condPart[:andIndex])
		cond = strings.TrimSpace(condPart[andIndex+len("&&"):])
	} else {
		args = strings.TrimSpace(args)
	}

	out := strings.TrimSpace(outPart)
	if args == "" || out == "" {
		return false, "", "", "", false
	}

	return true, args, cond, out, isEarly
}

// expandFormatSpecifiers replaces format specifiers in s with concrete values
// derived from elemBits (the element size in bits for the current vector type).
//
// Supported specifiers:
//
//	%s - lane size letter (B, H, S, D)
//	%a - arrangement suffix (16B, 8H, 4S, 2D) for neon instructions
//	%b - bits per lane (8, 16, 32, 64)
func expandFormatSpecifiers(s string, elemBits int) string {
	elemLetters := map[int]string{8: "B", 16: "H", 32: "S", 64: "D"}
	arrangements := map[int]string{8: "16B", 16: "8H", 32: "4S", 64: "2D"}
	s = strings.ReplaceAll(s, "%s", elemLetters[elemBits])
	s = strings.ReplaceAll(s, "%a", arrangements[elemBits])
	s = strings.ReplaceAll(s, "%b", fmt.Sprintf("%d", elemBits))
	return s
}

// writeSIMDRules generates the lowering and rewrite rules for ssa and writes it to simdAMD64.rules
// within the specified directory.
func writeSIMDRules(ops []Operation) *bytes.Buffer {
	buffer := new(bytes.Buffer)
	buffer.WriteString(generatedHeader() + "\n")

	// asm -> masked merging rules
	maskedMergeOpts := make(map[string]string)
	s2n := map[int]string{8: "B", 16: "W", 32: "D", 64: "Q"}
	asmCheck := map[string]bool{}    // for masked merge optimizations.
	sftimmCheck := map[string]bool{} // deduplicate sftimm rules
	var allData []tplRuleData
	var optData []tplRuleData    // for mask peephole optimizations, and other misc
	var memOptData []tplRuleData // for memory peephole optimizations
	memOpSeen := make(map[string]bool)
	ruleDone := make(map[string]struct{})

	for _, opr := range ops {
		opInShape, opOutShape, maskType, immType, gOp, _ := opr.shape()
		asm := machineOpName(maskType, gOp)
		vregInCnt := len(gOp.In)
		if maskType == OneMask {
			vregInCnt--
		}

		data := tplRuleData{
			GoOp: gOp.Go,
			Asm:  asm,
		}

		if vregInCnt == 1 {
			data.Args = "x"
			data.ArgsOut = data.Args
		} else if vregInCnt == 2 {
			data.Args = "x y"
			data.ArgsOut = data.Args
		} else if vregInCnt == 3 {
			data.Args = "x y z"
			data.ArgsOut = data.Args
		} else {
			panic(fmt.Errorf("simdgen does not support more than 3 vreg in inputs"))
		}
		if immType == ConstImm {
			data.ArgsOut = fmt.Sprintf("[%s] %s", *opr.In[0].Const, data.ArgsOut)
		} else if immType == VarImm || immType == VarImmLim {
			data.Args = fmt.Sprintf("[a] %s", data.Args)
			data.ArgsOut = fmt.Sprintf("[a] %s", data.ArgsOut)
		} else if immType == ConstVarImm {
			data.Args = fmt.Sprintf("[a] %s", data.Args)
			data.ArgsOut = fmt.Sprintf("[a+%s] %s", *opr.In[0].Const, data.ArgsOut)
		}

		goType := func(op Operation) string {
			if op.OperandOrder != nil {
				switch *op.OperandOrder {
				case "21Type1", "231Type1":
					// Permute uses operand[1] for method receiver.
					return *op.In[1].Go
				}
			}
			return *op.In[0].Go
		}
		var tplName string
		// If class overwrite is happening, that's not really a mask but a vreg.
		if opOutShape == OneVregOut || opOutShape == OneVregOutAtIn || opOutShape == OneVregOutScalar || gOp.Out[0].OverwriteClass != nil {
			switch opInShape {
			case OneImmIn:
				tplName = "pureVreg"
				data.GoType = goType(gOp)
			case PureVregIn, VlistIn:
				tplName = "pureVreg"
				data.GoType = goType(gOp)
			case OneKmaskImmIn:
				fallthrough
			case OneKmaskIn:
				tplName = "maskIn"
				data.GoType = goType(gOp)
				rearIdx := len(gOp.In) - 1
				// Mask is at the end.
				width := *gOp.In[rearIdx].ElemBits
				data.MaskInConvert = fmt.Sprintf("VPMOVVec%dx%dToM", width, *gOp.In[rearIdx].Lanes)
				data.ElementSize = width
			case PureKmaskIn:
				panic(fmt.Errorf("simdgen does not support pure k mask instructions, they should be generated by compiler optimizations"))
			}
		} else if opOutShape == OneGregOut {
			tplName = "pureVreg" // TODO this will be wrong
			data.GoType = goType(gOp)
		} else {
			// OneKmaskOut case
			data.MaskOutConvert = fmt.Sprintf("VPMOVMToVec%dx%d", *gOp.Out[0].ElemBits, *gOp.In[0].Lanes)
			switch opInShape {
			case OneImmIn:
				fallthrough
			case PureVregIn:
				tplName = "maskOut"
				data.GoType = goType(gOp)
			case OneKmaskImmIn:
				fallthrough
			case OneKmaskIn:
				tplName = "maskInMaskOut"
				data.GoType = goType(gOp)
				rearIdx := len(gOp.In) - 1
				data.MaskInConvert = fmt.Sprintf("VPMOVVec%dx%dToM", *gOp.In[rearIdx].ElemBits, *gOp.In[rearIdx].Lanes)
			case PureKmaskIn:
				panic(fmt.Errorf("simdgen does not support pure k mask instructions, they should be generated by compiler optimizations"))
			}
		}

		if gOp.SpecialLower != nil {
			if *gOp.SpecialLower == "sftimm" {
				if !sftimmCheck[data.Asm] {
					sftimmCheck[data.Asm] = true
					sftImmData := data
					if tplName == "maskIn" {
						sftImmData.TplName = "masksftimm"
					} else {
						sftImmData.TplName = "sftimm"
					}
					allData = append(allData, sftImmData)
					asmCheck[sftImmData.Asm+"const"] = true
				}
			} else if ok, cond, out := parseAsmRule(*gOp.SpecialLower); ok {
				if _, done := ruleDone[data.Asm]; !done {
					ruleDone[data.Asm] = struct{}{}
					optData := data
					optData.TplName = "asmRule"
					optData.RuleCond = cond
					if cond != "" {
						optData.RuleCond = "&& " + cond
					}
					optData.RuleOut = out
					if maskType == OneMask {
						optData.Args += " mask"
					}
					allData = append(allData, optData)
				}
			} else if ok, matchArgs, cond, out, isEarly := parseArgsMatchRule(*gOp.SpecialLower); ok {
				key := data.Asm
				if isEarly {
					key = data.GoOp + data.GoType
				}
				if _, done := ruleDone[key]; !done {
					ruleDone[key] = struct{}{}
					// Get elemBits from the operation's inputs.
					elemBits := 0
					for _, in := range gOp.In {
						if in.ElemBits != nil {
							elemBits = *in.ElemBits
							break
						}
					}
					optData := data
					if isEarly {
						optData.TplName = "earlyMatchRule"
					} else {
						optData.TplName = "argsMatchRule"
					}
					optData.RuleArgs = expandFormatSpecifiers(matchArgs, elemBits)
					optData.RuleCond = expandFormatSpecifiers(cond, elemBits)
					if optData.RuleCond != "" {
						optData.RuleCond = "&& " + optData.RuleCond
					}
					optData.RuleOut = expandFormatSpecifiers(out, elemBits)
					allData = append(allData, optData)
				}
				if isEarly {
					continue
				}
			} else {
				panic("simdgen sees unknown special lower " + *gOp.SpecialLower + ", maybe implement it?")
			}
		}
		if gOp.MemFeatures != nil && *gOp.MemFeatures == "vbcst" {
			// sanity check
			selected := true
			for _, a := range gOp.In {
				if a.TreatLikeAScalarOfSize != nil {
					selected = false
					break
				}
			}
			if _, ok := memOpSeen[data.Asm]; ok {
				selected = false
			}
			if selected {
				memOpSeen[data.Asm] = true
				lastVreg := gOp.In[vregInCnt-1]
				// sanity check
				if lastVreg.Class != "vreg" {
					panic(fmt.Errorf("simdgen expects vbcst replaced operand to be a vreg, but %v found", lastVreg))
				}
				memOpData := data
				// Remove the last vreg from the arg and change it to a load.
				origArgs := data.Args[:len(data.Args)-1]
				// Prepare imm args.
				immArg := ""
				immArgCombineOff := " [off] "
				if immType != NoImm && immType != InvalidImm {
					_, after, found := strings.Cut(origArgs, "]")
					if found {
						origArgs = after
					}
					immArg = "[c] "
					immArgCombineOff = " [makeValAndOff(int32(uint8(c)),off)] "
				}
				memOpData.ArgsLoadAddr = immArg + origArgs + fmt.Sprintf("l:(VMOVDQUload%d {sym} [off] ptr mem)", *lastVreg.Bits)
				// Remove the last vreg from the arg and change it to "ptr".
				memOpData.ArgsAddr = "{sym}" + immArgCombineOff + origArgs + "ptr"
				if maskType == OneMask {
					memOpData.ArgsAddr += " mask"
					memOpData.ArgsLoadAddr += " mask"
				}
				memOpData.ArgsAddr += " mem"
				if gOp.MemFeaturesData != nil {
					_, feat2 := getVbcstData(*gOp.MemFeaturesData)
					knownFeatChecks := map[string]string{
						"AVX":    "v.Block.CPUfeatures.hasFeature(CPUavx)",
						"AVX2":   "v.Block.CPUfeatures.hasFeature(CPUavx2)",
						"AVX512": "v.Block.CPUfeatures.hasFeature(CPUavx512)",
					}
					memOpData.FeatCheck = knownFeatChecks[feat2]
					memOpData.TplName = "vregMemFeatCheck"
				} else {
					memOpData.TplName = "vregMem"
				}
				memOptData = append(memOptData, memOpData)
				asmCheck[memOpData.Asm+"load"] = true
			}
		}
		// Generate the masked merging optimization rules
		if gOp.hasMaskedMerging(maskType, opOutShape) {
			// TODO: handle customized operand order and special lower.
			maskElem := gOp.In[len(gOp.In)-1]
			if maskElem.Bits == nil {
				panic("mask has no bits")
			}
			if maskElem.ElemBits == nil {
				panic("mask has no elemBits")
			}
			if maskElem.Lanes == nil {
				panic("mask has no lanes")
			}
			switch *maskElem.Bits {
			case 128, 256:
				// VPBLENDVB cases.
				noMaskName := machineOpName(NoMask, gOp)
				ruleExisting, ok := maskedMergeOpts[noMaskName]
				rule := fmt.Sprintf("(VPBLENDVB%d dst (%s %s) mask) && v.Block.CPUfeatures.hasFeature(CPUavx512) => (%sMerging dst %s (VPMOVVec%dx%dToM <types.TypeMask> mask))\n",
					*maskElem.Bits, noMaskName, data.Args, data.Asm, data.Args, *maskElem.ElemBits, *maskElem.Lanes)
				if ok && ruleExisting != rule {
					panic(fmt.Sprintf("multiple masked merge rules for one op:\n%s\n%s\n", ruleExisting, rule))
				} else {
					maskedMergeOpts[noMaskName] = rule
				}
			case 512:
				// VPBLENDM[BWDQ] cases.
				noMaskName := machineOpName(NoMask, gOp)
				ruleExisting, ok := maskedMergeOpts[noMaskName]
				rule := fmt.Sprintf("(VPBLENDM%sMasked%d dst (%s %s) mask) => (%sMerging dst %s mask)\n",
					s2n[*maskElem.ElemBits], *maskElem.Bits, noMaskName, data.Args, data.Asm, data.Args)
				if ok && ruleExisting != rule {
					panic(fmt.Sprintf("multiple masked merge rules for one op:\n%s\n%s\n", ruleExisting, rule))
				} else {
					maskedMergeOpts[noMaskName] = rule
				}
			}
		}

		if tplName == "pureVreg" && data.Args == data.ArgsOut {
			data.Args = "..."
			data.ArgsOut = "..."
		}
		data.TplName = tplName
		if opr.NoGenericOps != nil && *opr.NoGenericOps == "true" ||
			opr.SkipMaskedMethod() {
			optData = append(optData, data)
			continue
		}
		allData = append(allData, data)
		asmCheck[data.Asm] = true
	}

	slices.SortFunc(allData, compareTplRuleData)

	hiHalfRules := generateHiHalfFoldingRules(ops)
	for _, rule := range hiHalfRules {
		buffer.WriteString(rule)
	}

	for _, data := range allData {
		tpl := ruleTemplates.Get(data.TplName)
		if tpl == nil {
			panic(fmt.Errorf("template %s not found", data.TplName))
		}
		if err := tpl.Execute(buffer, data); err != nil {
			panic(fmt.Errorf("failed to execute template %s for %s: %w", data.TplName, data.GoOp+data.GoType, err))
		}
	}

	seen := make(map[string]bool)

	for _, data := range optData {
		if data.TplName == "maskIn" {
			rule := data.MaskOptimization(asmCheck)
			if seen[rule] {
				continue
			}
			seen[rule] = true
			buffer.WriteString(rule)
		}
	}

	maskedMergeOptsRules := []string{}
	for asm, rule := range maskedMergeOpts {
		if !asmCheck[asm] {
			continue
		}
		maskedMergeOptsRules = append(maskedMergeOptsRules, rule)
	}
	slices.Sort(maskedMergeOptsRules)
	for _, rule := range maskedMergeOptsRules {
		buffer.WriteString(rule)
	}

	for _, data := range memOptData {
		tpl := ruleTemplates.Get(data.TplName)
		if tpl == nil {
			panic(fmt.Errorf("template %s not found", data.TplName))
		}
		if err := tpl.Execute(buffer, data); err != nil {
			panic(fmt.Errorf("failed to execute template %s for %s: %w", data.TplName, data.Asm, err))
		}
	}

	return buffer
}

// generateHiHalfFoldingRules generates folding rules that combine SetHi/GetHi patterns
// with narrow/long base operations into the hi-half "2" variant instruction.
//
// x.GetHi() is lowered as (VDUPDextr[1] x)
//
// x.SetHi(lo) is lowered as (VMOVDins0 [1] x (VDUPDextr [0] lo))
//
// Narrow (e.g., SHRN): SetHi wraps the narrow result.
//
//	(VMOVDins0 [1] dst y:(VSHRN4S [a] x)) => (VSHRN2_4S dst [a] x)
//
// Unary Long (e.g., USHLL) getting high half (GetHi) as input:
//
//	(VUSHLL4H [a] (VDUPDextr [1] x)) => (VUSHLL2_4H [a] x)
//
// Binary Long (e.g., UMULL, SMULL), both inputs from GetHi:
//
//	(VUMULL4H (VDUPDextr [1] x) (VDUPDextr [1] y)) => (VUMULL2_4H x y)
func generateHiHalfFoldingRules(ops []Operation) []string {
	seen := make(map[string]bool)
	var rules []string

	for _, opr := range ops {
		if opr.HiHalfAsm == nil {
			continue
		}
		kind := opr.hiHalfKind()
		if kind == "" {
			continue
		}
		_, _, maskType, immType, gOp, _ := opr.shape()
		asm := machineOpName(maskType, gOp)
		asm2 := hiHalfOpName(*gOp.HiHalfAsm, gOp)

		if seen[asm] {
			continue
		}
		seen[asm] = true

		vregInCnt := 0
		for _, in := range gOp.In {
			if in.Class == "vreg" {
				vregInCnt++
			}
		}
		hasImm := immType == VarImm || immType == VarImmLim || immType == ConstVarImm

		switch kind {
		case "narrow":
			switch vregInCnt {
			case 1:
				// Narrow: the value used by SetHi may be produced with asm2 (hiHalf) variant.
				// TODO: Maybe have a separate rule: (VMOVDins0 [c] dst (VDUPDextr [0] src)) => (VMOVDins0 [c] dst src)
				if hasImm {
					rules = append(rules, fmt.Sprintf("(VMOVDins0 [1] dst (VDUPDextr [0] (%s [c] y))) => (%s dst [c] y)\n", asm, asm2))
				} else {
					rules = append(rules, fmt.Sprintf("(VMOVDins0 [1] dst (VDUPDextr [0] (%s y))) => (%s dst y)\n", asm, asm2))
				}
			default:
				panic("unsupported yet folding narrow ops cases")
			}
		case "long":
			switch vregInCnt {
			case 1:
				// Long: input from GetHi (VDUPDextr [1]), prefer hiHalf variant.
				if hasImm {
					rules = append(rules, fmt.Sprintf("(%s [a] (VDUPDextr [1] x)) => (%s [a] x)\n", asm, asm2))
				} else {
					rules = append(rules, fmt.Sprintf("(%s (VDUPDextr [1] x)) => (%s x)\n", asm, asm2))
				}
			case 2:
				// Binary Long: both inputs are from GetHi, fold into hiHalf variant.
				rules = append(rules, fmt.Sprintf("(%s (VDUPDextr [1] x) (VDUPDextr [1] y)) => (%s x y)\n", asm, asm2))
			default:
				panic("unsupported yet folding long ops cases")
			}
		}
	}

	slices.Sort(rules)
	return rules
}
