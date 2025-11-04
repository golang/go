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
)

type tplRuleData struct {
	tplName        string // e.g. "sftimm"
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
}

var (
	ruleTemplates = template.Must(template.New("simdRules").Parse(`
{{define "pureVreg"}}({{.GoOp}}{{.GoType}} {{.Args}}) => ({{.Asm}} {{.ArgsOut}})
{{end}}
{{define "maskIn"}}({{.GoOp}}{{.GoType}} {{.Args}} mask) => ({{.Asm}} {{.ArgsOut}} ({{.MaskInConvert}} <types.TypeMask> mask))
{{end}}
{{define "maskOut"}}({{.GoOp}}{{.GoType}} {{.Args}}) => ({{.MaskOutConvert}} ({{.Asm}} {{.ArgsOut}}))
{{end}}
{{define "maskInMaskOut"}}({{.GoOp}}{{.GoType}} {{.Args}} mask) => ({{.MaskOutConvert}} ({{.Asm}} {{.ArgsOut}} ({{.MaskInConvert}} <types.TypeMask> mask)))
{{end}}
{{define "sftimm"}}({{.Asm}} x (MOVQconst [c])) => ({{.Asm}}const [uint8(c)] x)
{{end}}
{{define "masksftimm"}}({{.Asm}} x (MOVQconst [c]) mask) => ({{.Asm}}const [uint8(c)] x mask)
{{end}}
{{define "vregMem"}}({{.Asm}} {{.ArgsLoadAddr}}) && canMergeLoad(v, l) && clobber(l) => ({{.Asm}}load {{.ArgsAddr}})
{{end}}
{{define "vregMemFeatCheck"}}({{.Asm}} {{.ArgsLoadAddr}}) && {{.FeatCheck}} && canMergeLoad(v, l) && clobber(l)=> ({{.Asm}}load {{.ArgsAddr}})
{{end}}
`))
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

// SSA rewrite rules need to appear in a most-to-least-specific order.  This works for that.
var tmplOrder = map[string]int{
	"masksftimm":    0,
	"sftimm":        1,
	"maskInMaskOut": 2,
	"maskOut":       3,
	"maskIn":        4,
	"pureVreg":      5,
	"vregMem":       6,
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
	if x.tplName == y.tplName {
		return 0
	}
	xo, xok := tmplOrder[x.tplName]
	yo, yok := tmplOrder[y.tplName]
	if !xok {
		panic(fmt.Errorf("Unexpected template name %s, please add to tmplOrder", x.tplName))
	}
	if !yok {
		panic(fmt.Errorf("Unexpected template name %s, please add to tmplOrder", y.tplName))
	}
	return xo - yo
}

// writeSIMDRules generates the lowering and rewrite rules for ssa and writes it to simdAMD64.rules
// within the specified directory.
func writeSIMDRules(ops []Operation) *bytes.Buffer {
	buffer := new(bytes.Buffer)
	buffer.WriteString(generatedHeader + "\n")

	asmCheck := map[string]bool{}
	var allData []tplRuleData
	var optData []tplRuleData    // for mask peephole optimizations, and other misc
	var memOptData []tplRuleData // for memory peephole optimizations
	memOpSeen := make(map[string]bool)

	for _, opr := range ops {
		opInShape, opOutShape, maskType, immType, gOp := opr.shape()
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
		} else if immType == VarImm {
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
		if opOutShape == OneVregOut || opOutShape == OneVregOutAtIn || gOp.Out[0].OverwriteClass != nil {
			switch opInShape {
			case OneImmIn:
				tplName = "pureVreg"
				data.GoType = goType(gOp)
			case PureVregIn:
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
				if data.GoType[0] == 'I' {
					// only do these for signed types, it is a duplicate rewrite for unsigned
					sftImmData := data
					if tplName == "maskIn" {
						sftImmData.tplName = "masksftimm"
					} else {
						sftImmData.tplName = "sftimm"
					}
					allData = append(allData, sftImmData)
					asmCheck[sftImmData.Asm+"const"] = true
				}
			} else {
				panic("simdgen sees unknwon special lower " + *gOp.SpecialLower + ", maybe implement it?")
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
					immArgCombineOff = " [makeValAndOff(int32(int8(c)),off)] "
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
					memOpData.tplName = "vregMemFeatCheck"
				} else {
					memOpData.tplName = "vregMem"
				}
				memOptData = append(memOptData, memOpData)
			}
		}

		if tplName == "pureVreg" && data.Args == data.ArgsOut {
			data.Args = "..."
			data.ArgsOut = "..."
		}
		data.tplName = tplName
		if opr.NoGenericOps != nil && *opr.NoGenericOps == "true" {
			optData = append(optData, data)
			continue
		}
		allData = append(allData, data)
		asmCheck[data.Asm] = true
	}

	slices.SortFunc(allData, compareTplRuleData)

	for _, data := range allData {
		if err := ruleTemplates.ExecuteTemplate(buffer, data.tplName, data); err != nil {
			panic(fmt.Errorf("failed to execute template %s for %s: %w", data.tplName, data.GoOp+data.GoType, err))
		}
	}

	seen := make(map[string]bool)

	for _, data := range optData {
		if data.tplName == "maskIn" {
			rule := data.MaskOptimization(asmCheck)
			if seen[rule] {
				continue
			}
			seen[rule] = true
			buffer.WriteString(rule)
		}
	}

	for _, data := range memOptData {
		if err := ruleTemplates.ExecuteTemplate(buffer, data.tplName, data); err != nil {
			panic(fmt.Errorf("failed to execute template %s for %s: %w", data.tplName, data.Asm, err))
		}
	}

	return buffer
}
