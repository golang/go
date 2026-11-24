// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssacompile

import (
	"fmt"
	"internal/buildcfg"
	"internal/goarch"
	"strings"
	"sync"

	"cmd/compile/internal/ssa"
	"cmd/compile/internal/ssa/block"
	"cmd/compile/internal/ssa/ssaop"
	"cmd/compile/internal/types"
	"cmd/internal/obj"
)

type localEffect struct {
	start    ssa.CPUfeatures    // features present at beginning of block
	internal ssa.CPUfeatures    // features implied by execution of block
	end      [2]ssa.CPUfeatures // for BlockIf, features present on outgoing edges
	visited  bool               // On the first iteration this will be false for backedges.
}

func (e localEffect) String() string {
	return fmt.Sprintf("visited=%v, start=%v, internal=%v, end[0]=%v, end[1]=%v", e.visited, e.start, e.internal, e.end[0], e.end[1])
}

// featureCheck pattern matches for a BlockIf conditional on a load of a field
// from internal/cpu.X86 or internal/cpu.ARM64, and returns the load, the
// field's name qualified by its struct (for example, "X86.HasAVX2"), and which
// successor runs when the feature is present.
func featureCheck(b *ssa.Block) (load *ssa.Value, match string, taken int) {
	if b.Kind != block.BlockIf {
		return
	}
	c := b.Controls[0]

	if c.Op == ssaop.OpNot {
		taken = 1
		c = c.Args[0]
	}
	if match = featureLoad(c); match == "" {
		return nil, "", 0
	}
	return c, match, taken
}

// featureLoad returns the name of the field of internal/cpu.X86 or
// internal/cpu.ARM64 that v loads, qualified by its struct (for example,
// "X86.HasAVX2"), or "" if v isn't such a load.
func featureLoad(c *ssa.Value) string {
	if c.Op != ssaop.OpLoad {
		return ""
	}
	offPtr := c.Args[0]
	if offPtr.Op != ssaop.OpOffPtr {
		return ""
	}
	addr := offPtr.Args[0]
	if addr.Op != ssaop.OpAddr || addr.Args[0].Op != ssaop.OpSB {
		return ""
	}
	sym := addr.Aux.(*obj.LSym)
	cpu, ok := strings.CutPrefix(sym.Name, "internal/cpu.")
	if !ok || cpu != "X86" && cpu != "ARM64" {
		return ""
	}
	o := offPtr.AuxInt
	t := addr.Type
	if !t.IsPtr() {
		c.Fatalf("The symbol %s is not a pointer, found %v instead", sym.Name, t)
	}
	t = t.Elem()
	if !t.IsStruct() {
		c.Fatalf("The referent of symbol %s is not a struct, found %v instead", sym.Name, t)
	}
	for _, f := range t.Fields() {
		if o == f.Offset && f.Sym != nil {
			return cpu + "." + f.Sym.Name
		}
	}
	return ""
}

// ifEffect pattern matches for a BlockIf conditional on a load
// of a field from internal/cpu.X86 or internal/cpu.ARM64 and returns the
// corresponding effect.
func ifEffect(b *ssa.Block) (features ssa.CPUfeatures, taken int) {
	load, match, taken := featureCheck(b)
	if load == nil {
		return
	}
	features = fieldFeatures(match)
	if b.Func.Pass.Debug > 2 {
		b.Func.Warnl(b.Pos, "%s, block b%v has features %s, features is %v", b.Func.Name, b.ID, match, features)
	}
	return
}

// fieldFeatures returns the features that the compiler can assume when the
// field of internal/cpu.X86 or internal/cpu.ARM64 named by match (for example,
// "X86.HasAVX2") is true.
func fieldFeatures(match string) (features ssa.CPUfeatures) {
	switch match {

	case "X86.HasAVX":
		features = ssa.CPUavx
	case "X86.HasAVXVNNI":
		features = ssa.CPUavx | ssa.CPUavxvnni
	case "X86.HasAVX2":
		features = ssa.CPUavx2 | ssa.CPUavx

		// Compiler currently treats these all alike.
	case "X86.HasAVX512", "X86.HasAVX512F", "X86.HasAVX512CD", "X86.HasAVX512BW",
		"X86.HasAVX512DQ", "X86.HasAVX512VL", "X86.HasAVX512VPCLMULQDQ":
		features = ssa.CPUavx512 | ssa.CPUavx2 | ssa.CPUavx

	case "X86.HasAVX512GFNI":
		features = ssa.CPUavx512 | ssa.CPUgfni | ssa.CPUavx2 | ssa.CPUavx
	case "X86.HasAVX512VNNI":
		features = ssa.CPUavx512 | ssa.CPUavx512vnni | ssa.CPUavx2 | ssa.CPUavx
	case "X86.HasAVX512VBMI":
		features = ssa.CPUavx512 | ssa.CPUvbmi | ssa.CPUavx2 | ssa.CPUavx
	case "X86.HasAVX512VBMI2":
		features = ssa.CPUavx512 | ssa.CPUvbmi2 | ssa.CPUavx2 | ssa.CPUavx
	case "X86.HasAVX512BITALG":
		features = ssa.CPUavx512 | ssa.CPUbitalg | ssa.CPUavx2 | ssa.CPUavx
	case "X86.HasAVX512VPOPCNTDQ":
		features = ssa.CPUavx512 | ssa.CPUvpopcntdq | ssa.CPUavx2 | ssa.CPUavx

	case "ARM64.HasSVE2":
		features = ssa.CPUsve2

		// Features that are not currently interesting to the compiler.
	// HasSVE is not among them: rules only ever upgrade an SVE lowering to an
	// SVE2 encoding, so baseline SVE is never a rule condition.
	case "X86.HasAES", "X86.HasADX", "X86.HasBMI1", "X86.HasBMI2", "X86.HasERMS",
		"X86.HasFSRM", "X86.HasFMA", "X86.HasGFNI", "X86.HasOSXSAVE",
		"X86.HasPCLMULQDQ", "X86.HasPOPCNT", "X86.HasRDTSCP", "X86.HasSHA",
		"X86.HasSSE3", "X86.HasSSSE3", "X86.HasSSE41", "X86.HasSSE42":

	}
	return
}

// goamd64Features returns the features that the GOAMD64 level guarantees
// everywhere: those of the internal/cpu.X86 fields that internal/cpu/mkoptions.go
// lists for the level.
var goamd64Features = sync.OnceValue(func() ssa.CPUfeatures {
	var features ssa.CPUfeatures
	for match, level := range x86FeatureLevels {
		if buildcfg.GOAMD64 >= level {
			features |= fieldFeatures(match)
		}
	}
	return features
})

// noCodeValue reports whether v is known to emit no machine code, so
// that even a SIMD type does not imply the presence of any CPU feature.
func noCodeValue(v *ssa.Value) bool {
	switch v.Op {
	case ssaop.OpZeroSIMD, ssaop.OpPhi, ssaop.OpCopy, ssaop.OpSelectN,
		ssaop.OpArg, ssaop.OpArgIntReg, ssaop.OpArgFloatReg:
		return true
	}
	return false
}

func cpufeatures(f *ssa.Func) {
	arch := f.Config.Ctxt.Arch.Family
	// TODO there are other SIMD architectures
	if arch != goarch.AMD64 && arch != goarch.ARM64 {
		return
	}
	propagateCPUFeatures(f, ifEffect)
	if f.Pass.Debug > 0 {
		for _, b := range f.Blocks {
			if b.CPUfeatures != ssa.CPUNone {
				f.Warnl(b.Pos, "%s, block b%v has features %v", b.Func.Name, b.ID, b.CPUfeatures)
			}

		}
	}
}

// propagateCPUFeatures sets the CPU features of each block of f, given the
// features that each block's branch guarantees on one of its outgoing edges.
func propagateCPUFeatures(f *ssa.Func, ifEffect func(b *ssa.Block) (features ssa.CPUfeatures, taken int)) {
	arch := f.Config.Ctxt.Arch.Family
	po := f.Postorder()

	effects := make([]localEffect, 1+f.NumBlocks(), 1+f.NumBlocks())

	features := func(t *types.Type) ssa.CPUfeatures {
		if t.IsSIMD() && arch == goarch.AMD64 {
			// On arm64 a SIMD type implies no feature a rule conditions on:
			// rules only upgrade SVE lowerings to SVE2 encodings, and using an
			// SVE type does not imply SVE2.
			switch t.Size() {
			case 16, 32:
				return ssa.CPUavx
			case 64:
				return ssa.CPUavx512 | ssa.CPUavx2 | ssa.CPUavx
			}
		}
		return ssa.CPUNone
	}

	// visit blocks in reverse post order
	// when b is visited, all of its predecessors (except for loop back edges)
	// will have been visited
	for i := len(po) - 1; i >= 0; i-- {
		b := po[i]

		var feat ssa.CPUfeatures

		if b == f.Entry {
			// Check the types of inputs and outputs, as well as annotations.
			// Start with none and union all that is implied by all the types seen.
			if f.Type != nil { // a problem for SSA tests
				for _, field := range f.Type.RecvParamsResults() {
					feat |= features(field.Type)
				}
			}

		} else {
			// Start with all and intersect over predecessors
			feat = ssa.CPUAll
			for _, p := range b.Preds {
				pb := p.Block()
				if !effects[pb.ID].visited {

					continue
				}
				pi := p.Index()
				if pb.Kind != block.BlockIf {
					pi = 0
				}

				feat &= effects[pb.ID].end[pi]
			}
		}

		e := localEffect{start: feat, visited: true}

		// Separately capture the internal effects of this block
		var internal ssa.CPUfeatures
		for _, v := range b.Values {
			// the rule applied here is, if the block contains any
			// instruction that would fault if the feature (avx, avx512)
			// were not present, then assume that the feature is present
			// for all the instructions in the block, a fault is a fault.
			if noCodeValue(v) {
				// v emits no instruction, so its type implies
				// nothing about the CPU. In particular a
				// SIMD-typed zero is just a reference to the
				// fixed all-zeros register and flows freely
				// through code that must run on machines
				// without AVX.
				continue
			}
			t := v.Type
			if t.IsResults() {
				for i := 0; i < t.NumFields(); i++ {
					feat |= features(t.FieldType(i))
				}
			} else {
				internal |= features(v.Type)
			}
		}
		e.internal = internal
		feat |= internal

		branchEffect, taken := ifEffect(b)
		e.end = [2]ssa.CPUfeatures{feat, feat}
		e.end[taken] |= branchEffect

		effects[b.ID] = e
		if f.Pass.Debug > 1 && feat != ssa.CPUNone {
			f.Warnl(b.Pos, "%s, block b%v has features %v", b.Func.Name, b.ID, feat)
		}

		b.CPUfeatures = feat
		f.MaxCPUFeatures |= feat // not necessary to refine this estimate below
	}

	// If the flow graph is irreducible, things can still change on backedges.
	change := true
	for change {
		change = false
		for i := len(po) - 1; i >= 0; i-- {
			b := po[i]

			if b == f.Entry {
				continue // cannot change
			}
			feat := ssa.CPUAll
			for _, p := range b.Preds {
				pb := p.Block()
				pi := p.Index()
				if pb.Kind != block.BlockIf {
					pi = 0
				}
				feat &= effects[pb.ID].end[pi]
			}
			e := effects[b.ID]
			if feat == e.start {
				continue
			}
			e.start = feat
			effects[b.ID] = e
			// uh-oh, something changed
			if f.Pass.Debug > 1 {
				f.Warnl(b.Pos, "%s, block b%v saw predecessor feature change", b.Func.Name, b.ID)
			}

			feat |= e.internal
			if feat == e.end[0]&e.end[1] {
				continue
			}

			branchEffect, taken := ifEffect(b)
			e.end = [2]ssa.CPUfeatures{feat, feat}
			e.end[taken] |= branchEffect

			effects[b.ID] = e
			b.CPUfeatures = feat
			if f.Pass.Debug > 1 {
				f.Warnl(b.Pos, "%s, block b%v has new features %v", b.Func.Name, b.ID, feat)
			}
			change = true
		}
	}
	// The GOAMD64 level guarantees some features everywhere. Add them only to
	// functions that use SIMD, so the passes that check MaxCPUFeatures still
	// skip the others. Since these features hold on every path, adding them
	// after the analysis is the same as including them in it.
	if f.MaxCPUFeatures != ssa.CPUNone && arch == goarch.AMD64 {
		baseline := goamd64Features()
		for _, b := range f.Blocks {
			b.CPUfeatures |= baseline
		}
		f.MaxCPUFeatures |= baseline
	}
}

// foldCPUFeatures replaces checks of the internal/cpu.X86 feature flags that
// the GOAMD64 level guarantees with true, in any code: a branch on one of
// them becomes unconditional, and any other load of one becomes a constant.
//
// It runs after the CPU feature analysis (see cpufeatures), so that the code
// a check guards is still credited with the feature, which enables rules
// that use it.
func foldCPUFeatures(f *ssa.Func) {
	if f.Config.Ctxt.Arch.Family != goarch.AMD64 || buildcfg.GOAMD64 < 2 {
		return
	}
	ensured := func(match string) bool {
		level, ok := x86FeatureLevels[match]
		return ok && buildcfg.GOAMD64 >= level
	}
	// folded maps each folded branch's block to the features that the
	// branch guaranteed on its taken edge, which is now its only edge.
	var folded map[*ssa.Block]ssa.CPUfeatures
	for _, b := range f.Blocks {
		// The generic rules that would fold a constant branch have already
		// run, so do it here.
		if load, match, taken := featureCheck(b); load != nil && ensured(match) {
			if folded == nil {
				folded = make(map[*ssa.Block]ssa.CPUfeatures)
			}
			folded[b], _ = ifEffect(b)
			if taken != 0 {
				b.SwapSuccessors()
			}
			b.Reset(block.BlockFirst)
		}
	}
	if folded != nil {
		// Remove the code that doesn't use the feature now, so the lowering
		// rules see the values that the guarded code computes, rather than
		// phis that merge them with that code's.
		deadcode(f)
		// deadcode removed the paths that bypassed the guarded code, so now
		// the blocks after it, such as the code after an "if" on the
		// feature, only follow the guarded code. Recompute the features of
		// each block so they get the feature too, keeping the features that
		// the folded branches guaranteed. propagateCPUFeatures runs to a
		// fixed point, and folding doesn't depend on the features, so one
		// call suffices.
		propagateCPUFeatures(f, func(b *ssa.Block) (ssa.CPUfeatures, int) {
			if feat, ok := folded[b]; ok {
				return feat, 0
			}
			return ifEffect(b)
		})
	}
	for _, b := range f.Blocks {
		for _, v := range b.Values {
			if v.Op == ssaop.OpLoad && v.Type.IsBoolean() && ensured(featureLoad(v)) {
				v.Reset(ssaop.OpConstBool)
				v.AuxInt = ssa.BoolToAuxInt(true)
			}
		}
	}
}
