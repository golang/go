// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import (
	"cmd/compile/internal/types"
	"cmd/internal/obj"
	"fmt"
	"internal/goarch"
)

type localEffect struct {
	start    CPUfeatures    // features present at beginning of block
	internal CPUfeatures    // features implied by execution of block
	end      [2]CPUfeatures // for BlockIf, features present on outgoing edges
	visited  bool           // On the first iteration this will be false for backedges.
}

func (e localEffect) String() string {
	return fmt.Sprintf("visited=%v, start=%v, internal=%v, end[0]=%v, end[1]=%v", e.visited, e.start, e.internal, e.end[0], e.end[1])
}

// ifEffect pattern matches for a BlockIf conditional on a load
// of a field from internal/cpu.X86 and returns the corresponding
// effect.
func ifEffect(b *Block) (features CPUfeatures, taken int) {
	// TODO generalize for other architectures.
	if b.Kind != BlockIf {
		return
	}
	c := b.Controls[0]

	if c.Op == OpNot {
		taken = 1
		c = c.Args[0]
	}
	if c.Op != OpLoad {
		return
	}
	offPtr := c.Args[0]
	if offPtr.Op != OpOffPtr {
		return
	}
	addr := offPtr.Args[0]
	if addr.Op != OpAddr || addr.Args[0].Op != OpSB {
		return
	}
	sym := addr.Aux.(*obj.LSym)
	if sym.Name != "internal/cpu.X86" {
		return
	}
	o := offPtr.AuxInt
	t := addr.Type
	if !t.IsPtr() {
		b.Func.Fatalf("The symbol %s is not a pointer, found %v instead", sym.Name, t)
	}
	t = t.Elem()
	if !t.IsStruct() {
		b.Func.Fatalf("The referent of symbol %s is not a struct, found %v instead", sym.Name, t)
	}
	match := ""
	for _, f := range t.Fields() {
		if o == f.Offset && f.Sym != nil {
			match = f.Sym.Name
			break
		}
	}

	switch match {

	case "HasAVX":
		features = CPUavx
	case "HasAVXVNNI":
		features = CPUavx | CPUavxvnni
	case "HasAVX2":
		features = CPUavx2 | CPUavx

		// Compiler currently treats these all alike.
	case "HasAVX512", "HasAVX512F", "HasAVX512CD", "HasAVX512BW",
		"HasAVX512DQ", "HasAVX512VL", "HasAVX512VPCLMULQDQ":
		features = CPUavx512 | CPUavx2 | CPUavx

	case "HasAVX512GFNI":
		features = CPUavx512 | CPUgfni | CPUavx2 | CPUavx
	case "HasAVX512VNNI":
		features = CPUavx512 | CPUavx512vnni | CPUavx2 | CPUavx
	case "HasAVX512VBMI":
		features = CPUavx512 | CPUvbmi | CPUavx2 | CPUavx
	case "HasAVX512VBMI2":
		features = CPUavx512 | CPUvbmi2 | CPUavx2 | CPUavx
	case "HasAVX512BITALG":
		features = CPUavx512 | CPUbitalg | CPUavx2 | CPUavx
	case "HasAVX512VPOPCNTDQ":
		features = CPUavx512 | CPUvpopcntdq | CPUavx2 | CPUavx

	case "HasBMI1":
		features = CPUvbmi
	case "HasBMI2":
		features = CPUvbmi2

		// Features that are not currently interesting to the compiler.
	case "HasAES", "HasADX", "HasERMS", "HasFSRM", "HasFMA", "HasGFNI", "HasOSXSAVE",
		"HasPCLMULQDQ", "HasPOPCNT", "HasRDTSCP", "HasSHA",
		"HasSSE3", "HasSSSE3", "HasSSE41", "HasSSE42":

	}
	if b.Func.pass.debug > 2 {
		b.Func.Warnl(b.Pos, "%s, block b%v has features offset %d, match is %s, features is %v", b.Func.Name, b.ID, o, match, features)
	}
	return
}

func cpufeatures(f *Func) {
	arch := f.Config.Ctxt().Arch.Family
	// TODO there are other SIMD architectures
	if arch != goarch.AMD64 {
		return
	}

	po := f.Postorder()

	effects := make([]localEffect, 1+f.NumBlocks(), 1+f.NumBlocks())

	features := func(t *types.Type) CPUfeatures {
		if t.IsSIMD() {
			switch t.Size() {
			case 16, 32:
				return CPUavx
			case 64:
				return CPUavx512 | CPUavx2 | CPUavx
			}
		}
		return CPUNone
	}

	// visit blocks in reverse post order
	// when b is visited, all of its predecessors (except for loop back edges)
	// will have been visited
	for i := len(po) - 1; i >= 0; i-- {
		b := po[i]

		var feat CPUfeatures

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
			feat = CPUAll
			for _, p := range b.Preds {
				pb := p.Block()
				if !effects[pb.ID].visited {

					continue
				}
				pi := p.Index()
				if pb.Kind != BlockIf {
					pi = 0
				}

				feat &= effects[pb.ID].end[pi]
			}
		}

		e := localEffect{start: feat, visited: true}

		// Separately capture the internal effects of this block
		var internal CPUfeatures
		for _, v := range b.Values {
			// the rule applied here is, if the block contains any
			// instruction that would fault if the feature (avx, avx512)
			// were not present, then assume that the feature is present
			// for all the instructions in the block, a fault is a fault.
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
		e.end = [2]CPUfeatures{feat, feat}
		e.end[taken] |= branchEffect

		effects[b.ID] = e
		if f.pass.debug > 1 && feat != CPUNone {
			f.Warnl(b.Pos, "%s, block b%v has features %v", b.Func.Name, b.ID, feat)
		}

		b.CPUfeatures = feat
		f.maxCPUFeatures |= feat // not necessary to refine this estimate below
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
			feat := CPUAll
			for _, p := range b.Preds {
				pb := p.Block()
				pi := p.Index()
				if pb.Kind != BlockIf {
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
			if f.pass.debug > 1 {
				f.Warnl(b.Pos, "%s, block b%v saw predecessor feature change", b.Func.Name, b.ID)
			}

			feat |= e.internal
			if feat == e.end[0]&e.end[1] {
				continue
			}

			branchEffect, taken := ifEffect(b)
			e.end = [2]CPUfeatures{feat, feat}
			e.end[taken] |= branchEffect

			effects[b.ID] = e
			b.CPUfeatures = feat
			if f.pass.debug > 1 {
				f.Warnl(b.Pos, "%s, block b%v has new features %v", b.Func.Name, b.ID, feat)
			}
			change = true
		}
	}
	if f.pass.debug > 0 {
		for _, b := range f.Blocks {
			if b.CPUfeatures != CPUNone {
				f.Warnl(b.Pos, "%s, block b%v has features %v", b.Func.Name, b.ID, b.CPUfeatures)
			}

		}
	}
}
