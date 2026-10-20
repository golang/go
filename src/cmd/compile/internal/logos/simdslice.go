// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"fmt"
	"strings"

	"cmd/compile/internal/base"
	"cmd/compile/internal/types2"
	"internal/buildcfg"
)

// LoGo/s M15-S2 vector form of the slice `.sum()` (SPEC Sec 15.2,
// T-303, #22). A `.sum()` on a bare slice receiver, with no fused
// operation, takes this form when the build has GOEXPERIMENT=simd on
// amd64 or arm64 and the element is a fixed-width numeric kind or an
// alias of one (the array rule of M14-S1: `int`, `uint`, `uintptr`, and
// defined element types stay scalar).
//
// Full vectors load through the archsimd array loader on the slice
// converted to an array pointer. The tail, fewer elements than one
// vector, loads once through a fault-suppressing masked load that never
// touches a byte past the last element: on amd64 the Load<T>SliceMasked
// intrinsics of logos_maskload_amd64.go (VPMASKMOVD and VPMASKMOVQ at
// 256 bits, the zeroing K-masked VMOVDQU8 and VMOVDQU16 at 512 bits)
// with a mask that enables the remaining elements; on arm64 the
// upstream Load<T>Part loaders, which insert only the elements present
// because NEON has no masked load. The build adds no guard padding to any
// slice. Every call on the amd64 path is an intrinsic, because a lowered
// package keeps a call to a non-intrinsic archsimd function as a CALL
// (codegen/simd.lgo pins the feature-check CALL), so the loop holds no
// such call.
//
// Integer sums accumulate in vector lanes and add the lanes at the end:
// wrapping integer addition is associative, so the result equals the
// pairwise recursion bit for bit. A float sum keeps the pinned halving
// tree, whose shape depends on the length alone and does not map to
// lane accumulation, so the vector form stages the elements through
// vector loads and stores into a fresh slice and runs the unchanged
// pairwise recursion over it; the printed bits equal the scalar form.
//
// On amd64 the vector form runs only when the archsimd feature its
// instructions need is true (AVX2 for 256-bit vectors, AVX512 for the
// 512-bit 8 and 16-bit lanes) and the scalar chain loop is the
// fallthrough; the check is omitted when GOAMD64 guarantees the
// feature, in which case the scalar loop is not emitted at all.
//
// A compile that writes eject text (-logos.eject) takes the scalar form
// for every slice sum: the Load<T>SliceMasked loaders exist only in this
// toolchain's archsimd, so the vector form would not build with an
// upstream toolchain (T-0.3). The scalar text is identical with and
// without the experiment, so the ejected bits are the native bits.

// simdSliceForm is the vector form of one slice element kind.
type simdSliceForm struct {
	vec      string // archsimd vector type, for example Int32x8
	lanes    int64
	laneBits int64
	total    int64  // vector width in bytes
	kind     string // spelling of the lane kind, for example int32
	feat     simdFeature
	isFloat  bool
}

// simdSliceWidth selects the vector byte width and the feature its
// instructions need: 16 bytes on arm64; on amd64 256 bits with the AVX2
// masked load for 32 and 64-bit lanes, and 512 bits with the AVX512
// K-masked load for 8 and 16-bit lanes, which have no AVX2 masked load.
func simdSliceWidth(lane simdLane) (int64, simdFeature) {
	switch {
	case buildcfg.GOARCH == "arm64":
		return 16, simdFeatBaseline
	case lane.width < 4:
		return 64, simdFeatAVX512
	}
	return 32, simdFeatAVX2
}

// simdSliceElem resolves the lane and the lane kind spelling of a slice
// element, or false when the element has no vector form. `int`, `uint`,
// and `uintptr` keep the scalar form, as in the array shape.
func (fl *fileLower) simdSliceElem(elem types2.Type) (simdLane, string, bool) {
	b, ok := types2.Unalias(elem).(*types2.Basic)
	if !ok || simdWordKind(b.Kind()) {
		return simdLane{}, "", false
	}
	lane, ok := simdLaneOf(b.Kind())
	if !ok {
		return simdLane{}, "", false
	}
	kind, ok := fl.spellType(b)
	return lane, kind, ok
}

// simdSliceFormOf returns the vector form of a slice of elem, or false
// for the scalar form, which an eject compile always takes.
func (fl *fileLower) simdSliceFormOf(elem types2.Type) (simdSliceForm, bool) {
	if !simdVectorEnabled() || base.LogosEject != "" || elem == nil {
		return simdSliceForm{}, false
	}
	lane, kind, ok := fl.simdSliceElem(elem)
	if !ok {
		return simdSliceForm{}, false
	}
	total, feat := simdSliceWidth(lane)
	lanes := total / lane.width
	return simdSliceForm{vec: fmt.Sprintf("%sx%d", lane.name, lanes), lanes: lanes, laneBits: lane.width * 8, total: total, kind: kind, feat: feat, isFloat: lane.isFloat}, true
}

// simdSliceTail spells the statements that bind the tail vector P_tv
// from the elements of P_rcv at and after P_i, which are fewer than one
// vector. On arm64 the upstream partial loader inserts only the
// elements present. On amd64 a mask enables exactly those elements.
func simdSliceTail(p string, f simdSliceForm) []string {
	src := fmt.Sprintf("%[1]s_rcv[%[1]s_i:]", p)
	load := logosSimdAlias + ".Load" + f.vec
	switch {
	case buildcfg.GOARCH == "arm64":
		return []string{fmt.Sprintf("%s_tv, _ := %sPart(%s)", p, load, src)}
	case f.total == 32:
		return simdSliceTailTable(p, f, load, src)
	}
	return simdSliceTailBits(p, f, load, src)
}

// simdSliceTailTable binds P_tv for the 256-bit vectors: a table of the
// lane count of -1 followed by as many zeros, read at an offset that
// leaves the first P_k lanes -1, then compared against zero, is the
// AVX2 mask of the remaining elements.
func simdSliceTailTable(p string, f simdSliceForm, load, src string) []string {
	kind := fmt.Sprintf("int%d", f.laneBits)
	vec := fmt.Sprintf("Int%dx%d", f.laneBits, f.lanes)
	ones := strings.TrimSuffix(strings.Repeat("-1, ", int(f.lanes)), ", ")
	return []string{
		fmt.Sprintf("%[1]s_k := len(%[1]s_rcv) - %[1]s_i", p),
		fmt.Sprintf("%s_mt := [%d]%s{%s}", p, 2*f.lanes, kind, ones),
		fmt.Sprintf("var %s_z %s.%s", p, logosSimdAlias, vec),
		fmt.Sprintf("%[1]s_tv := %[2]sSliceMasked(%[3]s, %[1]s_z.Greater(%[4]s.Load%[5]sArray((*[%[6]d]%[7]s)(%[1]s_mt[%[6]d-%[1]s_k:]))))", p, load, src, logosSimdAlias, vec, f.lanes, kind),
	}
}

// simdSliceTailBits binds P_tv for the 512-bit vectors: the low
// remaining bits of an all-ones constant are the K mask. The shift is
// the lane count when the tail is empty, which clears every bit.
func simdSliceTailBits(p string, f simdSliceForm, load, src string) []string {
	return []string{
		fmt.Sprintf("%[1]s_sh := %[2]d - len(%[1]s_rcv) + %[1]s_i", p, f.lanes),
		fmt.Sprintf("%[1]s_tv := %[2]sSliceMasked(%[3]s, %[4]s.Mask%[5]dx%[6]dFromBits(^uint%[6]d(0) >> %[1]s_sh))", p, load, src, logosSimdAlias, f.laneBits, f.lanes),
	}
}

// simdSliceBlocks spells the vector loop over the full vectors of
// P_rcv, calling step with each loaded vector, then the tail binding
// and the tail step over P_tv. P_i ends at len(P_rcv) rounded down to a
// whole vector. An empty tail enables no lane, so the tail runs
// unconditionally and needs no length test.
func simdSliceBlocks(p string, f simdSliceForm, step func(vec string) string, tailStep func(tv string) []string) []string {
	load := fmt.Sprintf("%s.Load%sArray((*[%d]%s)(%s_rcv[%s_i:]))", logosSimdAlias, f.vec, f.lanes, f.kind, p, p)
	lines := []string{
		p + "_i := 0",
		fmt.Sprintf("for %[1]s_i+%[2]d <= len(%[1]s_rcv) { %[3]s; %[1]s_i += %[2]d }", p, f.lanes, step(load)),
	}
	lines = append(lines, simdSliceTail(p, f)...)
	return append(lines, tailStep(p+"_tv")...)
}

// simdSliceAccumulate spells the integer form: vector lane
// accumulation, then a lane sum.
func simdSliceAccumulate(p string, f simdSliceForm) []string {
	add := func(vec string) string { return fmt.Sprintf("%[1]s_acc = %[1]s_acc.Add(%[2]s)", p, vec) }
	lines := []string{fmt.Sprintf("var %s_acc %s.%s", p, logosSimdAlias, f.vec)}
	lines = append(lines, simdSliceBlocks(p, f, add, func(tv string) []string { return []string{add(tv)} })...)
	return append(lines,
		fmt.Sprintf("var %[1]s_lanes [%[2]d]%[3]s", p, f.lanes, f.kind),
		fmt.Sprintf("%[1]s_acc.StoreArray(&%[1]s_lanes)", p),
		fmt.Sprintf("var %s_t %s", p, f.kind),
		fmt.Sprintf("for _, %[1]s_v := range %[1]s_lanes { %[1]s_t += %[1]s_v }", p),
		"return "+p+"_t",
	)
}

// simdSliceStage spells the float form: vector loads and stores copy
// the elements into P_out, which the scalar form's epilogue then sums
// with the pairwise recursion. The masked tail lands in a lane array
// first, so no store reaches past P_out.
func simdSliceStage(p string, f simdSliceForm) []string {
	lines := []string{fmt.Sprintf("%[1]s_out = make([]%[2]s, len(%[1]s_rcv))", p, f.kind)}
	store := func(vec string) string {
		return fmt.Sprintf("%s.StoreArray((*[%d]%s)(%s_out[%s_i:]))", vec, f.lanes, f.kind, p, p)
	}
	tail := func(tv string) []string {
		return []string{
			fmt.Sprintf("var %s_tmp [%d]%s", p, f.lanes, f.kind),
			fmt.Sprintf("%s.StoreArray(&%s_tmp)", tv, p),
			fmt.Sprintf("copy(%[1]s_out[%[1]s_i:], %[1]s_tmp[:])", p),
		}
	}
	return append(lines, simdSliceBlocks(p, f, store, tail)...)
}

// sliceSumForm returns the vector form of a bare slice `.sum()`: no
// fused operation, no `?` lambda, and an element with a vector form.
func (ce *chainEmit) sliceSumForm() (simdSliceForm, bool) {
	if len(ce.ops) != 0 || ce.term.method != "sum" || ce.isTry {
		return simdSliceForm{}, false
	}
	return ce.cg.fl.simdSliceFormOf(ce.elemT)
}

// guardedBody puts the vector lines ahead of the scalar body lines
// behind the archsimd feature check the form needs, or returns the
// vector lines alone when the GOAMD64 level guarantees the feature.
func (ce *chainEmit) guardedBody(f simdSliceForm, vector, scalar []string) []string {
	cond := ce.cg.fl.simdGuardCond(f.feat)
	if cond == "" {
		return vector
	}
	lines := []string{"if " + cond + " { " + strings.Join(vector, "; ") + " } else {"}
	return append(append(lines, scalar...), "}")
}

// runLoop builds the loop lines of a run. A bare slice `.sum()` with a
// vector form replaces the scalar loop with the vector form behind its
// feature check, the scalar loop staying as the else branch. A float run
// keeps the pairwise epilogue after either branch, so the recursion is
// spelled once. An integer run returns from the vector branch, and
// drops the scalar loop and the epilogue when no check guards it.
func (ce *chainEmit) runLoop() (prelude, body, epilogue []string) {
	prelude, body, epilogue = ce.loop()
	f, ok := ce.sliceSumForm()
	if !ok {
		return prelude, body, epilogue
	}
	ce.cg.fl.needSimd = true
	p := ce.cg.prefix
	if f.isFloat {
		return prelude, ce.guardedBody(f, simdSliceStage(p, f), body), epilogue
	}
	vector := simdSliceAccumulate(p, f)
	if ce.cg.fl.simdGuardCond(f.feat) == "" {
		return prelude[:1], vector, nil
	}
	return prelude, ce.guardedBody(f, vector, body), epilogue
}
