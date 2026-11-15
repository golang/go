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

type simdSliceForm struct {
	vec      string
	lanes    int64
	laneBits int64
	total    int64
	kind     string
	feat     simdFeature
	isFloat  bool
}

func simdSliceWidth(lane simdLane) (int64, simdFeature) {
	switch {
	case buildcfg.GOARCH == "arm64":
		return 16, simdFeatBaseline
	case lane.width < 4:
		return 64, simdFeatAVX512
	}
	return 32, simdFeatAVX2
}

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

func simdSliceTailBits(p string, f simdSliceForm, load, src string) []string {
	return []string{
		fmt.Sprintf("%[1]s_sh := %[2]d - len(%[1]s_rcv) + %[1]s_i", p, f.lanes),
		fmt.Sprintf("%[1]s_tv := %[2]sSliceMasked(%[3]s, %[4]s.Mask%[5]dx%[6]dFromBits(^uint%[6]d(0) >> %[1]s_sh))", p, load, src, logosSimdAlias, f.laneBits, f.lanes),
	}
}

func simdSliceBlocks(p string, f simdSliceForm, step func(vec string) string, tailStep func(tv string) []string) []string {
	load := fmt.Sprintf("%s.Load%sArray((*[%d]%s)(%s_rcv[%s_i:]))", logosSimdAlias, f.vec, f.lanes, f.kind, p, p)
	lines := []string{
		p + "_i := 0",
		fmt.Sprintf("for %[1]s_i+%[2]d <= len(%[1]s_rcv) { %[3]s; %[1]s_i += %[2]d }", p, f.lanes, step(load)),
	}
	lines = append(lines, simdSliceTail(p, f)...)
	return append(lines, tailStep(p+"_tv")...)
}

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

func (ce *chainEmit) sliceSumForm() (simdSliceForm, bool) {
	if len(ce.ops) != 0 || ce.term.method != "sum" || ce.isTry {
		return simdSliceForm{}, false
	}
	return ce.cg.fl.simdSliceFormOf(ce.elemT)
}

func (ce *chainEmit) guardedBody(f simdSliceForm, vector, scalar []string) []string {
	cond := ce.cg.fl.simdGuardCond(f.feat)
	if cond == "" {
		return vector
	}
	lines := []string{"if " + cond + " { " + strings.Join(vector, "; ") + " } else {"}
	return append(append(lines, scalar...), "}")
}
