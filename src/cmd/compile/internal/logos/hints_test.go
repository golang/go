// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"testing"
	"unicode/utf16"

	"cmd/compile/internal/syntax"
)

func hintsFixture(t *testing.T, name string) (*HintsReport, []byte) {
	t.Helper()
	src, err := os.ReadFile(filepath.Join("testdata", "hints", name))
	if err != nil {
		t.Fatal(err)
	}
	base := syntax.NewFileBase(name)
	file := ejectParse(t, base, src, syntax.LogosMode|syntax.CheckBranches)
	pkg, info := ejectCheck(t, file)
	h := NewHints(pkg.Path())
	h.AddFile(pkg, info, file, src)
	_, edits := Lower(pkg, info, []*syntax.File{file}, map[*syntax.File][]byte{file: src})
	h.Lowered(file, edits[file])
	h.Rodata(file)
	return h.Report(), src
}

func onlyFile(t *testing.T, r *HintsReport) FileHints {
	t.Helper()
	if len(r.Files) != 1 {
		t.Fatalf("report holds %d files, want 1", len(r.Files))
	}
	return r.Files[0]
}

func positionAt(src []byte, off int) HintPos {
	before := string(src[:off])
	text := before[strings.LastIndex(before, "\n")+1:]
	return HintPos{Line: strings.Count(before, "\n"), Character: len(utf16.Encode([]rune(text)))}
}

func startOf(t *testing.T, src []byte, needle string) HintPos {
	t.Helper()
	i := strings.Index(string(src), needle)
	if i < 0 {
		t.Fatalf("fixture lacks %q", needle)
	}
	return positionAt(src, i)
}

func endOf(t *testing.T, src []byte, needle string) HintPos {
	t.Helper()
	i := strings.Index(string(src), needle)
	if i < 0 {
		t.Fatalf("fixture lacks %q", needle)
	}
	return positionAt(src, i+len(needle))
}

func TestHintsNarrowedUses(t *testing.T) {
	r, src := hintsFixture(t, "sites.lgo")
	got := onlyFile(t, r).Narrowed
	want := []NarrowHint{
		{Position: endOf(t, src, "\t\treturn u"), Name: "u", Type: "*User"},
		{Position: endOf(t, src, "\t\treturn \"é😀\" + u"), Name: "u", Type: "*User"},
	}
	if !slices.Equal(got, want) {
		t.Errorf("narrowed uses:\n got %+v\nwant %+v", got, want)
	}
}

func TestHintsUnionMembers(t *testing.T) {
	r, src := hintsFixture(t, "sites.lgo")
	got := onlyFile(t, r).Unions
	if len(got) != 1 {
		t.Fatalf("union switches: %+v, want 1", got)
	}
	u := got[0]
	if u.Position != endOf(t, src, "switch s") || u.Type != "Shape" || !slices.Equal(u.Members, []string{"Circle", "Square"}) {
		t.Errorf("union switch:\n got %+v\nwant Shape [Circle Square] at %+v", u, endOf(t, src, "switch s"))
	}
}

func TestHintsInterpFixedBytes(t *testing.T) {
	r, src := hintsFixture(t, "interp.lgo")
	var sites []SiteHint
	for _, s := range onlyFile(t, r).Sites {
		if s.Kind == SiteInterp {
			sites = append(sites, s)
		}
	}
	want := []struct {
		literal  string
		fixed    int
		buffered bool
		lens     int
	}{
		{`$"user {name}: {n} items, {ratio}, {ok}, {raw}"`, 68, true, 2},
		{`$"{{id}}={n}"`, 25, true, 0},
		{`$"plain"`, 5, false, 0},
	}
	if len(sites) != len(want) {
		t.Fatalf("%d interpolation sites, want %d: %+v", len(sites), len(want), sites)
	}
	for i, w := range want {
		s := sites[i]
		if s.FixedBytes == nil || *s.FixedBytes != w.fixed {
			t.Errorf("%s: fixedBytes %v, want %d", w.literal, s.FixedBytes, w.fixed)
		}
		if s.Range.Start != startOf(t, src, w.literal) || s.Range.End != endOf(t, src, w.literal) {
			t.Errorf("%s: range %+v", w.literal, s.Range)
		}
		if s.Lowered == "" {
			t.Errorf("%s: no lowered code", w.literal)
			continue
		}
		fixed, lens, buffered := interpBufferBound(t, s.Lowered)
		if buffered != w.buffered || (buffered && (fixed != w.fixed || lens != w.lens)) {
			t.Errorf("%s: lowered buffer capacity %d fixed bytes + %d len terms (buffer %v), want %d + %d (buffer %v):\n%s",
				w.literal, fixed, lens, buffered, w.fixed, w.lens, w.buffered, s.Lowered)
		}
	}
}

func interpBufferBound(t *testing.T, lowered string) (fixed, lens int, buffered bool) {
	t.Helper()
	file := ejectParse(t, syntax.NewFileBase("lowered.go"), []byte("package p\n\nvar _ = "+lowered+"\n"), syntax.CheckBranches)
	var capacity syntax.Expr
	syntax.Inspect(file, func(n syntax.Node) bool {
		if call, ok := n.(*syntax.CallExpr); ok && capacity == nil && len(call.ArgList) == 3 {
			if name, ok := call.Fun.(*syntax.Name); ok && name.Value == "make" {
				capacity = call.ArgList[2]
			}
		}
		return capacity == nil
	})
	if capacity == nil {
		return 0, 0, false
	}
	fixed, lens = capacityTerms(t, capacity)
	return fixed, lens, true
}

func capacityTerms(t *testing.T, e syntax.Expr) (fixed, lens int) {
	t.Helper()
	switch e := e.(type) {
	case *syntax.BasicLit:
		n, err := strconv.Atoi(e.Value)
		if e.Kind != syntax.IntLit || err != nil {
			t.Fatalf("capacity term %s is not an int literal", e.Value)
		}
		return n, 0
	case *syntax.ParenExpr:
		return capacityTerms(t, e.X)
	case *syntax.CallExpr:
		if name, ok := e.Fun.(*syntax.Name); ok && name.Value == "len" {
			return 0, 1
		}
	case *syntax.Operation:
		if e.Op == syntax.Add && e.Y != nil {
			xf, xl := capacityTerms(t, e.X)
			yf, yl := capacityTerms(t, e.Y)
			return xf + yf, xl + yl
		}
	}
	t.Fatalf("capacity term %T is not a sum of int literals and len calls", e)
	return 0, 0
}

func TestHintsFixedBytesMatchesBound(t *testing.T) {
	if got := InterpFixedBytes([]string{"a", "bc", ""}, nil); got != 3 {
		t.Errorf("literal-only fixed bytes %d, want 3", got)
	}
}

func TestHintsLoweredTry(t *testing.T) {
	r, src := hintsFixture(t, "sites.lgo")
	var tries []SiteHint
	for _, s := range onlyFile(t, r).Sites {
		if s.Kind == SiteTry {
			tries = append(tries, s)
		}
	}
	if len(tries) != 1 {
		t.Fatalf("%d try sites, want 1", len(tries))
	}
	s := tries[0]
	if s.Range.Start != endOf(t, src, "n := get()") || s.Range.End != endOf(t, src, "n := get()?") {
		t.Errorf("try range %+v", s.Range)
	}
	for _, frag := range []string{"get()", "!= nil", "return"} {
		if !strings.Contains(s.Lowered, frag) {
			t.Errorf("lowered try lacks %q:\n%s", frag, s.Lowered)
		}
	}
	if s.FixedBytes != nil {
		t.Errorf("try site carries fixedBytes %d", *s.FixedBytes)
	}
}

func TestHintsLoweredConstSwitch(t *testing.T) {
	r, src := hintsFixture(t, "sites.lgo")
	var got []SiteHint
	for _, s := range onlyFile(t, r).Sites {
		if s.Kind == SiteConstSwitch {
			got = append(got, s)
		}
	}
	if len(got) != 1 {
		t.Fatalf("%d const switch sites, want 1", len(got))
	}
	s := got[0]
	if s.Range.Start != startOf(t, src, "const switch") || s.Range.End != endOf(t, src, "const switch") {
		t.Errorf("const switch range %+v", s.Range)
	}
	if !strings.Contains(s.Lowered, "switch") || strings.Contains(s.Lowered, "const switch") {
		t.Errorf("lowered const switch:\n%s", s.Lowered)
	}
}

func TestHintsLoweredConstFor(t *testing.T) {
	r, src := hintsFixture(t, "sites.lgo")
	var got []SiteHint
	for _, s := range onlyFile(t, r).Sites {
		if s.Kind == SiteConstFor {
			got = append(got, s)
		}
	}
	if len(got) != 1 {
		t.Fatalf("%d const for sites, want 1", len(got))
	}
	s := got[0]
	if s.Range.Start != startOf(t, src, "const for") || s.Range.End != endOf(t, src, "const for") {
		t.Errorf("const for range %+v", s.Range)
	}
	if !strings.Contains(s.Lowered, `out = append(out, "Name")`) || strings.Contains(s.Lowered, "const for") {
		t.Errorf("lowered const for:\n%s", s.Lowered)
	}
}

func TestHintsSiteWithoutEditIsDropped(t *testing.T) {
	src, err := os.ReadFile(filepath.Join("testdata", "hints", "sites.lgo"))
	if err != nil {
		t.Fatal(err)
	}
	file := ejectParse(t, syntax.NewFileBase("sites.lgo"), src, syntax.LogosMode|syntax.CheckBranches)
	pkg, info := ejectCheck(t, file)
	h := NewHints(pkg.Path())
	h.AddFile(pkg, info, file, src)
	h.Lowered(file, nil)
	if sites := onlyFile(t, h.Report()).Sites; len(sites) != 0 {
		t.Errorf("sites without edits survive: %+v", sites)
	}
	surviving := NewHints(pkg.Path())
	surviving.AddFile(pkg, info, file, src)
	_, edits := Lower(pkg, info, []*syntax.File{file}, map[*syntax.File][]byte{file: src})
	surviving.Lowered(file, edits[file])
	sites := onlyFile(t, surviving.Report()).Sites
	if len(sites) == 0 {
		t.Fatal("no hint site survives its own lowering edit")
	}
	var interp *SiteHint
	for i := range sites {
		if sites[i].Kind == SiteInterp {
			interp = &sites[i]
		}
	}
	if interp == nil {
		t.Fatalf("interpolation site dropped despite its lowering edit: %+v", sites)
	}
	if interp.Range.Start != startOf(t, src, `$"user {name}!"`) || interp.Range.End != endOf(t, src, `$"user {name}!"`) {
		t.Errorf("interpolation range %+v, want the lowering-map span", interp.Range)
	}
	if interp.FixedBytes == nil {
		t.Fatalf("interpolation survivor lacks fixed bytes: %+v", interp)
	}
	if fixed, lens, buffered := interpBufferBound(t, interp.Lowered); !buffered || fixed != *interp.FixedBytes || lens != 1 {
		t.Errorf("interpolation survivor buffer capacity %d fixed bytes + %d len terms (buffer %v), want %d + 1:\n%s", fixed, lens, buffered, *interp.FixedBytes, interp.Lowered)
	}
}

func TestHintsRodata(t *testing.T) {
	r, src := hintsFixture(t, "sites.lgo")
	var got []AllocHint
	for _, a := range onlyFile(t, r).Allocs {
		if a.Tier == TierRodata {
			got = append(got, a)
		}
	}
	if want := startOf(t, src, "primes"); len(got) != 1 || got[0].Position != want {
		t.Fatalf("rodata hints %+v, want one at the name primes (%+v)", got, want)
	}
}

func TestHintsAllocWorstTierWins(t *testing.T) {
	src := []byte("package p\n\nvar x = 1\n")
	file := ejectParse(t, syntax.NewFileBase("a.go"), src, syntax.CheckBranches)
	pkg, info := ejectCheck(t, file)
	h := NewHints(pkg.Path())
	h.AddFile(pkg, info, file, src)
	h.AddAlloc(Alloc{File: "a.go", Line: 3, Col: 9, Tier: TierStack, Message: "&T{...} does not escape"})
	h.AddAlloc(Alloc{File: "a.go", Line: 3, Col: 9, Tier: TierHeap, Message: "&T{...} escapes to heap"})
	h.AddAlloc(Alloc{File: "other.go", Line: 1, Col: 1, Tier: TierHeap, Message: "elsewhere"})
	got := onlyFile(t, h.Report()).Allocs
	if len(got) != 1 || got[0].Tier != TierHeap || got[0].Position != (HintPos{Line: 2, Character: 8}) {
		t.Errorf("merged allocs: %+v", got)
	}
}

func TestHintsAllocTierMerge(t *testing.T) {
	tests := []struct {
		first, second string
		want          string
	}{
		{TierStack, TierArena, TierArena},
		{TierArena, TierStack, TierArena},
		{TierArena, TierHeap, TierHeap},
		{TierHeap, TierArena, TierHeap},
		{TierStack, TierHeap, TierHeap},
		{TierHeap, TierStack, TierHeap},
		{TierRodata, TierArena, TierArena},
		{TierArena, TierRodata, TierArena},
		{TierRodata, TierStack, TierStack},
		{TierStack, TierRodata, TierStack},
		{TierRodata, TierHeap, TierHeap},
		{TierHeap, TierRodata, TierHeap},
	}
	for _, tc := range tests {
		t.Run(tc.first+"+"+tc.second, func(t *testing.T) {
			src := []byte("package p\n\nvar x = 1\n")
			file := ejectParse(t, syntax.NewFileBase("a.go"), src, syntax.CheckBranches)
			pkg, info := ejectCheck(t, file)
			h := NewHints(pkg.Path())
			h.AddFile(pkg, info, file, src)
			h.AddAlloc(Alloc{File: "a.go", Line: 3, Col: 9, Tier: tc.first, Message: tc.first})
			h.AddAlloc(Alloc{File: "a.go", Line: 3, Col: 9, Tier: tc.second, Message: tc.second})
			got := onlyFile(t, h.Report()).Allocs
			if len(got) != 1 || got[0].Tier != tc.want {
				t.Fatalf("merge(%s, %s) = %+v, want tier %s", tc.first, tc.second, got, tc.want)
			}
		})
	}
}
