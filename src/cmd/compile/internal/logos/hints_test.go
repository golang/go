// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"unicode/utf16"

	"cmd/compile/internal/syntax"
)

// hintsFixture parses, checks, collects hints from, and lowers one fixture
// of testdata/hints, in the order the compiler runs them: facts of the
// unlowered tree first, then the lowering edits.
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

// onlyFile returns the single file of the report.
func onlyFile(t *testing.T, r *HintsReport) FileHints {
	t.Helper()
	if len(r.Files) != 1 {
		t.Fatalf("report holds %d files, want 1", len(r.Files))
	}
	return r.Files[0]
}

// positionAt converts a byte offset of src to an LSP position (0-based line,
// UTF-16 character), independently of the implementation under test.
func positionAt(src []byte, off int) HintPos {
	before := string(src[:off])
	text := before[strings.LastIndex(before, "\n")+1:]
	return HintPos{Line: strings.Count(before, "\n"), Character: len(utf16.Encode([]rune(text)))}
}

// startOf returns the position of the first occurrence of needle in src.
func startOf(t *testing.T, src []byte, needle string) HintPos {
	t.Helper()
	i := strings.Index(string(src), needle)
	if i < 0 {
		t.Fatalf("fixture lacks %q", needle)
	}
	return positionAt(src, i)
}

// endOf returns the position just after the first occurrence of needle.
func endOf(t *testing.T, src []byte, needle string) HintPos {
	t.Helper()
	i := strings.Index(string(src), needle)
	if i < 0 {
		t.Fatalf("fixture lacks %q", needle)
	}
	return positionAt(src, i+len(needle))
}

// TestHintsNarrowedUses checks that every use of an option variable that the
// checker narrows carries the narrowed type, that unnarrowed uses carry none,
// and that columns count UTF-16 code units (SPEC Sec 21.2).
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

// TestHintsUnionMembers checks the member list at a type switch over a closed
// union, anchored just before the `.(type)` guard (SPEC Sec 21.2).
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

// TestHintsInterpFixedBytes checks fixedBytes per interpolation site: the
// literal bytes plus 20 per integer, 24 per float, and 5 per bool; operands
// that carry their own length add nothing (SPEC T-104.4, SPEC Sec 21.3).
func TestHintsInterpFixedBytes(t *testing.T) {
	r, src := hintsFixture(t, "interp.lgo")
	var sites []SiteHint
	for _, s := range onlyFile(t, r).Sites {
		if s.Kind == SiteInterp {
			sites = append(sites, s)
		}
	}
	want := []struct {
		literal string
		fixed   int
	}{
		{`$"user {name}: {n} items, {ratio}, {ok}, {raw}"`, 68},
		{`$"{{id}}={n}"`, 25},
		{`$"plain"`, 5},
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
		}
	}
	if !strings.Contains(sites[0].Lowered, "make([]byte, 0, 19+len(") {
		t.Errorf("lowered interpolation does not show the buffer bound:\n%s", sites[0].Lowered)
	}
}

// TestHintsFixedBytesMatchesBound pins fixedBytes to the lowering: the
// constant terms of the emitted make bound sum to the reported count.
func TestHintsFixedBytesMatchesBound(t *testing.T) {
	if got := InterpFixedBytes([]string{"a", "bc", ""}, nil); got != 3 {
		t.Errorf("literal-only fixed bytes %d, want 3", got)
	}
	if interpIntBound != 20 || interpFloatBound != 24 || interpBoolBound != 5 {
		t.Errorf("operand bounds %d, %d, %d, want 20, 24, 5", interpIntBound, interpFloatBound, interpBoolBound)
	}
}

// TestHintsLoweredTry checks the hover site of a `?`: the range is the `?`
// token and the lowered text is the statement the lowering emitted.
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

// TestHintsLoweredConstSwitch checks the hover site of `const switch T`: the
// range is the two keywords and the lowered text is the switch that replaced
// the statement.
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

// TestHintsLoweredConstFor checks the hover site of `const for`, which the
// parser of this tree does not yet accept: the site comes from the opening
// keywords of a lowering edit, so a synthetic edit over comment text proves
// the classification and the range.
func TestHintsLoweredConstFor(t *testing.T) {
	src := []byte("package p\n\n// const  for i, f := range x { }\n")
	base := syntax.NewFileBase("cf.lgo")
	file := ejectParse(t, base, src, syntax.LogosMode|syntax.CheckBranches)
	pkg, info := ejectCheck(t, file)
	h := NewHints(pkg.Path())
	h.AddFile(pkg, info, file, src)
	h.Lowered(file, []Edit{{
		Start: syntax.MakePos(file.Pos().Base(), 3, 4),
		End:   syntax.MakePos(file.Pos().Base(), 3, 30),
		Text:  "unrolled()",
	}})
	got := onlyFile(t, h.Report()).Sites
	if len(got) != 1 || got[0].Kind != SiteConstFor || got[0].Lowered != "unrolled()" {
		t.Fatalf("const for sites: %+v", got)
	}
	if got[0].Range.Start != startOf(t, src, "const") || got[0].Range.End != endOf(t, src, "const  for") {
		t.Errorf("const for range %+v", got[0].Range)
	}
}

// TestHintsSiteWithoutEditIsDropped checks that a site no lowering edit
// covers produces no hover: a hover without lowered code would be wrong.
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
}

// TestHintsRodata checks that a package-level const composite that lowering
// places in read-only data carries the rodata tier at its name.
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

// TestHintsAllocWorstTierWins checks that an allocation analyzed in several
// contexts reports its worst tier, and that a decision in a file outside the
// package is dropped.
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
