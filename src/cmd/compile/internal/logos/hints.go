// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"cmp"
	"encoding/json"
	"os"
	"slices"

	"cmd/compile/internal/syntax"
)

// This file holds the report of -logos.mode=hints (SPEC Sec 21.2 and 21.3):
// the facts the editor proxy turns into inlay hints and hovers. The compiler
// computes every position in the protocol's own terms (0-based line, UTF-16
// character) from the bytes it parsed, so the consumer needs no document
// text. Narrowing, union, and interpolation facts come from the checked tree
// before lowering; lowered text comes from the lowering edits; escape
// decisions come from escape analysis. Hints.Report assembles them.

// HintsVersion is the schema version of the report.
const HintsVersion = "1.0"

// Allocation tiers of AllocHint.
const (
	TierStack  = "stack"
	TierHeap   = "heap"
	TierRodata = "rodata"
)

// Site kinds of SiteHint.
const (
	SiteTry         = "try"
	SiteConstFor    = "const_for"
	SiteConstSwitch = "const_switch"
	SiteInterp      = "interp"
)

// HintPos is a protocol position: 0-based line, 0-based UTF-16 character.
type HintPos struct {
	Line      int `json:"line"`
	Character int `json:"character"`
}

// compare orders positions by line, then character.
func (p HintPos) compare(q HintPos) int {
	return cmp.Or(cmp.Compare(p.Line, q.Line), cmp.Compare(p.Character, q.Character))
}

// HintRange is a protocol range; End is exclusive.
type HintRange struct {
	Start HintPos `json:"start"`
	End   HintPos `json:"end"`
}

// NarrowHint is one use of an option variable that the checker narrowed.
// Position is just after the name; Type is the narrowed type.
type NarrowHint struct {
	Position HintPos `json:"position"`
	Name     string  `json:"name"`
	Type     string  `json:"type"`
}

// UnionHint is the member list of a closed union at a type switch.
// Position is just before the `.(type)` guard.
type UnionHint struct {
	Position HintPos  `json:"position"`
	Type     string   `json:"type"`
	Members  []string `json:"members"`
}

// SiteHint is a construct that lowering rewrites: Range is the token the
// hover answers on, Lowered the Go text that replaced the site. FixedBytes
// is set for interpolations only: the part of the buffer bound that does
// not depend on run-time lengths.
type SiteHint struct {
	Range      HintRange `json:"range"`
	Kind       string    `json:"kind"`
	Lowered    string    `json:"lowered"`
	FixedBytes *int      `json:"fixedBytes,omitempty"`
}

// AllocHint is one allocation decision: the tier, the -m diagnostic text,
// and the reason the compiler gave for a forced heap allocation.
type AllocHint struct {
	Position HintPos `json:"position"`
	Tier     string  `json:"tier"`
	Message  string  `json:"message"`
	Reason   string  `json:"reason,omitempty"`
}

// FileHints holds the hints of one source file. File is the name the
// compiler was given.
type FileHints struct {
	File     string       `json:"file"`
	Narrowed []NarrowHint `json:"narrowed"`
	Unions   []UnionHint  `json:"unions"`
	Sites    []SiteHint   `json:"sites"`
	Allocs   []AllocHint  `json:"allocs"`
}

// HintsReport is the content of the -logos.hintsout file: one JSON object
// per package.
type HintsReport struct {
	Version string      `json:"schemaVersion"`
	Package string      `json:"package"`
	Files   []FileHints `json:"files"`
}

// Alloc is an escape decision at a source position of a named file, with
// 1-based line and byte column like the positions of -m.
type Alloc struct {
	File    string
	Line    int
	Col     int
	Tier    string
	Message string
	Reason  string
}

// tierRank orders the tiers by cost; the worst tier of one source position
// wins.
var tierRank = map[string]int{TierRodata: 0, TierStack: 1, TierHeap: 2}

// Hints builds the report of one package.
type Hints struct {
	pkgPath string
	byFile  map[*syntax.File]*hintFile
	byName  map[string]*hintFile
}

// NewHints returns an empty builder for the package at pkgPath.
func NewHints(pkgPath string) *Hints {
	return &Hints{
		pkgPath: pkgPath,
		byFile:  make(map[*syntax.File]*hintFile),
		byName:  make(map[string]*hintFile),
	}
}

// hintFile is the state of one file.
type hintFile struct {
	hints   FileHints
	index   lineIndex
	pending []pendingSite
	allocs  map[HintPos]int
}

// newHintFile returns the state of the file named name with bytes src.
func newHintFile(name string, src []byte) *hintFile {
	return &hintFile{
		hints:  FileHints{File: name},
		index:  newLineIndex(src),
		allocs: make(map[HintPos]int),
	}
}

// AddAlloc records an escape decision. A decision in a file outside the
// package, such as an allocation inlined from another package, is dropped.
// Two decisions at one position merge into the worse tier and its text,
// because one source allocation can be analyzed in several inlined contexts
// and an inlined body reports at its call site.
func (h *Hints) AddAlloc(a Alloc) {
	f := h.byName[a.File]
	if f == nil {
		return
	}
	off, ok := f.index.offsetOf(a.Line, a.Col)
	if !ok {
		return
	}
	f.addAlloc(AllocHint{Position: f.index.pos(off), Tier: a.Tier, Message: a.Message, Reason: a.Reason})
}

// addAlloc records a, keeping the worse tier of a duplicate.
func (f *hintFile) addAlloc(a AllocHint) {
	if i, dup := f.allocs[a.Position]; dup {
		if tierRank[a.Tier] > tierRank[f.hints.Allocs[i].Tier] {
			f.hints.Allocs[i] = a
		}
		return
	}
	f.allocs[a.Position] = len(f.hints.Allocs)
	f.hints.Allocs = append(f.hints.Allocs, a)
}

// Report returns the finished report: files, and the hints inside each
// file, sorted by position, and every list present even when empty.
func (h *Hints) Report() *HintsReport {
	r := &HintsReport{Version: HintsVersion, Package: h.pkgPath, Files: make([]FileHints, 0, len(h.byName))}
	for _, f := range h.byName {
		r.Files = append(r.Files, f.finish())
	}
	slices.SortFunc(r.Files, func(a, b FileHints) int { return cmp.Compare(a.File, b.File) })
	return r
}

// finish sorts the hints of f and returns them.
func (f *hintFile) finish() FileHints {
	out := f.hints
	out.Narrowed = nonNil(out.Narrowed)
	out.Unions = nonNil(out.Unions)
	out.Sites = nonNil(out.Sites)
	out.Allocs = nonNil(out.Allocs)
	slices.SortFunc(out.Narrowed, func(a, b NarrowHint) int { return a.Position.compare(b.Position) })
	slices.SortFunc(out.Unions, func(a, b UnionHint) int { return a.Position.compare(b.Position) })
	slices.SortFunc(out.Sites, func(a, b SiteHint) int { return a.Range.Start.compare(b.Range.Start) })
	slices.SortFunc(out.Allocs, func(a, b AllocHint) int {
		return cmp.Or(a.Position.compare(b.Position), cmp.Compare(a.Message, b.Message))
	})
	return out
}

// nonNil returns s, or an empty slice when s is nil, so JSON holds [] and
// never null.
func nonNil[S ~[]E, E any](s S) S {
	if s == nil {
		return S{}
	}
	return s
}

// Write writes the report to path as one JSON object and a newline.
func (h *Hints) Write(path string) error {
	data, err := json.Marshal(h.Report())
	if err != nil {
		return err
	}
	return os.WriteFile(path, append(data, '\n'), 0o666)
}
