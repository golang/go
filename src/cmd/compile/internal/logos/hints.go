// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"cmp"
	"encoding/json"
	"os"
	"path/filepath"
	"slices"

	"cmd/compile/internal/syntax"
)

const HintsVersion = "1.0"

const (
	TierRodata = "rodata"
	TierArena  = "arena"
	TierStack  = "stack"
	TierHeap   = "heap"
)

const (
	SiteTry         = "try"
	SiteConstFor    = "const_for"
	SiteConstSwitch = "const_switch"
	SiteInterp      = "interp"
)

type HintPos struct {
	Line      int `json:"line"`
	Character int `json:"character"`
}

func (p HintPos) compare(q HintPos) int {
	return cmp.Or(cmp.Compare(p.Line, q.Line), cmp.Compare(p.Character, q.Character))
}

type HintRange struct {
	Start HintPos `json:"start"`
	End   HintPos `json:"end"`
}

type NarrowHint struct {
	Position HintPos `json:"position"`
	Name     string  `json:"name"`
	Type     string  `json:"type"`
}

type UnionHint struct {
	Position HintPos  `json:"position"`
	Type     string   `json:"type"`
	Members  []string `json:"members"`
}

type SiteHint struct {
	Range      HintRange `json:"range"`
	Kind       string    `json:"kind"`
	Lowered    string    `json:"lowered"`
	FixedBytes *int      `json:"fixedBytes,omitempty"`
}

type AllocHint struct {
	Position HintPos `json:"position"`
	Tier     string  `json:"tier"`
	Message  string  `json:"message"`
	Reason   string  `json:"reason,omitempty"`
}

type FileHints struct {
	File     string       `json:"file"`
	Narrowed []NarrowHint `json:"narrowed"`
	Unions   []UnionHint  `json:"unions"`
	Sites    []SiteHint   `json:"sites"`
	Allocs   []AllocHint  `json:"allocs"`
}

type HintsReport struct {
	Version string      `json:"schemaVersion"`
	Package string      `json:"package"`
	Files   []FileHints `json:"files"`
}

type Alloc struct {
	File    string
	Line    int
	Col     int
	Tier    string
	Message string
	Reason  string
}

var tierMergeRank = map[string]int{TierRodata: 0, TierStack: 1, TierArena: 2, TierHeap: 3}

func tierWorse(a, b string) bool {
	return tierMergeRank[a] > tierMergeRank[b]
}

type Hints struct {
	pkgPath string
	byFile  map[*syntax.File]*hintFile
	byName  map[string]*hintFile
}

func NewHints(pkgPath string) *Hints {
	return &Hints{
		pkgPath: pkgPath,
		byFile:  make(map[*syntax.File]*hintFile),
		byName:  make(map[string]*hintFile),
	}
}

type hintFile struct {
	hints   FileHints
	index   lineIndex
	pending []pendingSite
	allocs  map[HintPos]int
}

func newHintFile(name string, src []byte) *hintFile {
	return &hintFile{
		hints:  FileHints{File: name},
		index:  newLineIndex(src),
		allocs: make(map[HintPos]int),
	}
}

func hintFileKey(name string) string {
	if filepath.IsAbs(name) {
		return name
	}
	if abs, err := filepath.Abs(name); err == nil {
		return abs
	}
	return name
}

func (h *Hints) AddAlloc(a Alloc) {
	f := h.byName[a.File]
	if f == nil {
		f = h.byName[hintFileKey(a.File)]
	}
	if f == nil {
		return
	}
	off, ok := f.index.offsetOf(a.Line, a.Col)
	if !ok {
		return
	}
	f.addAlloc(AllocHint{Position: f.index.pos(off), Tier: a.Tier, Message: a.Message, Reason: a.Reason})
}

func (f *hintFile) addAlloc(a AllocHint) {
	if i, dup := f.allocs[a.Position]; dup {
		if tierWorse(a.Tier, f.hints.Allocs[i].Tier) {
			f.hints.Allocs[i] = a
		}
		return
	}
	f.allocs[a.Position] = len(f.hints.Allocs)
	f.hints.Allocs = append(f.hints.Allocs, a)
}

func (h *Hints) Report() *HintsReport {
	r := &HintsReport{Version: HintsVersion, Package: h.pkgPath, Files: make([]FileHints, 0, len(h.byFile))}
	for _, f := range h.byFile {
		r.Files = append(r.Files, f.finish())
	}
	slices.SortFunc(r.Files, func(a, b FileHints) int { return cmp.Compare(a.File, b.File) })
	return r
}

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

func nonNil[S ~[]E, E any](s S) S {
	if s == nil {
		return S{}
	}
	return s
}

func (h *Hints) Write(path string) error {
	data, err := json.Marshal(h.Report())
	if err != nil {
		return err
	}
	return os.WriteFile(path, append(data, '\n'), 0o666)
}
