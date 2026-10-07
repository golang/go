// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"cmp"
	"path/filepath"
	"regexp"
	"simd/archsimd/_gen/specgen"
	"slices"
	"strings"

	"gopkg.in/yaml.v3"
)

// specOp is one spec operation -- one function in package spec. It is the grain
// spec is authored at, and the only grain at which "implemented nowhere" means
// anything: one operation expands into many API names and many (receiver, name)
// pairs. The API has no corresponding notion, so this is a spec-side diagnostic
// rather than a fourth coverage grain (SPEC-TRANSITION.md §1.3).
type specOp struct {
	name     string // the operation's name in package spec
	covPairs int    // pairs it generates that the API also declares
}

// Spec and the API are each a map from name to the set of receivers that name
// is declared on. A relation is how those two sets stand for one name, and
// walking the union of the two key sets classifies every name into exactly one
// -- which the older "divergence classes" did not do, since a name whose sets
// cross contributes to both sides at once.
type relation int

const (
	relEqual    relation = iota // api = spec
	relSubset                   // api is a proper subset of spec: spec covers more receivers
	relSuperset                 // api is a proper superset of spec: spec covers fewer
	relOverlap                  // they share receivers, neither contains the other
	relDisjoint                 // they share the name and not one receiver
	relSpecOnly                 // api = ∅: no target declares the name
	relAPIOnly                  // spec = ∅: spec has never defined the name
	numRelations
)

// shared reports whether both sides define the name. The api-only receivers of
// a shared name are defects; under spec = ∅ they are ordinary coverage.
func (r relation) shared() bool { return r != relSpecOnly && r != relAPIOnly }

var relationName = [numRelations]string{
	relEqual:    "api = spec",
	relSubset:   "api ⊂ spec",
	relSuperset: "api ⊃ spec",
	relOverlap:  "overlapping",
	relDisjoint: "disjoint",
	relSpecOnly: "api = ∅",
	relAPIOnly:  "spec = ∅",
}

// nameRel is one name, and how the two receiver sets stand under it.
type nameRel struct {
	name      string
	rel       relation
	api, spec int      // receiver-set sizes
	apiOnly   []string // receivers only the API declares, sorted
	specOnly  int      // receivers only spec generates
}

// relTotal is one row of the relation summary: how many names stand in it, and
// how many receivers each side holds alone across them.
type relTotal struct{ names, apiOnly, specOnly int }

// fileCount pairs a filename with a count, for per-file breakdowns.
type fileCount struct {
	file string
	n    int
}

// metrics is the whole report as data. Every field is a tally or a pivot, so
// the summary and the sections below it read the same numbers and cannot
// disagree.
type metrics struct {
	facts []fact
	decls []*decl // the facts' declarations, deduplicated across configurations
	spec  map[methodKey]*specgen.Func

	// Coverage, at three grains and split two ways.
	covTriple *tally[archMethodKey]
	covPair   *tally[methodKey]
	covName   *tally[string]
	covTarget *pivot[archMethodKey]
	covGen    *pivot[methodKey]
	covMulti  *tally[string] // names emitted by more than one tool
	covCommon *tally[string] // comments.yaml's .common_methods, spec-common's scope

	specPairs, specNames int

	// Divergence classes
	ops                   []specOp               // spec source functions, sorted by name
	specOnly              []string               // spec operations no target implements
	names                 []nameRel              // every name either side defines, sorted
	rel                   [numRelations]relTotal // m.names summed by relation
	apiOnly, apiOnlyDecls int                    // the defect: over shared names only

	// Agreement with spec where the two overlap. Names are split the way
	// specdoc.Fill treats them: a named argument that disagrees with spec is a
	// mismatch, and an unnamed one is a name Fill will fill in from spec.
	agree, typeOK               *tally[declKey]
	paramNameOK, resultNameOK   *tally[declKey]
	paramsFilled, resultsFilled *tally[declKey]

	// The invariant, checked without reference to spec.
	consistent, typeSame, paramNameSame, resultNameSame *tally[methodKey]

	// Documentation.
	fromSpec           *tally[declKey]
	bounded            *tally[declKey]
	handConflict       int
	handConflictByFile []fileCount
	crossPkg           *tally[methodKey]
	mirror             *tally[methodKey]

	// The inputs being retired.
	categories                       map[string]int
	catEntries, catUniq, catExported int
	catCompilerOnly                  int
	catExportedFields                map[string]int
	comments                         int
	commentsSpecCovered              int
	commutDrift                      int
}

func measure(api []*decl, specFuncs []*specgen.Func, genDir string) *metrics {
	m := &metrics{spec: map[methodKey]*specgen.Func{}}

	specNames := map[string]bool{}
	for _, f := range specFuncs {
		m.spec[specPair(f)] = f
		specNames[f.Name] = true
	}
	m.specPairs, m.specNames = len(m.spec), len(specNames)

	// The fact table: one row per declaration per configuration it builds in.
	for _, d := range api {
		if nonSpecOp(d.name) {
			continue
		}
		m.decls = append(m.decls, d)
		for _, t := range d.targets {
			m.facts = append(m.facts, fact{d, t})
		}
	}

	covered := func(f fact) bool { return m.spec[f.methodKey()] != nil }

	// 1. Coverage. Same predicate, three grains, two dimensions.
	m.covTriple = tallyBy(m.facts, fact.archMethodKey, covered)
	m.covPair = tallyBy(m.facts, func(f fact) methodKey { return f.methodKey() }, covered)
	m.covName = tallyBy(m.facts, func(f fact) string { return f.name }, func(f fact) bool { return specNames[f.name] })
	m.covTarget = pivotBy(m.facts, func(f fact) string { return f.target }, fact.archMethodKey, covered)
	m.covGen = pivotBy(m.facts, func(f fact) string { return f.gen }, func(f fact) methodKey { return f.methodKey() }, covered)
	for _, g := range m.covGen.dims() {
		if t, _, _ := strings.Cut(g, "/"); t != g {
			m.covGen.fold(g, t)
		}
	}

	// Names emitted by more than one tool are where the invariant is most
	// exposed, because no single generator can see the conflict. simdgen's own
	// runs share their inputs, so they do not count as more than one.
	tools := groupBy(m.facts, func(f fact) string { return f.name })
	m.covMulti = new(tally[string])
	for name, fs := range tools {
		seen := map[string]bool{}
		for _, f := range fs {
			if f.gen != "hand-written" {
				seen[f.tool()] = true
			}
		}
		if len(seen) > 1 {
			m.covMulti.add(name, specNames[name])
		}
	}
	m.covCommon = new(tally[string])
	for _, n := range commonMethods(genDir) {
		m.covCommon.add(n, m.covName.done(n))
	}

	// 2. Divergence classes.
	byOp := map[string]*specOp{}
	for _, f := range specFuncs {
		src, _, _ := f.SpecFunc()
		op := byOp[src]
		if op == nil {
			op = &specOp{name: src}
			byOp[src] = op
		}
		if m.covPair.has(specPair(f)) {
			op.covPairs++
		}
	}
	for _, op := range byOp {
		if op.covPairs == 0 {
			// Operations implemented by spec but not by the API on any platform
			// is measured at the grain spec is authored at: the source
			// function. A spec operation with no covered declaration anywhere
			// is likely a defect, because that takes four simultaneous
			// omissions to explain away.
			m.specOnly = append(m.specOnly, op.name)
		}
		m.ops = append(m.ops, *op)
	}
	slices.Sort(m.specOnly)
	slices.SortFunc(m.ops, func(a, b specOp) int { return cmp.Compare(a.name, b.name) })
	// The two maps the rest of this section is about, and the union of their
	// keys. Everything below -- the defect count, the review queue, and the
	// coverage gap -- is a statement about how one name's two receiver sets
	// stand, so they are derived here once rather than counted three ways.
	apiRecv, specRecv := map[string]map[string]bool{}, map[string]map[string]bool{}
	add := func(m map[string]map[string]bool, name, recv string) {
		if m[name] == nil {
			m[name] = map[string]bool{}
		}
		m[name][recv] = true
	}
	for _, d := range m.decls {
		add(apiRecv, d.name, d.recv)
	}
	for p := range m.spec {
		if !nonSpecOp(p.name) {
			add(specRecv, p.name, p.recv)
		}
	}
	for name := range apiRecv {
		m.names = append(m.names, nameRelation(name, apiRecv[name], specRecv[name]))
	}
	for name := range specRecv {
		if apiRecv[name] == nil {
			m.names = append(m.names, nameRelation(name, nil, specRecv[name]))
		}
	}
	slices.SortFunc(m.names, func(a, b nameRel) int { return cmp.Compare(a.name, b.name) })
	for _, r := range m.names {
		t := &m.rel[r.rel]
		t.names++
		t.apiOnly += len(r.apiOnly)
		t.specOnly += r.specOnly
		if r.rel.shared() {
			m.apiOnly += len(r.apiOnly)
		}
	}
	// The same api-only receivers counted as declarations rather than pairs: one
	// pair can be declared once per target.
	m.apiOnlyDecls = countBy(filter(m.facts, func(f fact) bool {
		return !covered(f) && specNames[f.name]
	}), fact.declKey)

	// 3. Agreement with spec, over the declarations spec covers.
	overlap := filter(m.facts, covered)
	specIn := func(f fact) []param { return specParams(m.spec[f.methodKey()].In) }
	specOut := func(f fact) []param { return specParams(m.spec[f.methodKey()].Out) }
	typeOK := func(f fact) bool {
		return equalBy(f.in, specIn(f), paramType) && equalBy(f.out, specOut(f), paramType)
	}
	resultNameOK := func(f fact) bool { return !namesClash(f.out, specOut(f)) }
	paramNameOK := func(f fact) bool { return !namesClash(f.in, specIn(f)) }
	resultsFilled := func(f fact) bool { return !namesMissing(f.out, specOut(f)) }
	paramsFilled := func(f fact) bool { return !namesMissing(f.in, specIn(f)) }

	m.typeOK = tallyBy(overlap, fact.declKey, typeOK)
	m.resultNameOK = tallyBy(overlap, fact.declKey, resultNameOK)
	m.paramNameOK = tallyBy(overlap, fact.declKey, paramNameOK)
	m.resultsFilled = tallyBy(overlap, fact.declKey, resultsFilled)
	m.paramsFilled = tallyBy(overlap, fact.declKey, paramsFilled)
	m.agree = tallyBy(overlap, fact.declKey, func(f fact) bool {
		return typeOK(f) && resultNameOK(f) && paramNameOK(f)
	})

	// 4. The invariant: where a pair is declared more than once, do the
	// declarations agree with each other? An unnamed argument agrees with any
	// name, since only names that are used -- by a doc or a body -- need to agree.
	m.consistent = new(tally[methodKey])
	m.typeSame = new(tally[methodKey])
	m.paramNameSame = new(tally[methodKey])
	m.resultNameSame = new(tally[methodKey])
	for p, ds := range groupBy(m.decls, (*decl).methodKey) {
		if len(ds) < 2 {
			continue
		}
		params := func(d *decl) []param { return d.in }
		results := func(d *decl) []param { return d.out }
		typesAgree := func(get func(*decl) []param) bool {
			return !slices.ContainsFunc(ds[1:], func(d *decl) bool {
				return !equalBy(get(ds[0]), get(d), paramType)
			})
		}
		namesAgree := func(get func(*decl) []param) bool {
			lists := make([][]param, len(ds))
			for i, d := range ds {
				lists[i] = get(d)
			}
			return namesConsistent(lists)
		}
		t := typesAgree(params) && typesAgree(results)
		pn, rn := namesAgree(params), namesAgree(results)
		m.typeSame.add(p, t)
		m.paramNameSame.add(p, pn)
		m.resultNameSame.add(p, rn)
		m.consistent.add(p, t && pn && rn)
	}

	// 5. Documentation.
	// A declaration is sourced from spec when its doc begins with spec's text --
	// a prefix test, because Fill leaves the generator's notes below what it
	// injects. Prose that disagrees instead is a conflict for doc-triage; a missing
	// doc is not, since Fill would simply add one.
	documented := filter(overlap, func(f fact) bool { return len(m.spec[f.methodKey()].Doc) > 0 })
	saysSpec := func(f fact) bool {
		specDoc, err := specgen.FormatComment(m.spec[f.methodKey()].Doc)
		if err != nil {
			return false
		}
		return strings.HasPrefix(f.doc, strings.TrimRight(specDoc, "\n"))
	}
	m.fromSpec = tallyBy(documented, fact.declKey, saysSpec)
	handConflicts := filter(documented, func(f fact) bool {
		return f.gen == "hand-written" && f.doc != "" && !saysSpec(f)
	})
	m.handConflict = countBy(handConflicts, fact.declKey)
	byFile := map[string]int{}
	seenByFile := map[declKey]bool{}
	for _, f := range handConflicts {
		dk := f.declKey()
		if !seenByFile[dk] {
			seenByFile[dk] = true
			byFile[f.file]++
		}
	}
	for _, k := range sortedKeys(byFile) {
		m.handConflictByFile = append(m.handConflictByFile, fileCount{k, byFile[k]})
	}

	m.bounded = tallyBy(filter(m.facts, func(f fact) bool { return f.doc != "" }),
		fact.declKey, func(f fact) bool { return f.hasSep })

	// simd and archsimd document the same operations from different sources.
	// Target-specific implementation notes (such as Asm: and CPU Feature:)
	// and directives are ignored when comparing docs.
	m.crossPkg = new(tally[methodKey])
	docs := map[string]map[methodKey]string{"simd": {}, "archsimd": {}}
	for _, d := range m.decls {
		if d.doc != "" {
			docs[d.pkg][d.methodKey()] = d.doc
		}
	}
	for p, sd := range docs["simd"] {
		if ad, ok := docs["archsimd"][p]; ok {
			ssd, sad := stripDocNotes(sd), stripDocNotes(ad)
			m.crossPkg.add(p, ssd != "" && ssd == sad)
		}
	}

	// The hand-maintained mirror in package simd. This one deliberately spans
	// the whole of both files, residue included: midway owns those docs too.
	m.mirror = new(tally[methodKey])
	stubs, emul := map[methodKey]string{}, map[methodKey]string{}
	for _, d := range api {
		switch d.file {
		case "simd_stubs.go":
			stubs[d.methodKey()] = d.doc
		case "simd_emulated.go":
			emul[d.methodKey()] = d.doc
		}
	}
	for p, s := range stubs {
		if e, ok := emul[p]; ok {
			m.mirror.add(p, s == e)
		}
	}

	// 7. The inputs being retired.
	cat := makeCategoriesInventory(genDir)
	m.categories = cat.allFields
	m.catEntries = cat.entries
	m.catUniq = cat.uniq
	m.catExported = cat.exported
	m.catCompilerOnly = cat.compilerOnly
	m.catExportedFields = cat.exportedFields
	m.comments = countMidwayComments(genDir)

	// Comments.yaml entries that spec already covers, by name.
	root := readYAML(commentsPath(genDir))
	eachEntry(root, func(k, v *yaml.Node) {
		if isString(v) && specNames[k.Value] {
			m.commentsSpecCovered++
		}
	})

	// Commutativity drift: spec vs categories.yaml exported entries.
	m.commutDrift = commutativityDrift(genDir, m.spec)
	return m
}

// nameRelation classifies one name by how its two receiver sets stand.
func nameRelation(name string, api, spec map[string]bool) nameRel {
	r := nameRel{name: name, api: len(api), spec: len(spec)}
	shared := 0
	for recv := range api {
		if spec[recv] {
			shared++
		} else {
			r.apiOnly = append(r.apiOnly, recv)
		}
	}
	slices.Sort(r.apiOnly)
	r.specOnly = len(spec) - shared
	switch {
	case len(spec) == 0:
		r.rel = relAPIOnly
	case len(api) == 0:
		r.rel = relSpecOnly
	case len(r.apiOnly) == 0 && r.specOnly == 0:
		r.rel = relEqual
	case len(r.apiOnly) == 0:
		r.rel = relSubset
	case r.specOnly == 0:
		r.rel = relSuperset
	case shared == 0:
		r.rel = relDisjoint
	default:
		r.rel = relOverlap
	}
	return r
}

// commutativityDrift counts exported categories.yaml entries whose commutative field
// disagrees with spec. Some go: values are regexps, so each is tried as one.
func commutativityDrift(genDir string, spec map[methodKey]*specgen.Func) int {
	var n int
	paths, _ := filepath.Glob(filepath.Join(genDir, "simdgen", "ops", "*", "categories.yaml"))
	slices.Sort(paths)
	for _, p := range paths {
		root := readYAML(p)
		if root == nil || root.Kind != yaml.SequenceNode {
			continue
		}
		for _, entry := range root.Content {
			if entry.Kind != yaml.MappingNode {
				continue
			}
			goName := mapValue(entry, "go")
			if goName == nil {
				continue
			}
			isExported := goName.Value != "" && !(goName.Value[0] >= 'a' && goName.Value[0] <= 'z')
			if !isExported || mapValue(entry, "noTypes") != nil {
				continue
			}
			commNode := mapValue(entry, "commutative")
			if commNode == nil {
				continue
			}
			catCommut := commNode.Value == "true"
			re, err := regexp.Compile("^" + goName.Value + "$")
			if err != nil {
				continue
			}
			for p, f := range spec {
				if re.MatchString(p.name) && f.Commutative != catCommut {
					n++
					break
				}
			}
		}
	}
	return n
}

func filter[T any](s []T, keep func(T) bool) []T {
	var out []T
	for _, v := range s {
		if keep(v) {
			out = append(out, v)
		}
	}
	return out
}

// specPair returns the (receiver, name) key for a spec function.
func specPair(f *specgen.Func) methodKey {
	var recv string
	if f.Recv.Type != nil {
		recv = f.Recv.Type.String()
	}
	return methodKey{recv, f.Name}
}

// stripDocNotes returns doc with target-specific implementation notes and
// directives removed, leaving only spec-owned paragraphs.
func stripDocNotes(doc string) string {
	var keptParas []string
	var curPara []string
	flush := func() bool {
		if len(curPara) == 0 {
			return true
		}
		first := strings.TrimSpace(curPara[0])
		if strings.HasPrefix(first, "//go:") || strings.HasPrefix(first, "//line ") {
			return false
		}
		text := strings.TrimPrefix(first, "// ")
		text = strings.TrimPrefix(text, "//")
		if specgen.HasNotePrefix(text) {
			return false
		}
		keptParas = append(keptParas, strings.Join(curPara, "\n"))
		curPara = nil
		return true
	}

	for _, line := range strings.Split(doc, "\n") {
		trimmed := strings.TrimSpace(line)
		if strings.HasPrefix(trimmed, "//go:") || strings.HasPrefix(trimmed, "//line ") {
			if !flush() {
				return strings.Join(keptParas, "\n//\n")
			}
			break
		}
		if trimmed == "//" {
			if !flush() {
				return strings.Join(keptParas, "\n//\n")
			}
			continue
		}
		curPara = append(curPara, line)
	}
	flush()
	return strings.Join(keptParas, "\n//\n")
}
