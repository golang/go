// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gotestchecker

// This file defines the analyzer graph, its validation, and the
// metadata query (-describe), whose result is a [protocol.Metadata].

import (
	"encoding/json"
	"fmt"
	"go/analysis"
	"internal/analysis/gotestchecker/protocol"
	"log"
	"net/url"
	"os"
	"reflect"
	"runtime"
	"strings"
	"unicode"
)

// writeMetadata writes the metadata for the specified variables to
// the named file.
func writeMetadata(filename string, vars []Var) error {
	g := newGraph(vars)
	analyzers := []*protocol.Analyzer{}
	for _, a := range g.analyzers {
		var requires []int
		for _, req := range a.Requires {
			requires = append(requires, g.index[req])
		}
		run := runFunc(a)
		analyzers = append(analyzers, &protocol.Analyzer{
			Name:     a.Name,
			Doc:      a.Doc,
			URL:      a.URL,
			RunFunc:  run,
			PkgPath:  funcPkgPath(run),
			Requires: requires,
			Facts:    len(a.FactTypes) > 0,
		})
	}
	md := &protocol.Metadata{
		Vars:      g.vars,
		Analyzers: analyzers,
	}
	data, err := json.Marshal(md)
	if err != nil {
		log.Fatal(err) // can't happen
	}
	return os.WriteFile(filename, data, 0666)
}

// A graph is the validated graph of analyzers reachable from the
// variables (see [validate]).
type graph struct {
	vars      []*protocol.VarInfo        // description of each variable, by position in the argument of Main
	analyzers []*analysis.Analyzer       // valid analyzers (Requires-closed) in canonical order
	index     map[*analysis.Analyzer]int // index of each analyzer in order (its identifier in the protocol)
}

// newGraph computes the graph of analyzers reachable (via
// Analyzer.Requires) from the specified variables. We assume vars are
// initialized deterministically so the graph is the same on each
// execution.
func newGraph(vars []Var) *graph {
	g := &graph{
		vars:  make([]*protocol.VarInfo, len(vars)),
		index: make(map[*analysis.Analyzer]int),
	}

	// add ensures 'a' and its Requires dependencies
	// are in the canonical analyzer ordering.
	var add func(a *analysis.Analyzer)
	add = func(a *analysis.Analyzer) {
		if _, ok := g.index[a]; !ok {
			for _, req := range a.Requires {
				add(req)
			}
			g.index[a] = len(g.analyzers)
			g.analyzers = append(g.analyzers, a)
		}
	}

	// Validate each variable and each member's Requires closure,
	// and add the valid members (and their Requires closures) to
	// the canonical order.
	for i, v := range vars {
		info := &protocol.VarInfo{Symbol: v.Symbol}
		g.vars[i] = info
		analyzers, isSuite, err := varAnalyzers(v.Value)
		if err != nil {
			info.Error = err.Error() // invalid Var.Value
			continue
		}
		info.IsSuite = isSuite
		for _, a := range analyzers {
			if err := validate(a); err != "" {
				// invalid analyzer graph
				info.Members = append(info.Members, &protocol.Member{Name: a.Name, Analyzer: -1, Error: err})
				continue
			}
			// Valid: Add it to canonical ordering.
			add(a)
			info.Members = append(info.Members, &protocol.Member{
				Name:     a.Name,
				Analyzer: g.index[a],
			})
		}
	}
	return g
}

// varAnalyzers returns the distinct analyzers of a variable's value
// (the analyzer itself, or the elements of the suite, in order), and
// whether it is a suite.
func varAnalyzers(value any) ([]*analysis.Analyzer, bool, error) {
	switch value := value.(type) {
	case *analysis.Analyzer:
		if value == nil {
			return nil, false, fmt.Errorf("analyzer is nil")
		}
		return []*analysis.Analyzer{value}, false, nil

	case []*analysis.Analyzer:
		var (
			analyzers []*analysis.Analyzer
			seen      = make(map[*analysis.Analyzer]bool)
			byName    = make(map[string]bool)
		)
		for i, a := range value {
			if a == nil {
				return nil, false, fmt.Errorf("suite element %d is nil", i)
			}
			if seen[a] {
				// Repeated elements (by pointer) are one member.
				continue
			}
			seen[a] = true
			if byName[a.Name] {
				return nil, false, fmt.Errorf("suite contains distinct analyzers named %q", a.Name)
			}
			byName[a.Name] = true
			analyzers = append(analyzers, a)
		}
		return analyzers, true, nil
	}

	return nil, false, fmt.Errorf("variable has type %T, want *go/analysis.Analyzer or []*go/analysis.Analyzer", value)
}

// validate checks the Requires-closure of a single member analyzer,
// and returns the first error found, or "".
//
// Unlike [golang.org/x/tools/go/analysis.Validate], each member (and
// its Requires closure) is validated separately, so that each error
// is attributable to a single member of a single variable, and thus
// to the annotations that enable it. This ensures compositionality:
// if go test X and go test Y succeed, then so does go test X Y, even
// though the latter builds a single tool containing the analyzers of
// both.
func validate(root *analysis.Analyzer) string {
	var err string
	errorf := func(format string, args ...any) bool {
		err = fmt.Sprintf(format, args...)
		return false
	}

	// coloring for cycle check
	const (
		white = iota
		grey
		black
	)
	color := make(map[*analysis.Analyzer]uint8)

	// In its cache keys for pass files, the go command identifies
	// an analyzer by its Name and Run function so distinct
	// analyzers that share both ("evil twins") must not appear in
	// one Requires closure. (See planAnalysis function in go
	// test.)
	//
	// Twins in different Requires-closures are not detected: this is
	// unsound, but the tool cannot distinguish two closures of the
	// same func literal.
	//
	// Neither Name nor the Run function name unique, but they are
	// both nearly unique, and their collisions are uncorrelated.
	// That is, when Analyzer.Names conflict, it is usually
	// between unrelated packages from different authors, so the
	// Analyzer.Run functions have different symbols; and when two
	// closely-related analyzers use the same Run function, their
	// author invariably gives them different names.
	// Evil twins thus do not arise by accident.
	type twinKey struct{ name, run string }
	twins := make(map[twinKey]bool)

	var visit func(a *analysis.Analyzer, path []string) bool
	visit = func(a *analysis.Analyzer, path []string) bool {
		if a == nil {
			return errorf("nil analyzer in Requires (via %s)", strings.Join(path, " -> "))
		}
		descr := func() string {
			return fmt.Sprintf("analyzer %q (via %s)", a.Name, strings.Join(path, " -> "))
		}
		path = append(path, a.Name)
		switch color[a] {
		case grey:
			return errorf("%s has cycle in Requires graph", descr())
		case black:
			return true
		}
		color[a] = grey

		if !validIdent(a.Name) {
			return errorf("%s has an invalid name", descr())
		}
		if a.Doc == "" {
			return errorf("%s is undocumented", descr())
		}
		if a.Run == nil {
			return errorf("%s has nil Run", descr())
		}
		k := twinKey{a.Name, runFunc(a)}
		if twins[k] {
			return errorf("%s and another Analyzer conflict in both Name and Run function (%s)", descr(), k.run)
		}
		twins[k] = true

		// Unlike analysis.Validate, Analyzer.FactTypes need
		// not be unique across analyzers: our fact encoding
		// segregates fact by their analyzers.
		for _, f := range a.FactTypes {
			if t := reflect.TypeOf(f); t == nil || t.Kind() != reflect.Pointer {
				return errorf("%s has fact type %v, which is not a pointer", descr(), t)
			}
		}

		// Analyzer.Flags are ignored.

		for _, req := range a.Requires {
			if !visit(req, path) {
				return false
			}
		}
		color[a] = black
		return true
	}
	visit(root, nil)
	return err
}

// runFunc returns the symbol name of the analyzer's Run function.
func runFunc(a *analysis.Analyzer) string {
	return runtime.FuncForPC(reflect.ValueOf(a.Run).Pointer()).Name()
}

// funcPkgPath returns the import path of the package that declares
// the function with the specified symbol name (see runtime.Func.Name),
// such as "example.com/a%2eb.(*T).f" (of package "example.com/a.b").
//
// It parses the name as does the unexported runtime.funcpkgpath (used
// by GODEBUG=inittrace): the import path ends at the first dot after
// the last slash. This is sound because the linker escapes the dots
// in the last segment of the import path (see
// [cmd/internal/objabi.PathToPrefix]), and because the function part
// of the name contains no slash: Func.Name abbreviates the type
// arguments of generic functions, which may contain slashes, as
// "[...]".
//
// Unlike runtime.funcpkgpath, it then reverses the linker's
// escaping, defined by [cmd/internal/objabi.PathToPrefix], which
// replaces with "%xx" each byte of the import path that is a dot in
// the last segment, '%', '"', a space or control character, or
// non-ASCII. Since every escaped byte has the form "%xx" and '%'
// itself is escaped, [url.PathUnescape], which decodes each "%xx" and
// nothing else, is its inverse.
//
// The result may not be the import path of a linked package: the
// symbol of a function may be chosen by //go:linkname, and the
// compiler escapes the import path twice in the symbol of a closure
// within a function inlined from another package (for example,
// "example.com/a%252eb.F.func1").
func funcPkgPath(name string) string {
	// The import path ends at the first dot after the last slash.
	// (This is equivalent to the loops of runtime.funcpkgpath.)
	path := name
	slash := strings.LastIndexByte(path, '/')
	if dot := strings.IndexByte(path[slash+1:], '.'); dot >= 0 {
		path = path[:slash+1+dot]
	}

	// Reverse the escaping of [cmd/internal/objabi.PathToPrefix].
	// If any escape is malformed (which the linker never produces),
	// the path is returned as is.
	if p, err := url.PathUnescape(path); err == nil {
		path = p
	}
	return path
}

func validIdent(name string) bool {
	for i, r := range name {
		if !(r == '_' || unicode.IsLetter(r) || i > 0 && unicode.IsDigit(r)) {
			return false
		}
	}
	return name != ""
}
