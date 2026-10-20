// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lint

import (
	"bytes"
	"cmp"
	"regexp"
	"slices"
	"strings"

	"cmd/compile/internal/syntax"
)

// ZeroAllocRule is the ID of the diagnostic for a function marked
// //logos:zeroalloc that performs a heap allocation (SPEC Sec 22.4). It is
// not a configurable lint rule: the directive is the author's own assertion,
// so no logos.toml severity lowers the diagnostic and no //logos:allow
// directive silences it.
const ZeroAllocRule = "perf::zeroalloc_violation"

// zeroAllocMarker is the text every //logos:zeroalloc comment contains.
const zeroAllocMarker = "//logos:zeroalloc"

// zeroAllocDirective matches //logos:zeroalloc, optionally followed by a
// space and free text.
var zeroAllocDirective = regexp.MustCompile(`^//logos:zeroalloc(?:\s.*)?$`)

// ZeroAllocFunc is a function marked //logos:zeroalloc: its declared name,
// and the file and line of its declaration, the line of the token after
// func, where the compiler finds the function again.
type ZeroAllocFunc struct {
	Name string
	File string
	Line uint
}

// ZeroAllocFinding is one heap allocation site that escape analysis found
// inside a marked function: the position of the allocating expression, the
// declared name of the marked function, and the reason.
type ZeroAllocFinding struct {
	Func      string
	File      string
	Line, Col uint
	Reason    string
}

// ZeroAllocFuncs returns the functions of in marked //logos:zeroalloc, and
// one result for each directive that marks nothing: a comment that is not on
// its own line, or that does not end directly above a function declaration,
// so a blank line between the directive and the function leaves the function
// unchecked and is an error.
func ZeroAllocFuncs(in *Input) ([]ZeroAllocFunc, []Result) {
	mode := syntax.Mode(0)
	if in.Kind == KindLgo {
		mode = syntax.LogosMode
	}
	var funcs []ZeroAllocFunc
	var misplaced []Result
	for _, f := range in.Files {
		src, ok := in.Sources[f]
		if !ok || !bytes.Contains(src, []byte(zeroAllocMarker)) {
			continue
		}
		fs, ms := fileZeroAlloc(f, src, mode)
		funcs = append(funcs, fs...)
		misplaced = append(misplaced, ms...)
	}
	return funcs, misplaced
}

// fileZeroAlloc scans the comments of one file with the given parser mode.
func fileZeroAlloc(f *syntax.File, src []byte, mode syntax.Mode) ([]ZeroAllocFunc, []Result) {
	lines := strings.Split(string(src), "\n")
	decls := funcDeclsByLine(f)
	fileBase := f.Pos().FileBase()
	var funcs []ZeroAllocFunc
	var misplaced []Result
	syntax.CommentsDoMode(bytes.NewReader(src), mode, func(line, col uint, text string) {
		if !zeroAllocDirective.MatchString(text) {
			return
		}
		decl, ok := decls[nextCodeLine(lines, line)]
		if !ok || !ownLine(lines, line, col) {
			pos := syntax.MakePos(fileBase, line, col)
			misplaced = append(misplaced, Result{
				Rule:     ZeroAllocRule,
				Severity: Forbid,
				Pos:      pos,
				End:      syntax.MakePos(fileBase, line, col+uint(len(strings.TrimRight(text, " \t\r")))),
				Message:  zeroAllocMarker + " must be on its own line directly above a function declaration",
			})
			return
		}
		funcs = append(funcs, ZeroAllocFunc{Name: declName(decl), File: f.Pos().RelFilename(), Line: decl.Pos().Line()})
	})
	return funcs, misplaced
}

// funcDeclsByLine maps each line on which a function declaration starts to
// the declaration.
func funcDeclsByLine(f *syntax.File) map[uint]*syntax.FuncDecl {
	decls := make(map[uint]*syntax.FuncDecl)
	for _, d := range f.DeclList {
		if fn, ok := d.(*syntax.FuncDecl); ok {
			decls[fn.Pos().Line()] = fn
		}
	}
	return decls
}

// declName returns the name a diagnostic gives a function: the plain name,
// or the method expression form T.M and (*T).M.
func declName(d *syntax.FuncDecl) string {
	if d.Recv == nil {
		return d.Name.Value
	}
	recv := syntax.String(d.Recv.Type)
	if strings.HasPrefix(recv, "*") {
		recv = "(" + recv + ")"
	}
	return recv + "." + d.Name.Value
}

// ZeroAllocResults resolves the allocation sites of the marked functions to
// results, ordered by line. Every result is at forbid, because the directive
// is the assertion. It reports a reason once for each line of a function,
// because inlined bodies and generic instantiations are analyzed more than
// once and a compiler-made wrapper of a generic function sits at another
// column of the same line, and it ignores a site in a file the package does
// not hold: the body of a function from another package, inlined here,
// reaches the report at the call that inlined it.
func ZeroAllocResults(in *Input, found []ZeroAllocFinding) []Result {
	var out []Result
	for _, f := range uniqueAllocs(found) {
		pos, ok := filePos(in, f.File, f.Line, f.Col)
		if !ok {
			continue
		}
		out = append(out, Result{
			Rule:     ZeroAllocRule,
			Severity: Forbid,
			Pos:      pos,
			End:      pos,
			Message:  f.Func + " allocates: " + f.Reason,
		})
	}
	return out
}

// uniqueAllocs returns the findings ordered by line, then reason, with the
// later columns of a repeated reason on one line of one function removed.
func uniqueAllocs(found []ZeroAllocFinding) []ZeroAllocFinding {
	sorted := slices.Clone(found)
	slices.SortStableFunc(sorted, func(a, b ZeroAllocFinding) int {
		return cmp.Or(cmp.Compare(a.File, b.File), cmp.Compare(a.Line, b.Line), cmp.Compare(a.Func, b.Func), cmp.Compare(a.Reason, b.Reason), cmp.Compare(a.Col, b.Col))
	})
	return slices.CompactFunc(sorted, func(a, b ZeroAllocFinding) bool {
		return a.File == b.File && a.Line == b.Line && a.Func == b.Func && a.Reason == b.Reason
	})
}
