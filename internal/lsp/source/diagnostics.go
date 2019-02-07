// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package source

import (
	"bytes"
	"context"
	"fmt"
	"go/token"
	"sort"
	"strconv"
	"strings"
	"sync"

	"golang.org/x/tools/go/analysis/passes/asmdecl"
	"golang.org/x/tools/go/analysis/passes/assign"
	"golang.org/x/tools/go/analysis/passes/atomic"
	"golang.org/x/tools/go/analysis/passes/atomicalign"
	"golang.org/x/tools/go/analysis/passes/bools"
	"golang.org/x/tools/go/analysis/passes/buildtag"
	"golang.org/x/tools/go/analysis/passes/cgocall"
	"golang.org/x/tools/go/analysis/passes/composite"
	"golang.org/x/tools/go/analysis/passes/copylock"
	"golang.org/x/tools/go/analysis/passes/httpresponse"
	"golang.org/x/tools/go/analysis/passes/loopclosure"
	"golang.org/x/tools/go/analysis/passes/nilfunc"
	"golang.org/x/tools/go/analysis/passes/shift"
	"golang.org/x/tools/go/analysis/passes/stdmethods"
	"golang.org/x/tools/go/analysis/passes/structtag"
	"golang.org/x/tools/go/analysis/passes/unmarshal"
	"golang.org/x/tools/go/analysis/passes/unreachable"
	"golang.org/x/tools/go/analysis/passes/unsafeptr"
	"golang.org/x/tools/go/analysis/passes/unusedresult"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/analysis/passes/tests"

	"golang.org/x/tools/go/packages"
)

type Diagnostic struct {
	Range
	Message string
	Source  string
}

func Diagnostics(ctx context.Context, v View, uri URI) (map[string][]Diagnostic, error) {
	f, err := v.GetFile(ctx, uri)
	if err != nil {
		return nil, err
	}
	pkg, err := f.GetPackage()
	if err != nil {
		return nil, err
	}
	// Prepare the reports we will send for this package.
	reports := make(map[string][]Diagnostic)
	for _, filename := range pkg.GoFiles {
		reports[filename] = []Diagnostic{}
	}
	var parseErrors, typeErrors []packages.Error
	for _, err := range pkg.Errors {
		switch err.Kind {
		case packages.ParseError:
			parseErrors = append(parseErrors, err)
		case packages.TypeError:
			typeErrors = append(typeErrors, err)
		default:
			// ignore other types of errors
			continue
		}
	}
	// Don't report type errors if there are parse errors.
	diags := typeErrors
	if len(parseErrors) > 0 {
		diags = parseErrors
	}
	for _, diag := range diags {
		pos := errorPos(diag)
		diagFile, err := v.GetFile(ctx, ToURI(pos.Filename))
		if err != nil {
			continue
		}
		diagTok, err := diagFile.GetToken()
		if err != nil {
			continue
		}
		content, err := diagFile.Read()
		if err != nil {
			continue
		}
		end, err := identifierEnd(content, pos.Line, pos.Column)
		// Don't set a range if it's anything other than a type error.
		if err != nil || diag.Kind != packages.TypeError {
			end = 0
		}
		startPos := fromTokenPosition(diagTok, pos.Line, pos.Column)
		if !startPos.IsValid() {
			continue
		}
		endPos := fromTokenPosition(diagTok, pos.Line, pos.Column+end)
		if !endPos.IsValid() {
			continue
		}
		diagnostic := Diagnostic{
			Range: Range{
				Start: startPos,
				End:   endPos,
			},
			Message: diag.Msg,
		}
		if _, ok := reports[pos.Filename]; ok {
			reports[pos.Filename] = append(reports[pos.Filename], diagnostic)
		}
	}
	if len(diags) > 0 {
		return reports, nil
	}
	// Type checking and parsing succeeded. Run analyses.
	runAnalyses(pkg, func(a *analysis.Analyzer, diag analysis.Diagnostic) {
		pos := pkg.Fset.Position(diag.Pos)
		category := a.Name
		if diag.Category != "" {
			category += "." + category
		}

		reports[pos.Filename] = append(reports[pos.Filename], Diagnostic{
			Source:  category,
			Range:   Range{Start: diag.Pos, End: diag.Pos},
			Message: fmt.Sprintf(diag.Message),
		})
	})

	return reports, nil
}

// fromTokenPosition converts a token.Position (1-based line and column
// number) to a token.Pos (byte offset value). This requires the token.File
// to which the token.Pos belongs.
func fromTokenPosition(f *token.File, line, col int) token.Pos {
	linePos := lineStart(f, line)
	// TODO: This is incorrect, as pos.Column represents bytes, not characters.
	// This needs to be handled to address golang.org/issue/29149.
	return linePos + token.Pos(col-1)
}

func errorPos(pkgErr packages.Error) token.Position {
	remainder1, first, hasLine := chop(pkgErr.Pos)
	remainder2, second, hasColumn := chop(remainder1)
	var pos token.Position
	if hasLine && hasColumn {
		pos.Filename = remainder2
		pos.Line = second
		pos.Column = first
	} else if hasLine {
		pos.Filename = remainder1
		pos.Line = first
	}
	return pos
}

func chop(text string) (remainder string, value int, ok bool) {
	i := strings.LastIndex(text, ":")
	if i < 0 {
		return text, 0, false
	}
	v, err := strconv.ParseInt(text[i+1:], 10, 64)
	if err != nil {
		return text, 0, false
	}
	return text[:i], int(v), true
}

// identifierEnd returns the length of an identifier within a string,
// given the starting line and column numbers of the identifier.
func identifierEnd(content []byte, l, c int) (int, error) {
	lines := bytes.Split(content, []byte("\n"))
	if len(lines) < l {
		return 0, fmt.Errorf("invalid line number: got %v, but only %v lines", l, len(lines))
	}
	line := lines[l-1]
	if len(line) < c {
		return 0, fmt.Errorf("invalid column number: got %v, but the length of the line is %v", c, len(line))
	}
	return bytes.IndexAny(line[c-1:], " \n,():;[]"), nil
}

func runAnalyses(pkg *packages.Package, report func(a *analysis.Analyzer, diag analysis.Diagnostic)) error {
	// These are the analyses in the vetsuite, except for lostcancel and printf.
	// Those rely on facts from other packages.
	// TODO(matloob): Add fact support.
	analyzers := []*analysis.Analyzer{
		asmdecl.Analyzer,
		assign.Analyzer,
		atomic.Analyzer,
		atomicalign.Analyzer,
		bools.Analyzer,
		buildtag.Analyzer,
		cgocall.Analyzer,
		composite.Analyzer,
		copylock.Analyzer,
		httpresponse.Analyzer,
		loopclosure.Analyzer,
		nilfunc.Analyzer,
		shift.Analyzer,
		stdmethods.Analyzer,
		structtag.Analyzer,
		tests.Analyzer,
		unmarshal.Analyzer,
		unreachable.Analyzer,
		unsafeptr.Analyzer,
		unusedresult.Analyzer,
	}

	// Execute actions in parallel. Based on unitchecker's analysis execution logic.
	type action struct {
		once        sync.Once
		result      interface{}
		err         error
		diagnostics []analysis.Diagnostic
	}
	actions := make(map[*analysis.Analyzer]*action)

	var registerActions func(a *analysis.Analyzer)
	registerActions = func(a *analysis.Analyzer) {
		act, ok := actions[a]
		if !ok {
			act = new(action)
			for _, req := range a.Requires {
				registerActions(req)
			}
			actions[a] = act
		}
	}
	for _, a := range analyzers {
		registerActions(a)
	}

	// In parallel, execute the DAG of analyzers.
	var exec func(a *analysis.Analyzer) *action
	var execAll func(analyzers []*analysis.Analyzer)
	exec = func(a *analysis.Analyzer) *action {
		act := actions[a]
		act.once.Do(func() {
			if len(a.FactTypes) > 0 {
				panic("for analyzer " + a.Name + " modular analyses needing facts are not yet supported")
			}

			execAll(a.Requires) // prefetch dependencies in parallel

			// The inputs to this analysis are the
			// results of its prerequisites.
			inputs := make(map[*analysis.Analyzer]interface{})
			var failed []string
			for _, req := range a.Requires {
				reqact := exec(req)
				if reqact.err != nil {
					failed = append(failed, req.String())
					continue
				}
				inputs[req] = reqact.result
			}

			// Report an error if any dependency failed.
			if failed != nil {
				sort.Strings(failed)
				act.err = fmt.Errorf("failed prerequisites: %s", strings.Join(failed, ", "))
				return
			}

			pass := &analysis.Pass{
				Analyzer:   a,
				Fset:       pkg.Fset,
				Files:      pkg.Syntax,
				OtherFiles: pkg.OtherFiles,
				Pkg:        pkg.Types,
				TypesInfo:  pkg.TypesInfo,
				TypesSizes: pkg.TypesSizes,
				ResultOf:   inputs,
				Report:     func(d analysis.Diagnostic) { act.diagnostics = append(act.diagnostics, d) },
			}

			act.result, act.err = a.Run(pass)
		})
		return act
	}
	execAll = func(analyzers []*analysis.Analyzer) {
		var wg sync.WaitGroup
		for _, a := range analyzers {
			wg.Add(1)
			go func(a *analysis.Analyzer) {
				_ = exec(a)
				wg.Done()
			}(a)
		}
		wg.Wait()
	}

	execAll(analyzers)

	// Report diagnostics and errors from root analyzers.
	for _, a := range analyzers {
		act := actions[a]
		if act.err != nil {
			// TODO(matloob): This isn't quite right: we might return a failed prerequisites error,
			// which isn't super useful...
			return act.err
		}
		for _, diag := range act.diagnostics {
			report(a, diag)
		}
	}

	return nil
}
