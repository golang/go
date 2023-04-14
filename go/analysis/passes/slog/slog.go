// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// TODO(jba) deduce which functions wrap the log/slog functions, and use the
// fact mechanism to propagate this information, so we can provide diagnostics
// for user-supplied wrappers.

package slog

import (
	_ "embed"
	"fmt"
	"go/ast"
	"go/token"
	"go/types"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/analysis/passes/inspect"
	"golang.org/x/tools/go/analysis/passes/internal/analysisutil"
	"golang.org/x/tools/go/ast/inspector"
	"golang.org/x/tools/go/types/typeutil"
)

//go:embed doc.go
var doc string

var Analyzer = &analysis.Analyzer{
	Name:     "slog",
	Doc:      analysisutil.MustExtractDoc(doc, "slog"),
	URL:      "https://pkg.go.dev/golang.org/x/tools/go/analysis/passes/slog",
	Requires: []*analysis.Analyzer{inspect.Analyzer},
	Run:      run,
}

var stringType = types.Universe.Lookup("string").Type()

// A position describes what is expected to appear in an argument position.
type position int

const (
	// key is an argument position that should hold a string key or an Attr.
	key position = iota
	// value is an argument position that should hold a value.
	value
	// unknown represents that we do not know if position should hold a key or a value.
	unknown
)

func run(pass *analysis.Pass) (any, error) {
	inspect := pass.ResultOf[inspect.Analyzer].(*inspector.Inspector)
	nodeFilter := []ast.Node{
		(*ast.CallExpr)(nil),
	}
	inspect.Preorder(nodeFilter, func(node ast.Node) {
		call := node.(*ast.CallExpr)
		fn := typeutil.StaticCallee(pass.TypesInfo, call)
		if fn == nil {
			return // not a static call
		}
		if call.Ellipsis != token.NoPos {
			return // skip calls with "..." args
		}
		skipArgs, ok := kvFuncSkipArgs(fn)
		if !ok {
			// Not a slog function that takes key-value pairs.
			return
		}
		if len(call.Args) <= skipArgs {
			// Too few args; perhaps there are no k-v pairs.
			return
		}

		// Check this call.
		// The first position should hold a key or Attr.
		pos := key
		sawUnknown := false
		for _, arg := range call.Args[skipArgs:] {
			t := pass.TypesInfo.Types[arg].Type
			switch pos {
			case key:
				// Expect a string or Attr.
				switch {
				case t == stringType:
					pos = value
				case isAttr(t):
					pos = key
				case types.IsInterface(t):
					// We don't know if this arg is a string or an Attr, so we don't know what to expect next.
					// (We could see if one of interface's methods isn't a method of Attr, and thus know
					// for sure that this type is definitely not a string or Attr, but it doesn't seem
					// worth the effort for such an unlikely case.)
					pos = unknown
				default:
					// Definitely not a key.
					pass.ReportRangef(call, "%s arg %q should be a string or a slog.Attr (possible missing key or value)",
						shortName(fn), analysisutil.Format(pass.Fset, arg))
					// Assume this was supposed to be a value, and expect a key next.
					pos = key
				}

			case value:
				// Anything can appear in this position.
				// The next position should be a key.
				pos = key

			case unknown:
				// We don't know anything about this position, but all hope is not lost.
				if t != stringType && !isAttr(t) && !types.IsInterface(t) {
					// This argument is definitely not a key.
					//
					// The previous argument could have been a key, in which case this is the
					// corresponding value, and the next position should hold another key.
					// We will assume that.
					pos = key
					// Another possibility: the previous argument was an Attr, and this is
					// a value incorrectly placed in a key position.
					// If we assumed this case instead, we might produce a false positive
					// (since the first case might actually hold).

					// Once we encounter an unknown position, we can never be
					// sure if a problem at the end of the call is due to a
					// missing final value, or a non-key in key position.
					sawUnknown = true
				}
			}
		}
		if pos == value {
			if sawUnknown {
				pass.ReportRangef(call, "call to %s has a missing or misplaced value", shortName(fn))
			} else {
				pass.ReportRangef(call, "call to %s missing a final value", shortName(fn))
			}
		}
	})
	return nil, nil
}

func isAttr(t types.Type) bool {
	return t.String() == "log/slog.Attr"
}

// shortName returns a name for the function that is shorter than FullName.
// Examples:
//
//	"slog.Info" (instead of "log/slog.Info")
//	"slog.Logger.With" (instead of "(*log/slog.Logger).With")
func shortName(fn *types.Func) string {
	var r string
	if recv := fn.Type().(*types.Signature).Recv(); recv != nil {
		t := recv.Type()
		if pt, ok := t.(*types.Pointer); ok {
			t = pt.Elem()
		}
		if nt, ok := t.(*types.Named); ok {
			r = nt.Obj().Name()
		} else {
			r = recv.Type().String()
		}
		r += "."
	}
	return fmt.Sprintf("%s.%s%s", fn.Pkg().Name(), r, fn.Name())
}

// If fn is a slog function that has a ...any parameter for key-value pairs,
// kvFuncSkipArgs returns the number of arguments to skip over to reach the
// corresponding arguments, and true.
// Otherwise it returns (0, false).
func kvFuncSkipArgs(fn *types.Func) (int, bool) {
	if pkg := fn.Pkg(); pkg == nil || pkg.Path() != "log/slog" {
		return 0, false
	}
	recv := fn.Type().(*types.Signature).Recv()
	if recv == nil {
		// TODO: If #59204 is accepted, uncomment the lines below.
		// if fn.Name() == "Group" {
		// 	return 0, true
		// }
		skip, ok := slogOutputFuncs[fn.Name()]
		return skip, ok
	}
	var recvName string
	if pt, ok := recv.Type().(*types.Pointer); ok {
		if nt, ok := pt.Elem().(*types.Named); ok {
			recvName = nt.Obj().Name()
		}
	}
	if recvName == "" {
		return 0, false
	}
	// The methods on *Logger include all the top-level output methods, as well as "With".
	if recvName == "Logger" {
		if fn.Name() == "With" {
			return 0, true
		}
		skip, ok := slogOutputFuncs[fn.Name()]
		return skip, ok
	}
	if recvName == "Record" && fn.Name() == "Add" {
		return 0, true
	}
	return 0, false
}

// The names of top-level functions and *Logger methods in log/slog that take
// ...any for key-value pairs, mapped to the number of initial args to skip in
// order to get to the ones that match the ...any parameter.
var slogOutputFuncs = map[string]int{
	"Debug":    1,
	"Info":     1,
	"Warn":     1,
	"Error":    1,
	"DebugCtx": 2,
	"InfoCtx":  2,
	"WarnCtx":  2,
	"ErrorCtx": 2,
	"Log":      3,
}
