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
	"golang.org/x/tools/internal/typesinternal"
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
	var attrType types.Type // The type of slog.Attr
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
		// Here we know that fn.Pkg() is "log/slog".
		if attrType == nil {
			attrType = fn.Pkg().Scope().Lookup("Attr").Type()
		}

		if isMethodExpr(pass.TypesInfo, call) {
			// Call is to a method value. Skip the first argument.
			skipArgs++
		}
		if len(call.Args) <= skipArgs {
			// Too few args; perhaps there are no k-v pairs.
			return
		}

		// Check this call.
		// The first position should hold a key or Attr.
		pos := key
		var unknownArg ast.Expr // nil or the last unknown argument
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
					// As we do not do dataflow, we do not know what the dynamic type is.
					// But we might be able to learn enough to make a decision.
					if types.AssignableTo(stringType, t) {
						// t must be an empty interface. So it can also be an Attr.
						// We don't know enough to make an assumption.
						pos = unknown
						continue
					} else if attrType != nil && types.AssignableTo(attrType, t) {
						// Assume it is an Attr.
						pos = key
						continue
					}
					// Can't be either a string or Attr. Definitely an error.
					fallthrough
				default:
					if unknownArg == nil {
						pass.ReportRangef(arg, "%s arg %q should be a string or a slog.Attr (possible missing key or value)",
							shortName(fn), analysisutil.Format(pass.Fset, arg))
					} else {
						pass.ReportRangef(arg, "%s arg %q should probably be a string or a slog.Attr (previous arg %q cannot be a key)",
							shortName(fn), analysisutil.Format(pass.Fset, arg), analysisutil.Format(pass.Fset, unknownArg))
					}
					// Stop here so we report at most one missing key per call.
					return
				}

			case value:
				// Anything can appear in this position.
				// The next position should be a key.
				pos = key

			case unknown:
				// Once we encounter an unknown position, we can never be
				// sure if a problem later or at the end of the call is due to a
				// missing final value, or a non-key in key position.
				// In both cases, unknownArg != nil.
				unknownArg = arg

				// We don't know what is expected about this position, but all hope is not lost.
				if t != stringType && !isAttr(t) && !types.IsInterface(t) {
					// This argument is definitely not a key.
					//
					// unknownArg cannot have been a key, in which case this is the
					// corresponding value, and the next position should hold another key.
					pos = key
				}
			}
		}
		if pos == value {
			if unknownArg == nil {
				pass.ReportRangef(call, "call to %s missing a final value", shortName(fn))
			} else {
				pass.ReportRangef(call, "call to %s has a missing or misplaced value", shortName(fn))
			}
		}
	})
	return nil, nil
}

func isAttr(t types.Type) bool {
	return analysisutil.IsNamedType(t, "log/slog", "Attr")
}

// shortName returns a name for the function that is shorter than FullName.
// Examples:
//
//	"slog.Info" (instead of "log/slog.Info")
//	"slog.Logger.With" (instead of "(*log/slog.Logger).With")
func shortName(fn *types.Func) string {
	var r string
	if recv := fn.Type().(*types.Signature).Recv(); recv != nil {
		if _, named := typesinternal.ReceiverNamed(recv); named != nil {
			r = named.Obj().Name()
		} else {
			r = recv.Type().String() // anon struct/interface
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
	var recvName string // by default a slog package function
	if recv := fn.Type().(*types.Signature).Recv(); recv != nil {
		_, named := typesinternal.ReceiverNamed(recv)
		if named == nil {
			return 0, false // anon struct/interface
		}
		recvName = named.Obj().Name()
	}
	skip, ok := kvFuncs[recvName][fn.Name()]
	return skip, ok
}

// The names of functions and methods in log/slog that take
// ...any for key-value pairs, mapped to the number of initial args to skip in
// order to get to the ones that match the ...any parameter.
// The first key is the dereferenced receiver type name, or "" for a function.
var kvFuncs = map[string]map[string]int{
	"": {
		"Debug":        1,
		"Info":         1,
		"Warn":         1,
		"Error":        1,
		"DebugContext": 2,
		"InfoContext":  2,
		"WarnContext":  2,
		"ErrorContext": 2,
		"Log":          3,
		"Group":        1,
	},
	"Logger": {
		"Debug":        1,
		"Info":         1,
		"Warn":         1,
		"Error":        1,
		"DebugContext": 2,
		"InfoContext":  2,
		"WarnContext":  2,
		"ErrorContext": 2,
		"Log":          3,
		"With":         0,
	},
	"Record": {
		"Add": 0,
	},
}

// isMethodExpr reports whether a call is to a MethodExpr.
func isMethodExpr(info *types.Info, c *ast.CallExpr) bool {
	s, ok := c.Fun.(*ast.SelectorExpr)
	if !ok {
		return false
	}
	sel := info.Selections[s]
	return sel != nil && sel.Kind() == types.MethodExpr
}
