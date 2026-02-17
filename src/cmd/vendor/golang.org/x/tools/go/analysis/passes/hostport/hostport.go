// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package hostport defines an analyzer for calls to net.Dial with
// addresses of the form "%s:%d" or "%s:%s", which work only with IPv4.
package hostport

import (
	"fmt"
	"go/ast"
	"go/constant"
	"go/token"
	"go/types"
	"strconv"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/analysis/passes/inspect"
	"golang.org/x/tools/go/types/typeutil"
	typeindexanalyzer "golang.org/x/tools/internal/analysis/typeindex"
	"golang.org/x/tools/internal/typesinternal/typeindex"
)

const Doc = `check format of addresses passed to net.Dial

This analyzer flags code that produce network address strings using
fmt.Sprintf, as in this example:

    addr := fmt.Sprintf("%s:%d", host, 12345) // "will not work with IPv6"
    ...
    conn, err := net.Dial("tcp", addr)       // "when passed to dial here"

The analyzer suggests a fix to use the correct approach, a call to
net.JoinHostPort:

    addr := net.JoinHostPort(host, "12345")
    ...
    conn, err := net.Dial("tcp", addr)

A similar diagnostic and fix are produced for a format string of "%s:%s".
`

var Analyzer = &analysis.Analyzer{
	Name:     "hostport",
	Doc:      Doc,
	URL:      "https://pkg.go.dev/golang.org/x/tools/go/analysis/passes/hostport",
	Requires: []*analysis.Analyzer{inspect.Analyzer, typeindexanalyzer.Analyzer},
	Run:      run,
}

func run(pass *analysis.Pass) (any, error) {
	var (
		index      = pass.ResultOf[typeindexanalyzer.Analyzer].(*typeindex.Index)
		info       = pass.TypesInfo
		fmtSprintf = index.Object("fmt", "Sprintf")
	)
	if !index.Used(fmtSprintf) {
		return nil, nil // fast path: package doesn't use fmt.Sprintf
	}

	// checkAddr reports a diagnostic (and returns true) if e
	// is a call of the form fmt.Sprintf("%s:%d", ...).
	// The diagnostic includes a fix.
	//
	// dialCall is non-nil if the Dial call is non-local
	// but within the same file.
	checkAddr := func(e ast.Expr, dialCall *ast.CallExpr) {
		if call, ok := e.(*ast.CallExpr); ok &&
			len(call.Args) == 3 &&
			typeutil.Callee(info, call) == fmtSprintf {

			// Examine format string.
			formatArg := call.Args[0]
			if tv := info.Types[formatArg]; tv.Value != nil {
				numericPort := false
				format := constant.StringVal(tv.Value)
				switch format {
				case "%s:%d":
					// Have: fmt.Sprintf("%s:%d", host, port)
					numericPort = true

				case "%s:%s":
					// Have: fmt.Sprintf("%s:%s", host, portStr)
					// Keep port string as is.

				default:
					return
				}

				// Use granular edits to preserve original formatting.
				edits := []analysis.TextEdit{
					{
						// Replace fmt.Sprintf with net.JoinHostPort.
						Pos:     call.Fun.Pos(),
						End:     call.Fun.End(),
						NewText: []byte("net.JoinHostPort"),
					},
					{
						// Delete format string.
						Pos: formatArg.Pos(),
						End: call.Args[1].Pos(),
					},
				}

				// Turn numeric port into a string.
				if numericPort {
					port := call.Args[2]

					// Is port an integer literal?
					//
					// (Don't allow arbitrary constants k otherwise the
					// transformation k => fmt.Sprintf("%d", "123")
					// loses the symbolic connection to k.)
					var kPort int64 = -1
					if lit, ok := port.(*ast.BasicLit); ok && lit.Kind == token.INT {
						if v, err := strconv.ParseInt(lit.Value, 0, 64); err == nil {
							kPort = v
						}
					}
					if kPort >= 0 {
						// literal: 0x7B  => "123"
						edits = append(edits, analysis.TextEdit{
							Pos:     port.Pos(),
							End:     port.End(),
							NewText: fmt.Appendf(nil, `"%d"`, kPort), // (decimal)
						})
					} else {
						// non-literal: port => fmt.Sprintf("%d", port)
						edits = append(edits, []analysis.TextEdit{
							{
								Pos:     port.Pos(),
								End:     port.Pos(),
								NewText: []byte(`fmt.Sprintf("%d", `),
							},
							{
								Pos:     port.End(),
								End:     port.End(),
								NewText: []byte(`)`),
							},
						}...)
					}
				}

				// Refer to Dial call, if not adjacent.
				suffix := ""
				if dialCall != nil {
					suffix = fmt.Sprintf(" (passed to net.Dial at L%d)",
						pass.Fset.Position(dialCall.Pos()).Line)
				}

				pass.Report(analysis.Diagnostic{
					// Highlight the format string.
					Pos:     formatArg.Pos(),
					End:     formatArg.End(),
					Message: fmt.Sprintf("address format %q does not work with IPv6%s", format, suffix),
					SuggestedFixes: []analysis.SuggestedFix{{
						Message:   "Replace fmt.Sprintf with net.JoinHostPort",
						TextEdits: edits,
					}},
				})
			}
		}
	}

	// Check address argument of each call to net.Dial et al.
	for _, callee := range []types.Object{
		index.Object("net", "Dial"),
		index.Object("net", "DialTimeout"),
		index.Selection("net", "Dialer", "Dial"),
	} {
		for curCall := range index.Calls(callee) {
			call := curCall.Node().(*ast.CallExpr)
			switch address := call.Args[1].(type) {
			case *ast.CallExpr:
				if len(call.Args) == 2 { // avoid spread-call edge case
					// net.Dial("tcp", fmt.Sprintf("%s:%d", ...))
					checkAddr(address, nil)
				}

			case *ast.Ident:
				// addr := fmt.Sprintf("%s:%d", ...)
				// ...
				// net.Dial("tcp", addr)

				// Search for decl of addrVar within common ancestor of addrVar and Dial call.
				// TODO(adonovan): abstract "find RHS of statement that assigns var v".
				// TODO(adonovan): reject if there are other assignments to var v.
				if addrVar, ok := info.Uses[address].(*types.Var); ok {
					if curId, ok := index.Def(addrVar); ok {
						// curIdent is the declaring ast.Ident of addr.
						switch parent := curId.Parent().Node().(type) {
						case *ast.AssignStmt:
							if len(parent.Rhs) == 1 {
								// Have: addr := fmt.Sprintf("%s:%d", ...)
								checkAddr(parent.Rhs[0], call)
							}

						case *ast.ValueSpec:
							if len(parent.Values) == 1 {
								// Have: var addr = fmt.Sprintf("%s:%d", ...)
								checkAddr(parent.Values[0], call)
							}
						}
					}
				}
			}
		}
	}
	return nil, nil
}
