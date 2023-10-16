// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package source

import (
	"context"
	"fmt"
	"go/ast"
	"go/parser"
	"go/types"

	"golang.org/x/tools/go/ast/astutil"
	"golang.org/x/tools/go/types/typeutil"
	"golang.org/x/tools/gopls/internal/bug"
	"golang.org/x/tools/gopls/internal/lsp/protocol"
	"golang.org/x/tools/gopls/internal/span"
	"golang.org/x/tools/internal/refactor/inline"
)

// inlineAllCalls inlines all calls to the original function declaration
// described by callee, returning the resulting modified file content.
//
// inlining everything is currently an expensive operation: it involves re-type
// checking every package that contains a potential call, as reported by
// References. In cases where there are multiple calls per file, inlineAllCalls
// must type check repeatedly for each additional call.
//
// The provided post processing function is applied to the resulting source
// after each transformation. This is necessary because we are using this
// function to inline synthetic wrappers for the purpose of signature
// rewriting. The delegated function has a fake name that doesn't exist in the
// snapshot, and so we can't re-type check until we replace this fake name.
//
// TODO(rfindley): this only works because removing a parameter is a very
// narrow operation. A better solution would be to allow for ad-hoc snapshots
// that expose the full machinery of real snapshots: minimal invalidation,
// batched type checking, etc. Then we could actually rewrite the declaring
// package in this snapshot (and so 'post' would not be necessary), and could
// robustly re-type check for the purpose of iterative inlining, even if the
// inlined code pulls in new imports that weren't present in export data.
//
// The code below notes where are assumptions are made that only hold true in
// the case of parameter removal (annotated with 'Assumption:')
func inlineAllCalls(ctx context.Context, logf func(string, ...any), snapshot Snapshot, pkg Package, pgf *ParsedGoFile, origDecl *ast.FuncDecl, callee *inline.Callee, post func([]byte) []byte) (map[span.URI][]byte, error) {
	// Collect references.
	var refs []protocol.Location
	{
		funcPos, err := pgf.Mapper.PosPosition(pgf.Tok, origDecl.Name.NamePos)
		if err != nil {
			return nil, err
		}
		fh, err := snapshot.ReadFile(ctx, pgf.URI)
		if err != nil {
			return nil, err
		}
		refs, err = References(ctx, snapshot, fh, funcPos, false)
		if err != nil {
			return nil, fmt.Errorf("finding references to rewrite: %v", err)
		}
	}

	// Type-check the narrowest package containing each reference.
	// TODO(rfindley): we should expose forEachPackage in order to operate in
	// parallel and to reduce peak memory for this operation.
	var (
		pkgForRef = make(map[protocol.Location]PackageID)
		pkgs      = make(map[PackageID]Package)
	)
	{
		needPkgs := make(map[PackageID]struct{})
		for _, ref := range refs {
			md, err := NarrowestMetadataForFile(ctx, snapshot, ref.URI.SpanURI())
			if err != nil {
				return nil, fmt.Errorf("finding ref metadata: %v", err)
			}
			pkgForRef[ref] = md.ID
			needPkgs[md.ID] = struct{}{}
		}
		var pkgIDs []PackageID
		for id := range needPkgs { // TODO: use maps.Keys once it is available to us
			pkgIDs = append(pkgIDs, id)
		}

		refPkgs, err := snapshot.TypeCheck(ctx, pkgIDs...)
		if err != nil {
			return nil, fmt.Errorf("type checking reference packages: %v", err)
		}

		for _, p := range refPkgs {
			pkgs[p.Metadata().ID] = p
		}
	}

	// Organize calls by top file declaration. Calls within a single file may
	// affect each other, as the inlining edit may affect the surrounding scope
	// or imports Therefore, when inlining subsequent calls in the same
	// declaration, we must re-type check.

	type fileCalls struct {
		pkg   Package
		pgf   *ParsedGoFile
		calls []*ast.CallExpr
	}

	refsByFile := make(map[span.URI]*fileCalls)
	for _, ref := range refs {
		refpkg := pkgs[pkgForRef[ref]]
		pgf, err := refpkg.File(ref.URI.SpanURI())
		if err != nil {
			return nil, bug.Errorf("finding %s in %s: %v", ref.URI, refpkg.Metadata().ID, err)
		}
		start, end, err := pgf.RangePos(ref.Range)
		if err != nil {
			return nil, bug.Errorf("RangePos(ref): %v", err)
		}

		// Look for the surrounding call expression.
		var (
			name *ast.Ident
			call *ast.CallExpr
		)
		path, _ := astutil.PathEnclosingInterval(pgf.File, start, end)
		name, _ = path[0].(*ast.Ident)
		if _, ok := path[1].(*ast.SelectorExpr); ok {
			call, _ = path[2].(*ast.CallExpr)
		} else {
			call, _ = path[1].(*ast.CallExpr)
		}
		if name == nil || call == nil {
			// TODO(rfindley): handle this case with eta-abstraction:
			// a reference to the target function f in a non-call position
			//    use(f)
			// is replaced by
			//    use(func(...) { f(...) })
			return nil, fmt.Errorf("cannot inline: found non-call function reference %v", ref)
		}
		// Sanity check.
		if obj := refpkg.GetTypesInfo().ObjectOf(name); obj == nil ||
			obj.Name() != origDecl.Name.Name ||
			obj.Pkg() == nil ||
			obj.Pkg().Path() != string(pkg.Metadata().PkgPath) {
			return nil, bug.Errorf("cannot inline: corrupted reference %v", ref)
		}

		callInfo, ok := refsByFile[ref.URI.SpanURI()]
		if !ok {
			callInfo = &fileCalls{
				pkg: refpkg,
				pgf: pgf,
			}
			refsByFile[ref.URI.SpanURI()] = callInfo
		}
		callInfo.calls = append(callInfo.calls, call)
	}

	// Inline each call within the same decl in sequence, re-typechecking after
	// each one. If there is only a single call within the decl, we can avoid
	// additional type checking.
	//
	// Assumption: inlining does not affect the package scope, so we can operate
	// on separate files independently.
	result := make(map[span.URI][]byte)
	for uri, callInfo := range refsByFile {
		var (
			calls   = callInfo.calls
			fset    = callInfo.pkg.FileSet()
			tpkg    = callInfo.pkg.GetTypes()
			tinfo   = callInfo.pkg.GetTypesInfo()
			file    = callInfo.pgf.File
			content = callInfo.pgf.Src
		)

		// Check for overlapping calls (such as Foo(Foo())). We can't handle these
		// because inlining may change the source order of the inner call with
		// respect to the inlined outer call, and so the heuristic we use to find
		// the next call (counting from top-to-bottom) does not work.
		for i := range calls {
			if i > 0 && calls[i-1].End() > calls[i].Pos() {
				return nil, fmt.Errorf("%s: can't inline overlapping call %s", uri, types.ExprString(calls[i-1]))
			}
		}

		currentCall := 0
		for currentCall < len(calls) {
			caller := &inline.Caller{
				Fset:    fset,
				Types:   tpkg,
				Info:    tinfo,
				File:    file,
				Call:    calls[currentCall],
				Content: content,
			}
			var err error
			content, err = inline.Inline(logf, caller, callee)
			if err != nil {
				return nil, fmt.Errorf("inlining failed: %v", err)
			}
			if post != nil {
				content = post(content)
			}
			if len(calls) <= 1 {
				// No need to re-type check, as we've inlined all calls.
				break
			}

			// TODO(rfindley): develop a theory of "trivial" inlining, which are
			// inlinings that don't require re-type checking.
			//
			// In principle, if the inlining only involves replacing one call with
			// another, the scope of the caller is unchanged and there is no need to
			// type check again before inlining subsequent calls (edits should not
			// overlap, and should not affect each other semantically). However, it
			// feels sufficiently complicated that, to be safe, this optimization is
			// deferred until later.

			file, err = parser.ParseFile(fset, uri.Filename(), content, parser.ParseComments|parser.SkipObjectResolution)
			if err != nil {
				return nil, bug.Errorf("inlined file failed to parse: %v", err)
			}

			// After inlining one call with a removed parameter, the package will
			// fail to type check due to "not enough arguments". Therefore, we must
			// allow type errors here.
			//
			// Assumption: the resulting type errors do not affect the correctness of
			// subsequent inlining, because invalid arguments to a call do not affect
			// anything in the surrounding scope.
			//
			// TODO(rfindley): improve this.
			tpkg, tinfo, err = reTypeCheck(logf, callInfo.pkg, map[span.URI]*ast.File{uri: file}, true)
			if err != nil {
				return nil, bug.Errorf("type checking after inlining failed: %v", err)
			}

			// Collect calls to the target function in the modified declaration.
			var calls2 []*ast.CallExpr
			ast.Inspect(file, func(n ast.Node) bool {
				if call, ok := n.(*ast.CallExpr); ok {
					fn := typeutil.StaticCallee(tinfo, call)
					if fn != nil && fn.Pkg().Path() == string(pkg.Metadata().PkgPath) && fn.Name() == origDecl.Name.Name {
						calls2 = append(calls2, call)
					}
				}
				return true
			})

			// If the number of calls has increased, this process will never cease.
			// If the number of calls has decreased, assume that inlining removed a
			// call.
			// If the number of calls didn't change, assume that inlining replaced
			// a call, and move on to the next.
			//
			// Assumption: we're inlining a call that has at most one recursive
			// reference (which holds for signature rewrites).
			//
			// TODO(rfindley): this isn't good enough. We should be able to support
			// inlining all existing calls even if they increase calls. How do we
			// correlate the before and after syntax?
			switch {
			case len(calls2) > len(calls):
				return nil, fmt.Errorf("inlining increased calls %d->%d, possible recursive call? content:\n%s", len(calls), len(calls2), content)
			case len(calls2) < len(calls):
				calls = calls2
			case len(calls2) == len(calls):
				calls = calls2
				currentCall++
			}
		}

		result[callInfo.pgf.URI] = content
	}
	return result, nil
}
