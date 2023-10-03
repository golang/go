// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package source

import (
	"bytes"
	"context"
	"fmt"
	"go/ast"
	"go/format"
	"go/parser"
	"go/token"
	"go/types"
	"regexp"

	"golang.org/x/tools/go/ast/astutil"
	"golang.org/x/tools/gopls/internal/bug"
	"golang.org/x/tools/gopls/internal/lsp/protocol"
	"golang.org/x/tools/gopls/internal/lsp/safetoken"
	"golang.org/x/tools/gopls/internal/span"
	"golang.org/x/tools/imports"
	internalastutil "golang.org/x/tools/internal/astutil"
	"golang.org/x/tools/internal/diff"
	"golang.org/x/tools/internal/refactor/inline"
	"golang.org/x/tools/internal/tokeninternal"
	"golang.org/x/tools/internal/typesinternal"
)

// RemoveUnusedParameter computes a refactoring to remove the parameter
// indicated by the given range, which must be contained within an unused
// parameter name or field.
//
// This operation is a work in progress. Remaining TODO:
//   - Handle function assignment correctly.
//   - Improve the extra newlines in output.
//   - Stream type checking via ForEachPackage.
//   - Avoid unnecessary additional type checking.
func RemoveUnusedParameter(ctx context.Context, fh FileHandle, rng protocol.Range, snapshot Snapshot) ([]protocol.DocumentChanges, error) {
	pkg, pgf, err := NarrowestPackageForFile(ctx, snapshot, fh.URI())
	if err != nil {
		return nil, err
	}
	if perrors, terrors := pkg.GetParseErrors(), pkg.GetTypeErrors(); len(perrors) > 0 || len(terrors) > 0 {
		var sample string
		if len(perrors) > 0 {
			sample = perrors[0].Error()
		} else {
			sample = terrors[0].Error()
		}
		return nil, fmt.Errorf("can't change signatures for packages with parse or type errors: (e.g. %s)", sample)
	}

	info := FindParam(pgf, rng)
	if info.Decl == nil {
		return nil, fmt.Errorf("failed to find declaration")
	}
	if info.Decl.Recv != nil {
		return nil, fmt.Errorf("can't change signature of methods (yet)")
	}
	if info.Field == nil {
		return nil, fmt.Errorf("failed to find field")
	}

	// Create the new declaration, which is a copy of the original decl with the
	// unnecessary parameter removed.
	newDecl := internalastutil.CloneNode(info.Decl)
	if info.Name != nil {
		names := remove(newDecl.Type.Params.List[info.FieldIndex].Names, info.NameIndex)
		newDecl.Type.Params.List[info.FieldIndex].Names = names
	}
	if len(newDecl.Type.Params.List[info.FieldIndex].Names) == 0 {
		// Unnamed, or final name was removed: in either case, remove the field.
		newDecl.Type.Params.List = remove(newDecl.Type.Params.List, info.FieldIndex)
	}

	// Compute inputs into building a wrapper function around the modified
	// signature.
	var (
		params   = internalastutil.CloneNode(info.Decl.Type.Params) // "_" names will be modified
		args     []ast.Expr                                         // arguments to delegate
		variadic = false                                            // whether the signature is variadic
	)
	{
		allNames := make(map[string]bool) // for renaming blanks
		for _, fld := range params.List {
			for _, n := range fld.Names {
				if n.Name != "_" {
					allNames[n.Name] = true
				}
			}
		}
		blanks := 0
		for i, fld := range params.List {
			for j, n := range fld.Names {
				if i == info.FieldIndex && j == info.NameIndex {
					continue
				}
				if n.Name == "_" {
					// Create names for blank (_) parameters so the delegating wrapper
					// can refer to them.
					for {
						newName := fmt.Sprintf("blank%d", blanks)
						blanks++
						if !allNames[newName] {
							n.Name = newName
							break
						}
					}
				}
				args = append(args, &ast.Ident{Name: n.Name})
				if i == len(params.List)-1 {
					_, variadic = fld.Type.(*ast.Ellipsis)
				}
			}
		}
	}

	// Rewrite all referring calls.
	newContent, err := rewriteCalls(ctx, signatureRewrite{
		snapshot: snapshot,
		pkg:      pkg,
		pgf:      pgf,
		origDecl: info.Decl,
		newDecl:  newDecl,
		params:   params,
		callArgs: args,
		variadic: variadic,
	})
	if err != nil {
		return nil, err
	}
	// Finally, rewrite the original declaration. We do this after inlining all
	// calls, as there may be calls in the same file as the declaration. But none
	// of the inlining should have changed the location of the original
	// declaration.
	{
		idx := findDecl(pgf.File, info.Decl)
		if idx < 0 {
			return nil, bug.Errorf("didn't find original decl")
		}

		src, ok := newContent[pgf.URI]
		if !ok {
			src = pgf.Src
		}
		fset := tokeninternal.FileSetFor(pgf.Tok)
		src, err = rewriteSignature(fset, idx, src, newDecl)
		newContent[pgf.URI] = src
	}

	// Translate the resulting state into document changes.
	var changes []protocol.DocumentChanges
	for uri, after := range newContent {
		fh, err := snapshot.ReadFile(ctx, uri)
		if err != nil {
			return nil, err
		}
		before, err := fh.Content()
		if err != nil {
			return nil, err
		}
		edits := diff.Bytes(before, after)
		mapper := protocol.NewMapper(uri, before)
		pedits, err := ToProtocolEdits(mapper, edits)
		if err != nil {
			return nil, fmt.Errorf("computing edits for %s: %v", uri, err)
		}
		changes = append(changes, protocol.DocumentChanges{
			TextDocumentEdit: &protocol.TextDocumentEdit{
				TextDocument: protocol.OptionalVersionedTextDocumentIdentifier{
					Version:                fh.Version(),
					TextDocumentIdentifier: protocol.TextDocumentIdentifier{URI: protocol.URIFromSpanURI(uri)},
				},
				Edits: pedits,
			},
		})
	}
	return changes, nil
}

// rewriteSignature rewrites the signature of the declIdx'th declaration in src
// to use the signature of newDecl (described by fset).
//
// TODO(rfindley): I think this operation could be generalized, for example by
// using a concept of a 'nodepath' to correlate nodes between two related
// files.
//
// Note that with its current application, rewriteSignature is expected to
// succeed. Separate bug.Errorf calls are used below (rather than one call at
// the callsite) in order to have greater precision.
func rewriteSignature(fset *token.FileSet, declIdx int, src0 []byte, newDecl *ast.FuncDecl) ([]byte, error) {
	// Parse the new file0 content, to locate the original params.
	file0, err := parser.ParseFile(fset, "", src0, parser.ParseComments|parser.SkipObjectResolution)
	if err != nil {
		return nil, bug.Errorf("re-parsing declaring file failed: %v", err)
	}
	decl0, _ := file0.Decls[declIdx].(*ast.FuncDecl)
	// Inlining shouldn't have changed the location of any declarations, but do
	// a sanity check.
	if decl0 == nil || decl0.Name.Name != newDecl.Name.Name {
		return nil, bug.Errorf("inlining affected declaration order: found %v, not func %s", decl0, newDecl.Name.Name)
	}
	opening0, closing0, err := safetoken.Offsets(fset.File(decl0.Pos()), decl0.Type.Params.Opening, decl0.Type.Params.Closing)
	if err != nil {
		return nil, bug.Errorf("can't find params: %v", err)
	}

	// Format the modified signature and apply a textual replacement. This
	// minimizes comment disruption.
	formattedType := FormatNode(fset, newDecl.Type)
	expr, err := parser.ParseExprFrom(fset, "", []byte(formattedType), 0)
	if err != nil {
		return nil, bug.Errorf("parsing modified signature: %v", err)
	}
	newType := expr.(*ast.FuncType)
	opening1, closing1, err := safetoken.Offsets(fset.File(newType.Pos()), newType.Params.Opening, newType.Params.Closing)
	if err != nil {
		return nil, bug.Errorf("param offsets: %v", err)
	}
	newParams := formattedType[opening1 : closing1+1]

	// Splice.
	var buf bytes.Buffer
	buf.Write(src0[:opening0])
	buf.WriteString(newParams)
	buf.Write(src0[closing0+1:])
	newSrc := buf.Bytes()
	if len(file0.Imports) > 0 {
		formatted, err := imports.Process("output", newSrc, nil)
		if err != nil {
			return nil, bug.Errorf("imports.Process failed: %v", err)
		}
		newSrc = formatted
	}
	return newSrc, nil
}

// ParamInfo records information about a param identified by a position.
type ParamInfo struct {
	Decl       *ast.FuncDecl // enclosing func decl, or nil
	FieldIndex int           // index of Field in Decl.Type.Params, or -1
	Field      *ast.Field    // enclosing field of Decl, or nil
	NameIndex  int           // index of Name in Field.Names, or nil
	Name       *ast.Ident    // indicated name (either enclosing, or Field.Names[0] if len(Field.Names) == 1)
}

// FindParam finds the parameter information spanned by the given range.
func FindParam(pgf *ParsedGoFile, rng protocol.Range) ParamInfo {
	info := ParamInfo{FieldIndex: -1, NameIndex: -1}
	start, end, err := pgf.RangePos(rng)
	if err != nil {
		bug.Reportf("(file=%v).RangePos(%v) failed: %v", pgf.URI, rng, err)
		return info
	}

	path, _ := astutil.PathEnclosingInterval(pgf.File, start, end)
	var (
		id    *ast.Ident
		field *ast.Field
		decl  *ast.FuncDecl
	)
	// Find the outermost enclosing node of each kind, whether or not they match
	// the semantics described in the docstring.
	for _, n := range path {
		switch n := n.(type) {
		case *ast.Ident:
			id = n
		case *ast.Field:
			field = n
		case *ast.FuncDecl:
			decl = n
		}
	}
	// Check the conditions described in the docstring.
	if decl == nil {
		return info
	}
	info.Decl = decl
	for fi, f := range decl.Type.Params.List {
		if f == field {
			info.FieldIndex = fi
			info.Field = f
			for ni, n := range f.Names {
				if n == id {
					info.NameIndex = ni
					info.Name = n
					break
				}
			}
			if info.Name == nil && len(info.Field.Names) == 1 {
				info.NameIndex = 0
				info.Name = info.Field.Names[0]
			}
			break
		}
	}
	return info
}

// signatureRewrite defines a rewritten function signature.
//
// See rewriteCalls for more details.
type signatureRewrite struct {
	snapshot          Snapshot
	pkg               Package
	pgf               *ParsedGoFile
	origDecl, newDecl *ast.FuncDecl
	params            *ast.FieldList
	callArgs          []ast.Expr
	variadic          bool
}

// rewriteCalls returns the document changes required to rewrite the
// signature of origDecl to that of newDecl.
//
// This is a rather complicated factoring of the rewrite operation, but is able
// to describe arbitrary rewrites. Specifically, rewriteCalls creates a
// synthetic copy of pkg, where the original function declaration is changed to
// be a trivial wrapper around the new declaration. params and callArgs are
// used to perform this delegation: params must have the same type as origDecl,
// but may have renamed parameters (such as is required for delegating blank
// parameters). callArgs are the arguments of the delegated call (i.e. using
// params).
//
// For example, consider removing the unused 'b' parameter below, rewriting
//
//	func Foo(a, b, c, _ int) int {
//	  return a+c
//	}
//
// To
//
//	func Foo(a, c, _ int) int {
//	  return a+c
//	}
//
// In this case, rewriteCalls is parameterized as follows:
//   - origDecl is the original declaration
//   - newDecl is the new declaration, which is a copy of origDecl less the 'b'
//     parameter.
//   - params is a new parameter list (a, b, c, blank0 int) to be used for the
//     new wrapper.
//   - callArgs is the argument list (a, c, blank0), to be used to call the new
//     delegate.
//
// rewriting is expressed this way so that rewriteCalls can own the details
// of *how* this rewriting is performed. For example, as of writing it names
// the synthetic delegate G_o_p_l_s_foo, but the caller need not know this.
//
// By passing an entirely new declaration, rewriteCalls may be used for
// signature refactorings that may affect the function body, such as removing
// or adding return values.
func rewriteCalls(ctx context.Context, rw signatureRewrite) (map[span.URI][]byte, error) {
	// tag is a unique prefix that is added to the delegated declaration.
	//
	// It must have a ~0% probability of causing collisions with existing names.
	const tag = "G_o_p_l_s_"

	var (
		modifiedSrc  []byte
		modifiedFile *ast.File
		modifiedDecl *ast.FuncDecl
	)
	{
		delegate := internalastutil.CloneNode(rw.newDecl) // clone before modifying
		delegate.Name.Name = tag + delegate.Name.Name
		if obj := rw.pkg.GetTypes().Scope().Lookup(delegate.Name.Name); obj != nil {
			return nil, fmt.Errorf("synthetic name %q conflicts with an existing declaration", delegate.Name.Name)
		}

		wrapper := internalastutil.CloneNode(rw.origDecl)
		wrapper.Type.Params = rw.params
		call := &ast.CallExpr{
			Fun:  &ast.Ident{Name: delegate.Name.Name},
			Args: rw.callArgs,
		}
		if rw.variadic {
			call.Ellipsis = 1 // must not be token.NoPos
		}

		var stmt ast.Stmt
		if delegate.Type.Results.NumFields() > 0 {
			stmt = &ast.ReturnStmt{
				Results: []ast.Expr{call},
			}
		} else {
			stmt = &ast.ExprStmt{
				X: call,
			}
		}
		wrapper.Body = &ast.BlockStmt{
			List: []ast.Stmt{stmt},
		}

		fset := tokeninternal.FileSetFor(rw.pgf.Tok)
		var err error
		modifiedSrc, err = replaceFileDecl(rw.pgf, rw.origDecl, delegate)
		if err != nil {
			return nil, err
		}
		// TODO(rfindley): we can probably get away with one fewer parse operations
		// by returning the modified AST from replaceDecl. Investigate if that is
		// accurate.
		modifiedSrc = append(modifiedSrc, []byte("\n\n"+FormatNode(fset, wrapper))...)
		modifiedFile, err = parser.ParseFile(rw.pkg.FileSet(), rw.pgf.URI.Filename(), modifiedSrc, parser.ParseComments|parser.SkipObjectResolution)
		if err != nil {
			return nil, err
		}
		modifiedDecl = modifiedFile.Decls[len(modifiedFile.Decls)-1].(*ast.FuncDecl)
	}

	// Type check pkg again with the modified file, to compute the synthetic
	// callee.
	logf := logger(ctx, "change signature", rw.snapshot.Options().VerboseOutput)
	pkg2, info, err := reTypeCheck(logf, rw.pkg, map[span.URI]*ast.File{rw.pgf.URI: modifiedFile}, false)
	if err != nil {
		return nil, err
	}
	calleeInfo, err := inline.AnalyzeCallee(logf, rw.pkg.FileSet(), pkg2, info, modifiedDecl, modifiedSrc)
	if err != nil {
		return nil, fmt.Errorf("analyzing callee: %v", err)
	}

	post := func(got []byte) []byte { return bytes.ReplaceAll(got, []byte(tag), nil) }
	return inlineAllCalls(ctx, logf, rw.snapshot, rw.pkg, rw.pgf, rw.origDecl, calleeInfo, post)
}

// reTypeCheck re-type checks orig with new file contents defined by fileMask.
//
// It expects that any newly added imports are already present in the
// transitive imports of orig.
//
// If expectErrors is true, reTypeCheck allows errors in the new package.
// TODO(rfindley): perhaps this should be a filter to specify which errors are
// acceptable.
func reTypeCheck(logf func(string, ...any), orig Package, fileMask map[span.URI]*ast.File, expectErrors bool) (*types.Package, *types.Info, error) {
	pkg := types.NewPackage(string(orig.Metadata().PkgPath), string(orig.Metadata().Name))
	info := &types.Info{
		Types:      make(map[ast.Expr]types.TypeAndValue),
		Defs:       make(map[*ast.Ident]types.Object),
		Uses:       make(map[*ast.Ident]types.Object),
		Implicits:  make(map[ast.Node]types.Object),
		Selections: make(map[*ast.SelectorExpr]*types.Selection),
		Scopes:     make(map[ast.Node]*types.Scope),
		Instances:  make(map[*ast.Ident]types.Instance),
	}
	{
		var files []*ast.File
		for _, pgf := range orig.CompiledGoFiles() {
			if mask, ok := fileMask[pgf.URI]; ok {
				files = append(files, mask)
			} else {
				files = append(files, pgf.File)
			}
		}

		// Implement a BFS for imports in the transitive package graph.
		//
		// Note that this only works if any newly added imports are expected to be
		// present among transitive imports. In general we cannot assume this to
		// be the case, but in the special case of removing a parameter it works
		// because any parameter types must be present in export data.
		var importer func(importPath string) (*types.Package, error)
		{
			var (
				importsByPath = make(map[string]*types.Package)   // cached imports
				toSearch      = []*types.Package{orig.GetTypes()} // packages to search
				searched      = make(map[string]bool)             // path -> (false, if present in toSearch; true, if already searched)
			)
			importer = func(path string) (*types.Package, error) {
				if p, ok := importsByPath[path]; ok {
					return p, nil
				}
				for len(toSearch) > 0 {
					pkg := toSearch[0]
					toSearch = toSearch[1:]
					searched[pkg.Path()] = true
					for _, p := range pkg.Imports() {
						// TODO(rfindley): this is incorrect: p.Path() is a package path,
						// whereas path is an import path. We can fix this by reporting any
						// newly added imports from inlining, or by using the ImporterFrom
						// interface and package metadata.
						//
						// TODO(rfindley): can't the inliner also be wrong here? It's
						// possible that an import path means different things depending on
						// the location.
						importsByPath[p.Path()] = p
						if _, ok := searched[p.Path()]; !ok {
							searched[p.Path()] = false
							toSearch = append(toSearch, p)
						}
					}
					if p, ok := importsByPath[path]; ok {
						return p, nil
					}
				}
				return nil, fmt.Errorf("missing import")
			}
		}
		cfg := &types.Config{
			Sizes:    orig.Metadata().TypesSizes,
			Importer: ImporterFunc(importer),
		}

		// Copied from cache/check.go.
		// TODO(rfindley): factor this out and fix goVersionRx.
		// Set Go dialect.
		if module := orig.Metadata().Module; module != nil && module.GoVersion != "" {
			goVersion := "go" + module.GoVersion
			// types.NewChecker panics if GoVersion is invalid.
			// An unparsable mod file should probably stop us
			// before we get here, but double check just in case.
			if goVersionRx.MatchString(goVersion) {
				typesinternal.SetGoVersion(cfg, goVersion)
			}
		}
		if expectErrors {
			cfg.Error = func(err error) {
				logf("re-type checking: expected error: %v", err)
			}
		}
		typesinternal.SetUsesCgo(cfg)
		checker := types.NewChecker(cfg, orig.FileSet(), pkg, info)
		if err := checker.Files(files); err != nil && !expectErrors {
			return nil, nil, fmt.Errorf("type checking rewritten package: %v", err)
		}
	}
	return pkg, info, nil
}

// TODO(golang/go#63472): this looks wrong with the new Go version syntax.
var goVersionRx = regexp.MustCompile(`^go([1-9][0-9]*)\.(0|[1-9][0-9]*)$`)

func remove[T any](s []T, i int) []T {
	return append(s[:i], s[i+1:]...)
}

// replaceFileDecl replaces old with new in the file described by pgf.
//
// TODO(rfindley): generalize, and combine with rewriteSignature.
func replaceFileDecl(pgf *ParsedGoFile, old, new ast.Decl) ([]byte, error) {
	i := findDecl(pgf.File, old)
	if i == -1 {
		return nil, bug.Errorf("didn't find old declaration")
	}
	start, end, err := safetoken.Offsets(pgf.Tok, old.Pos(), old.End())
	if err != nil {
		return nil, err
	}
	var out bytes.Buffer
	out.Write(pgf.Src[:start])
	fset := tokeninternal.FileSetFor(pgf.Tok)
	if err := format.Node(&out, fset, new); err != nil {
		return nil, bug.Errorf("formatting new node: %v", err)
	}
	out.Write(pgf.Src[end:])
	return out.Bytes(), nil
}

// findDecl finds the index of decl in file.Decls.
//
// TODO: use slices.Index when it is available.
func findDecl(file *ast.File, decl ast.Decl) int {
	for i, d := range file.Decls {
		if d == decl {
			return i
		}
	}
	return -1
}
