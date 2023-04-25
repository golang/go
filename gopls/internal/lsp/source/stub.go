// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package source

import (
	"bytes"
	"context"
	"fmt"
	"go/format"
	"go/parser"
	"go/token"
	"go/types"
	"io"
	"path"
	"strings"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/ast/astutil"
	"golang.org/x/tools/gopls/internal/bug"
	"golang.org/x/tools/gopls/internal/lsp/analysis/stubmethods"
	"golang.org/x/tools/gopls/internal/lsp/protocol"
	"golang.org/x/tools/gopls/internal/lsp/safetoken"
	"golang.org/x/tools/internal/tokeninternal"
	"golang.org/x/tools/internal/typeparams"
)

// stubSuggestedFixFunc returns a suggested fix to declare the missing
// methods of the concrete type that is assigned to an interface type
// at the cursor position.
func stubSuggestedFixFunc(ctx context.Context, snapshot Snapshot, fh FileHandle, rng protocol.Range) (*token.FileSet, *analysis.SuggestedFix, error) {
	pkg, pgf, err := NarrowestPackageForFile(ctx, snapshot, fh.URI())
	if err != nil {
		return nil, nil, fmt.Errorf("GetTypedFile: %w", err)
	}
	start, end, err := pgf.RangePos(rng)
	if err != nil {
		return nil, nil, err
	}
	nodes, _ := astutil.PathEnclosingInterval(pgf.File, start, end)
	si := stubmethods.GetStubInfo(pkg.FileSet(), pkg.GetTypesInfo(), nodes, start)
	if si == nil {
		return nil, nil, fmt.Errorf("nil interface request")
	}
	return stub(ctx, snapshot, si)
}

// stub returns a suggested fix to declare the missing methods of si.Concrete.
func stub(ctx context.Context, snapshot Snapshot, si *stubmethods.StubInfo) (*token.FileSet, *analysis.SuggestedFix, error) {
	// A function-local type cannot be stubbed
	// since there's nowhere to put the methods.
	conc := si.Concrete.Obj()
	if conc.Parent() != conc.Pkg().Scope() {
		return nil, nil, fmt.Errorf("local type %q cannot be stubbed", conc.Name())
	}

	// Parse the file declaring the concrete type.
	declPGF, _, err := parseFull(ctx, snapshot, si.Fset, conc.Pos())
	if err != nil {
		return nil, nil, fmt.Errorf("failed to parse file %q declaring implementation type: %w", declPGF.URI, err)
	}
	if declPGF.Fixed() {
		return nil, nil, fmt.Errorf("file contains parse errors: %s", declPGF.URI)
	}

	// Build import environment for the declaring file.
	importEnv := make(map[ImportPath]string) // value is local name
	for _, imp := range declPGF.File.Imports {
		importPath := UnquoteImportPath(imp)
		var name string
		if imp.Name != nil {
			name = imp.Name.Name
			if name == "_" {
				continue
			} else if name == "." {
				name = "" // see types.Qualifier
			}
		} else {
			// TODO(adonovan): may omit a vendor/ prefix; consult the Metadata.
			name = path.Base(string(importPath))
		}
		importEnv[importPath] = name // latest alias wins
	}

	// Find subset of interface methods that the concrete type lacks.
	var missing []*types.Func
	ifaceType := si.Interface.Type().Underlying().(*types.Interface)
	for i := 0; i < ifaceType.NumMethods(); i++ {
		imethod := ifaceType.Method(i)
		cmethod, _, _ := types.LookupFieldOrMethod(si.Concrete, si.Pointer, imethod.Pkg(), imethod.Name())
		if cmethod == nil {
			missing = append(missing, imethod)
			continue
		}

		if _, ok := cmethod.(*types.Var); ok {
			// len(LookupFieldOrMethod.index) = 1 => conflict, >1 => shadow.
			return nil, nil, fmt.Errorf("adding method %s.%s would conflict with (or shadow) existing field",
				conc.Name(), imethod.Name())
		}

		if !types.Identical(cmethod.Type(), imethod.Type()) {
			return nil, nil, fmt.Errorf("method %s.%s already exists but has the wrong type: got %s, want %s",
				conc.Name(), imethod.Name(), cmethod.Type(), imethod.Type())
		}
	}
	if len(missing) == 0 {
		return nil, nil, fmt.Errorf("no missing methods found")
	}

	// Create a package name qualifier that uses the
	// locally appropriate imported package name.
	// It records any needed new imports.
	// TODO(adonovan): factor with source.FormatVarType, stubmethods.RelativeToFiles?
	//
	// Prior to CL 469155 this logic preserved any renaming
	// imports from the file that declares the interface
	// method--ostensibly the preferred name for imports of
	// frequently renamed packages such as protobufs.
	// Now we use the package's declared name. If this turns out
	// to be a mistake, then use parseHeader(si.iface.Pos()).
	//
	type newImport struct{ name, importPath string }
	var newImports []newImport // for AddNamedImport
	qual := func(pkg *types.Package) string {
		// TODO(adonovan): don't ignore vendor prefix.
		importPath := ImportPath(pkg.Path())
		name, ok := importEnv[importPath]
		if !ok {
			// Insert new import using package's declared name.
			//
			// TODO(adonovan): resolve conflict between declared
			// name and existing file-level (declPGF.File.Imports)
			// or package-level (si.Concrete.Pkg.Scope) decls by
			// generating a fresh name.
			name = pkg.Name()
			importEnv[importPath] = name
			new := newImport{importPath: string(importPath)}
			// For clarity, use a renaming import whenever the
			// local name does not match the path's last segment.
			if name != path.Base(new.importPath) {
				new.name = name
			}
			newImports = append(newImports, new)
		}
		return name
	}

	// Format interface name (used only in a comment).
	iface := si.Interface.Name()
	if ipkg := si.Interface.Pkg(); ipkg != nil && ipkg != conc.Pkg() {
		iface = ipkg.Name() + "." + iface
	}

	// Pointer receiver?
	var star string
	if si.Pointer {
		star = "*"
	}

	// Format the new methods.
	var newMethods bytes.Buffer
	for _, method := range missing {
		fmt.Fprintf(&newMethods, `// %s implements %s.
func (%s%s%s) %s%s {
	panic("unimplemented")
}
`,
			method.Name(),
			iface,
			star,
			si.Concrete.Obj().Name(),
			FormatTypeParams(typeparams.ForNamed(si.Concrete)),
			method.Name(),
			strings.TrimPrefix(types.TypeString(method.Type(), qual), "func"))
	}

	// Compute insertion point for new methods:
	// after the top-level declaration enclosing the (package-level) type.
	insertOffset, err := safetoken.Offset(declPGF.Tok, declPGF.File.End())
	if err != nil {
		return nil, nil, bug.Errorf("internal error: end position outside file bounds: %v", err)
	}
	concOffset, err := safetoken.Offset(si.Fset.File(conc.Pos()), conc.Pos())
	if err != nil {
		return nil, nil, bug.Errorf("internal error: finding type decl offset: %v", err)
	}
	for _, decl := range declPGF.File.Decls {
		declEndOffset, err := safetoken.Offset(declPGF.Tok, decl.End())
		if err != nil {
			return nil, nil, bug.Errorf("internal error: finding decl offset: %v", err)
		}
		if declEndOffset > concOffset {
			insertOffset = declEndOffset
			break
		}
	}

	// Splice the new methods into the file content.
	var buf bytes.Buffer
	input := declPGF.Mapper.Content // unfixed content of file
	buf.Write(input[:insertOffset])
	buf.WriteByte('\n')
	io.Copy(&buf, &newMethods)
	buf.Write(input[insertOffset:])

	// Re-parse the file.
	fset := token.NewFileSet()
	newF, err := parser.ParseFile(fset, declPGF.File.Name.Name, buf.Bytes(), parser.ParseComments)
	if err != nil {
		return nil, nil, fmt.Errorf("could not reparse file: %w", err)
	}

	// Splice the new imports into the syntax tree.
	for _, imp := range newImports {
		astutil.AddNamedImport(fset, newF, imp.name, imp.importPath)
	}

	// Pretty-print.
	var output strings.Builder
	if err := format.Node(&output, fset, newF); err != nil {
		return nil, nil, fmt.Errorf("format.Node: %w", err)
	}

	// Report the diff.
	diffs := snapshot.View().Options().ComputeEdits(string(input), output.String())
	var edits []analysis.TextEdit
	for _, edit := range diffs {
		edits = append(edits, analysis.TextEdit{
			Pos:     declPGF.Tok.Pos(edit.Start),
			End:     declPGF.Tok.Pos(edit.End),
			NewText: []byte(edit.New),
		})
	}
	return tokeninternal.FileSetFor(declPGF.Tok), // edits use declPGF.Tok
		&analysis.SuggestedFix{TextEdits: edits},
		nil
}
