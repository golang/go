// Copyright 2022 The Go Authors. All rights reserved.
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
	"sort"
	"strings"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/ast/astutil"
	"golang.org/x/tools/gopls/internal/lsp/analysis/stubmethods"
	"golang.org/x/tools/gopls/internal/lsp/protocol"
	"golang.org/x/tools/gopls/internal/lsp/safetoken"
	"golang.org/x/tools/gopls/internal/span"
	"golang.org/x/tools/internal/typeparams"
)

func stubSuggestedFixFunc(ctx context.Context, snapshot Snapshot, fh FileHandle, rng protocol.Range) (*token.FileSet, *analysis.SuggestedFix, error) {
	pkg, pgf, err := PackageForFile(ctx, snapshot, fh.URI(), TypecheckFull, NarrowestPackage)
	if err != nil {
		return nil, nil, fmt.Errorf("GetTypedFile: %w", err)
	}
	nodes, pos, err := getStubNodes(pgf, rng)
	if err != nil {
		return nil, nil, fmt.Errorf("getNodes: %w", err)
	}
	si := stubmethods.GetStubInfo(pkg.GetTypesInfo(), nodes, pos)
	if si == nil {
		return nil, nil, fmt.Errorf("nil interface request")
	}

	// A function-local type cannot be stubbed
	// since there's nowhere to put the methods.
	conc := si.Concrete.Obj()
	if conc != conc.Pkg().Scope().Lookup(conc.Name()) {
		return nil, nil, fmt.Errorf("local type %q cannot be stubbed", conc.Name())
	}

	// Parse the file defining the concrete type.
	concreteFilename := safetoken.StartPosition(snapshot.FileSet(), si.Concrete.Obj().Pos()).Filename
	concreteFH, err := snapshot.GetFile(ctx, span.URIFromPath(concreteFilename))
	if err != nil {
		return nil, nil, err
	}
	parsedConcreteFile, err := snapshot.ParseGo(ctx, concreteFH, ParseFull)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to parse file declaring implementation type: %w", err)
	}
	var (
		methodsSrc  []byte
		stubImports []*stubImport // additional imports needed for method stubs
	)
	if si.Interface.Pkg() == nil && si.Interface.Name() == "error" && si.Interface.Parent() == types.Universe {
		methodsSrc = stubErr(ctx, parsedConcreteFile.File, si, snapshot)
	} else {
		methodsSrc, stubImports, err = stubMethods(ctx, parsedConcreteFile.File, si, snapshot)
		if err != nil {
			return nil, nil, fmt.Errorf("stubMethods: %w", err)
		}
	}

	// Splice the methods into the file.
	// The insertion point is after the top-level declaration
	// enclosing the (package-level) type object.
	insertPos := parsedConcreteFile.File.End()
	for _, decl := range parsedConcreteFile.File.Decls {
		if decl.End() > conc.Pos() {
			insertPos = decl.End()
			break
		}
	}
	concreteSrc, err := concreteFH.Read()
	if err != nil {
		return nil, nil, fmt.Errorf("error reading concrete file source: %w", err)
	}
	insertOffset, err := safetoken.Offset(parsedConcreteFile.Tok, insertPos)
	if err != nil || insertOffset >= len(concreteSrc) {
		return nil, nil, fmt.Errorf("insertion position is past the end of the file")
	}
	var buf bytes.Buffer
	buf.Write(concreteSrc[:insertOffset])
	buf.WriteByte('\n')
	buf.Write(methodsSrc)
	buf.Write(concreteSrc[insertOffset:])

	// Re-parse it, splice in imports, pretty-print it.
	fset := token.NewFileSet()
	newF, err := parser.ParseFile(fset, parsedConcreteFile.File.Name.Name, buf.Bytes(), parser.ParseComments)
	if err != nil {
		return nil, nil, fmt.Errorf("could not reparse file: %w", err)
	}
	for _, imp := range stubImports {
		astutil.AddNamedImport(fset, newF, imp.Name, imp.Path)
	}
	var source strings.Builder
	if err := format.Node(&source, fset, newF); err != nil {
		return nil, nil, fmt.Errorf("format.Node: %w", err)
	}

	// Return the diff.
	diffs := snapshot.View().Options().ComputeEdits(string(parsedConcreteFile.Src), source.String())
	var edits []analysis.TextEdit
	for _, edit := range diffs {
		edits = append(edits, analysis.TextEdit{
			Pos:     parsedConcreteFile.Tok.Pos(edit.Start),
			End:     parsedConcreteFile.Tok.Pos(edit.End),
			NewText: []byte(edit.New),
		})
	}
	return snapshot.FileSet(), // to match snapshot.ParseGo above
		&analysis.SuggestedFix{TextEdits: edits},
		nil
}

// stubMethods returns the Go code of all methods
// that implement the given interface
func stubMethods(ctx context.Context, concreteFile *ast.File, si *stubmethods.StubInfo, snapshot Snapshot) ([]byte, []*stubImport, error) {
	concMS := types.NewMethodSet(types.NewPointer(si.Concrete.Obj().Type()))
	missing, err := missingMethods(ctx, snapshot, concMS, si.Concrete.Obj().Pkg(), si.Interface, map[string]struct{}{})
	if err != nil {
		return nil, nil, fmt.Errorf("missingMethods: %w", err)
	}
	if len(missing) == 0 {
		return nil, nil, fmt.Errorf("no missing methods found")
	}
	var (
		stubImports   []*stubImport
		methodsBuffer bytes.Buffer
	)
	for _, mi := range missing {
		for _, m := range mi.missing {
			// TODO(marwan-at-work): this should share the same logic with source.FormatVarType
			// as it also accounts for type aliases.
			sig := types.TypeString(m.Type(), stubmethods.RelativeToFiles(si.Concrete.Obj().Pkg(), concreteFile, mi.imports, func(name, path string) {
				for _, imp := range stubImports {
					if imp.Name == name && imp.Path == path {
						return
					}
				}
				stubImports = append(stubImports, &stubImport{name, path})
			}))
			_, err = methodsBuffer.Write(printStubMethod(methodData{
				Method:    m.Name(),
				Concrete:  getStubReceiver(si),
				Interface: deduceIfaceName(si.Concrete.Obj().Pkg(), si.Interface.Pkg(), si.Interface),
				Signature: strings.TrimPrefix(sig, "func"),
			}))
			if err != nil {
				return nil, nil, fmt.Errorf("error printing method: %w", err)
			}
			methodsBuffer.WriteRune('\n')
		}
	}
	return methodsBuffer.Bytes(), stubImports, nil
}

// stubErr reurns the Go code implementation
// of an error interface relevant to the
// concrete type
func stubErr(ctx context.Context, concreteFile *ast.File, si *stubmethods.StubInfo, snapshot Snapshot) []byte {
	return printStubMethod(methodData{
		Method:    "Error",
		Interface: "error",
		Concrete:  getStubReceiver(si),
		Signature: "() string",
	})
}

// getStubReceiver returns the concrete type's name as a method receiver.
// It accounts for type parameters if they exist.
func getStubReceiver(si *stubmethods.StubInfo) string {
	var concrete string
	if si.Pointer {
		concrete += "*"
	}
	concrete += si.Concrete.Obj().Name()
	concrete += FormatTypeParams(typeparams.ForNamed(si.Concrete))
	return concrete
}

type methodData struct {
	Method    string
	Interface string
	Concrete  string
	Signature string
}

// printStubMethod takes methodData and returns Go code that represents the given method such as:
//
//	// {{ .Method }} implements {{ .Interface }}
//	func ({{ .Concrete }}) {{ .Method }}{{ .Signature }} {
//		panic("unimplemented")
//	}
func printStubMethod(md methodData) []byte {
	var b bytes.Buffer
	fmt.Fprintf(&b, "// %s implements %s\n", md.Method, md.Interface)
	fmt.Fprintf(&b, "func (%s) %s%s {\n\t", md.Concrete, md.Method, md.Signature)
	fmt.Fprintln(&b, `panic("unimplemented")`)
	fmt.Fprintln(&b, "}")
	return b.Bytes()
}

func deduceIfaceName(concretePkg, ifacePkg *types.Package, ifaceObj types.Object) string {
	if concretePkg.Path() == ifacePkg.Path() {
		return ifaceObj.Name()
	}
	return fmt.Sprintf("%s.%s", ifacePkg.Name(), ifaceObj.Name())
}

func getStubNodes(pgf *ParsedGoFile, pRng protocol.Range) ([]ast.Node, token.Pos, error) {
	rng, err := pgf.RangeToTokenRange(pRng)
	if err != nil {
		return nil, 0, err
	}
	nodes, _ := astutil.PathEnclosingInterval(pgf.File, rng.Start, rng.End)
	return nodes, rng.Start, nil
}

/*
missingMethods takes a concrete type and returns any missing methods for the given interface as well as
any missing interface that might have been embedded to its parent. For example:

	type I interface {
		io.Writer
		Hello()
	}

returns

	[]*missingInterface{
		{
			iface: *types.Interface (io.Writer),
			file: *ast.File: io.go,
			missing []*types.Func{Write},
		},
		{
			iface: *types.Interface (I),
			file: *ast.File: myfile.go,
			missing: []*types.Func{Hello}
		},
	}
*/
func missingMethods(ctx context.Context, snapshot Snapshot, concMS *types.MethodSet, concPkg *types.Package, ifaceObj *types.TypeName, visited map[string]struct{}) ([]*missingInterface, error) {
	iface, ok := ifaceObj.Type().Underlying().(*types.Interface)
	if !ok {
		return nil, fmt.Errorf("expected %v to be an interface but got %T", iface, ifaceObj.Type().Underlying())
	}

	// The built-in error interface is special.
	if ifaceObj.Pkg() == nil && ifaceObj.Name() == "error" {
		var missingInterfaces []*missingInterface
		if concMS.Lookup(nil, "Error") == nil {
			errorMethod, _, _ := types.LookupFieldOrMethod(iface, false, nil, "Error")
			missingInterfaces = append(missingInterfaces, &missingInterface{
				iface:   ifaceObj,
				missing: []*types.Func{errorMethod.(*types.Func)},
			})
		}
		return missingInterfaces, nil
	}

	// Parse the imports from the file that declares the interface.
	ifaceFilename := safetoken.StartPosition(snapshot.FileSet(), ifaceObj.Pos()).Filename
	ifaceFH, err := snapshot.GetFile(ctx, span.URIFromPath(ifaceFilename))
	if err != nil {
		return nil, err
	}
	ifaceFile, err := snapshot.ParseGo(ctx, ifaceFH, ParseHeader)
	if err != nil {
		return nil, fmt.Errorf("error parsing imports from interface file: %w", err)
	}

	var missing []*types.Func

	// Add all the interface methods not defined by the concrete type to missing.
	for i := 0; i < iface.NumExplicitMethods(); i++ {
		method := iface.ExplicitMethod(i)
		if sel := concMS.Lookup(concPkg, method.Name()); sel == nil {
			// Concrete type does not have the interface method.
			if _, ok := visited[method.Name()]; !ok {
				missing = append(missing, method)
				visited[method.Name()] = struct{}{}
			}
		} else {
			// Concrete type does have the interface method.
			implSig := sel.Type().(*types.Signature)
			ifaceSig := method.Type().(*types.Signature)
			if !types.Identical(ifaceSig, implSig) {
				return nil, fmt.Errorf("mimsatched %q function signatures:\nhave: %s\nwant: %s", method.Name(), implSig, ifaceSig)
			}
		}
	}

	// Process embedded interfaces, recursively.
	//
	// TODO(adonovan): this whole computation could be expressed
	// more simply without recursion, driven by the method
	// sets of the interface and concrete types. Once the set
	// difference (missing methods) is computed, the imports
	// from the declaring file(s) could be loaded as needed.
	var missingInterfaces []*missingInterface
	for i := 0; i < iface.NumEmbeddeds(); i++ {
		eiface := iface.Embedded(i).Obj()
		em, err := missingMethods(ctx, snapshot, concMS, concPkg, eiface, visited)
		if err != nil {
			return nil, err
		}
		missingInterfaces = append(missingInterfaces, em...)
	}
	// The type checker is deterministic, but its choice of
	// ordering of embedded interfaces varies with Go version
	// (e.g. go1.17 was sorted, go1.18 was lexical order).
	// Sort to ensure test portability.
	sort.Slice(missingInterfaces, func(i, j int) bool {
		return missingInterfaces[i].iface.Id() < missingInterfaces[j].iface.Id()
	})

	if len(missing) > 0 {
		missingInterfaces = append(missingInterfaces, &missingInterface{
			iface:   ifaceObj,
			imports: ifaceFile.File.Imports,
			missing: missing,
		})
	}
	return missingInterfaces, nil
}

// missingInterface represents an interface
// that has all or some of its methods missing
// from the destination concrete type
type missingInterface struct {
	iface   *types.TypeName
	imports []*ast.ImportSpec // the interface's import environment
	missing []*types.Func
}

// stubImport represents a newly added import
// statement to the concrete type. If name is not
// empty, then that import is required to have that name.
type stubImport struct{ Name, Path string }
