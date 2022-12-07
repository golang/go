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
	"strings"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/ast/astutil"
	"golang.org/x/tools/gopls/internal/lsp/analysis/stubmethods"
	"golang.org/x/tools/gopls/internal/lsp/protocol"
	"golang.org/x/tools/gopls/internal/lsp/safetoken"
	"golang.org/x/tools/gopls/internal/span"
	"golang.org/x/tools/internal/typeparams"
)

func stubSuggestedFixFunc(ctx context.Context, snapshot Snapshot, fh VersionedFileHandle, rng protocol.Range) (*token.FileSet, *analysis.SuggestedFix, error) {
	pkg, pgf, err := GetTypedFile(ctx, snapshot, fh, NarrowestPackage)
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

	// Parse the file defining the concrete type.
	concreteFilename := snapshot.FileSet().Position(si.Concrete.Obj().Pos()).Filename
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
	}
	if err != nil {
		return nil, nil, fmt.Errorf("stubMethods: %w", err)
	}
	nodes, _ = astutil.PathEnclosingInterval(parsedConcreteFile.File, si.Concrete.Obj().Pos(), si.Concrete.Obj().Pos())
	concreteSrc, err := concreteFH.Read()
	if err != nil {
		return nil, nil, fmt.Errorf("error reading concrete file source: %w", err)
	}
	insertPos, err := safetoken.Offset(parsedConcreteFile.Tok, nodes[1].End())
	if err != nil || insertPos >= len(concreteSrc) {
		return nil, nil, fmt.Errorf("insertion position is past the end of the file")
	}
	var buf bytes.Buffer
	buf.Write(concreteSrc[:insertPos])
	buf.WriteByte('\n')
	buf.Write(methodsSrc)
	buf.Write(concreteSrc[insertPos:])
	fset := token.NewFileSet()
	newF, err := parser.ParseFile(fset, parsedConcreteFile.File.Name.Name, buf.Bytes(), parser.ParseComments)
	if err != nil {
		return nil, nil, fmt.Errorf("could not reparse file: %w", err)
	}
	for _, imp := range stubImports {
		astutil.AddNamedImport(fset, newF, imp.Name, imp.Path)
	}
	var source bytes.Buffer
	err = format.Node(&source, fset, newF)
	if err != nil {
		return nil, nil, fmt.Errorf("format.Node: %w", err)
	}
	diffs := snapshot.View().Options().ComputeEdits(string(parsedConcreteFile.Src), source.String())
	tf := parsedConcreteFile.Mapper.TokFile
	var edits []analysis.TextEdit
	for _, edit := range diffs {
		edits = append(edits, analysis.TextEdit{
			Pos:     tf.Pos(edit.Start),
			End:     tf.Pos(edit.End),
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
	spn, err := pgf.Mapper.RangeSpan(pRng)
	if err != nil {
		return nil, 0, err
	}
	rng, err := spn.Range(pgf.Mapper.TokFile)
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
func missingMethods(ctx context.Context, snapshot Snapshot, concMS *types.MethodSet, concPkg *types.Package, ifaceObj types.Object, visited map[string]struct{}) ([]*missingInterface, error) {
	iface, ok := ifaceObj.Type().Underlying().(*types.Interface)
	if !ok {
		return nil, fmt.Errorf("expected %v to be an interface but got %T", iface, ifaceObj.Type().Underlying())
	}
	missing := []*missingInterface{}
	for i := 0; i < iface.NumEmbeddeds(); i++ {
		eiface := iface.Embedded(i).Obj()
		em, err := missingMethods(ctx, snapshot, concMS, concPkg, eiface, visited)
		if err != nil {
			return nil, err
		}
		missing = append(missing, em...)
	}

	// Parse the imports from the file that declares the interface.
	ifaceFilename := snapshot.FileSet().Position(ifaceObj.Pos()).Filename
	ifaceFH, err := snapshot.GetFile(ctx, span.URIFromPath(ifaceFilename))
	if err != nil {
		return nil, err
	}
	ifaceFile, err := snapshot.ParseGo(ctx, ifaceFH, ParseHeader)
	if err != nil {
		return nil, fmt.Errorf("error parsing imports from interface file: %w", err)
	}
	mi := &missingInterface{
		iface:   iface,
		imports: ifaceFile.File.Imports,
	}
	for i := 0; i < iface.NumExplicitMethods(); i++ {
		method := iface.ExplicitMethod(i)
		// if the concrete type does not have the interface method
		if concMS.Lookup(concPkg, method.Name()) == nil {
			if _, ok := visited[method.Name()]; !ok {
				mi.missing = append(mi.missing, method)
				visited[method.Name()] = struct{}{}
			}
		}
		if sel := concMS.Lookup(concPkg, method.Name()); sel != nil {
			implSig := sel.Type().(*types.Signature)
			ifaceSig := method.Type().(*types.Signature)
			if !types.Identical(ifaceSig, implSig) {
				return nil, fmt.Errorf("mimsatched %q function signatures:\nhave: %s\nwant: %s", method.Name(), implSig, ifaceSig)
			}
		}
	}
	if len(mi.missing) > 0 {
		missing = append(missing, mi)
	}
	return missing, nil
}

// missingInterface represents an interface
// that has all or some of its methods missing
// from the destination concrete type
type missingInterface struct {
	iface   *types.Interface
	imports []*ast.ImportSpec // the interface's import environment
	missing []*types.Func
}

// stubImport represents a newly added import
// statement to the concrete type. If name is not
// empty, then that import is required to have that name.
type stubImport struct{ Name, Path string }
