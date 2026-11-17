// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package midway

import (
	"fmt"
	"strings"
	"testing"

	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
)

type rewriteImporter map[string]*types2.Package

func (imports rewriteImporter) Import(path string) (*types2.Package, error) {
	if pkg := imports[path]; pkg != nil {
		return pkg, nil
	}
	return nil, fmt.Errorf("unexpected rewrite import %q", path)
}

func TestSIMDRewriteImports(t *testing.T) {
	for _, tc := range []struct {
		name   string
		source string
	}{
		{"missing", `func Boundary() {}`},
		{"package_shadows", `var simd, bridge int; func Boundary() { _, _ = simd, bridge }`},
		{"parameter_shadows", `func Boundary(simd, bridge int) {}`},
		{"aliased", `import vector "simd"; var _ = vector.VectorBitSize; func Boundary() {}`},
		{"dot", `import . "simd"; var _ = VectorBitSize; func Boundary() {}`},
		{"blank", `import _ "simd"; func Boundary() {}`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			file, err := syntax.Parse(syntax.NewFileBase("consumer.go"), strings.NewReader("package consumer; "+tc.source), nil, nil, 0)
			if err != nil {
				t.Fatal(err)
			}
			simd := types2.NewPackage("simd", "simd")
			for _, fn := range []struct {
				name   string
				result *types2.Basic
			}{{"VectorBitSize", types2.Typ[types2.Int]}, {"Emulated", types2.Typ[types2.Bool]}} {
				result := types2.NewVar(syntax.Pos{}, simd, "", fn.result)
				signature := types2.NewSignatureType(nil, nil, nil, nil, types2.NewTuple(result), false)
				simd.Scope().Insert(types2.NewFunc(syntax.Pos{}, simd, fn.name, signature))
			}
			bridge := types2.NewPackage("simd/internal/bridge", "bridge")
			bridge.Scope().Insert(types2.NewTypeName(syntax.Pos{}, bridge, "ZeroSized", types2.NewStruct(nil, nil)))
			simd.MarkComplete()
			bridge.MarkComplete()
			conf := types2.Config{Importer: rewriteImporter{simd.Path(): simd, bridge.Path(): bridge}}
			info := &types2.Info{
				StoreTypesInSyntax: true,
				Defs:               make(map[*syntax.Name]types2.Object),
				Uses:               make(map[*syntax.Name]types2.Object),
				Implicits:          make(map[syntax.Node]types2.Object),
			}
			pkg, err := conf.Check("example.com/consumer", []*syntax.File{file}, info)
			if err != nil {
				t.Fatal(err)
			}
			analyzer := NewAnalyzer(pkg, info)
			analyzer.isDependentObj[pkg.Scope().Lookup("Boundary")] = true
			rewriter := NewRewriter(pkg, info, analyzer, []int{0, 128})
			rewriter.Rewrite([]*syntax.File{file})
			if _, err := conf.Check("example.com/consumer", []*syntax.File{file}, nil); err != nil {
				t.Logf("rewritten check: %v", err)
				t.Fatal("rewritten SIMD imports are not self-contained")
			}
			for _, decl := range file.DeclList {
				v, ok := decl.(*syntax.VarDecl)
				if ok && len(v.NameList) == 1 && v.NameList[0].Value == "_" {
					if _, called := v.Values.(*syntax.CallExpr); called {
						t.Fatal("import retention added an initialization-time call")
					}
				}
			}
		})
	}
}
