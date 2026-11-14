// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package specgen

import (
	"go/ast"
	"go/build"
	"go/importer"
	"go/parser"
	"go/token"
	"go/types"
	"path/filepath"
	"simd/archsimd/_gen/specgen/specexpr"
)

// specPackage represents the parsed _gen/spec package.
type specPackage struct {
	Fset      *token.FileSet
	Pkg       *types.Package
	TypesInfo *types.Info
	Funcs     []*specFunc

	TypeElems  map[types.Type]specexpr.Basic
	TypeWidths map[types.Type]specexpr.Num

	ElemTypes  map[specexpr.Basic]types.Type
	WidthTypes map[specexpr.Num]types.Type

	VecType       types.Type // Uninstantiated Vec type
	ArrayType     types.Type // Uninstantiated Array type
	UintNType     types.Type // Uninstantiated UintN type
	VariationType types.Type // Uninstantiated Variation type
}

// specFunc represents an exported function in the spec source.
type specFunc struct {
	Pkg          *specPackage
	Name         string       // Source name
	NameTmpl     specTemplate // API name template from `//specgen:name` directive, or same as Name.
	Pos          token.Pos
	Doc          specTemplate // Comment markers removed; \n-terminated unless ""
	Category     string
	Commutative  bool
	Sig          *types.Signature
	TypeParams   []*types.TypeParam
	Params       []*types.Var
	Results      []*types.Var
	Requirements []specexpr.Expr
}

// specGoVersion is the oldest Go toolchain version that must be able to parse
// and type-check the spec package.
const specGoVersion = "go1.26"

// loadAndTypeCheck imports and parses the spec package in dir, ensuring it is
// target-independent, and type-checks it.
func loadAndTypeCheck(ctx context, dir string) (*types.Package, *types.Info, []*ast.File) {
	bp, err := build.ImportDir(dir, 0)
	if err != nil {
		ctx.errorf("failed to import spec directory %s: %s", dir, err)
		return nil, nil, nil
	}
	if len(bp.AllTags) > 0 {
		ctx.errorf("internal/spec must be target-independent, but found build tags: %v", bp.AllTags)
		return nil, nil, nil
	}

	var astFiles []*ast.File
	for _, name := range bp.GoFiles {
		filePath := filepath.Join(bp.Dir, name)
		file, err := parser.ParseFile(&ctx.root.fset, filePath, nil, parser.ParseComments)
		if err != nil {
			ctx.errorf("failed to parse %s: %s", filePath, err)
			continue
		}
		astFiles = append(astFiles, file)
	}

	if len(astFiles) == 0 {
		ctx.errorf("no Go source files found in directory %s", dir)
		return nil, nil, nil
	}

	info := &types.Info{
		Types:      make(map[ast.Expr]types.TypeAndValue),
		Defs:       make(map[*ast.Ident]types.Object),
		Uses:       make(map[*ast.Ident]types.Object),
		Implicits:  make(map[ast.Node]types.Object),
		Selections: make(map[*ast.SelectorExpr]*types.Selection),
		Scopes:     make(map[ast.Node]*types.Scope),
		Instances:  make(map[*ast.Ident]types.Instance),
	}

	// Note: Imported packages resolve against the running toolchain's standard
	// library, not the dev tree's. For math and math/bits that is immaterial.
	// Constraint: internal/spec may import only long-stable standard library packages.
	conf := types.Config{
		GoVersion: specGoVersion,
		Importer:  importer.ForCompiler(&ctx.root.fset, "source", nil),
		Error: func(err error) {
			ctx.errorf("%s", err)
		},
	}

	typesPkg, err := conf.Check("simd/internal/spec", &ctx.root.fset, astFiles, info)
	if err != nil && typesPkg == nil {
		return nil, nil, nil
	}
	if len(ctx.root.errors) > 0 {
		return nil, nil, nil
	}

	return typesPkg, info, astFiles
}

// loadSpecPackage parses the spec package in the given directory path.
func loadSpecPackage(ctx context, dir string, opts *LoadOptions) *specPackage {
	typesPkg, info, astFiles := loadAndTypeCheck(ctx, dir)
	if typesPkg == nil {
		return nil
	}

	pkg := &specPackage{
		Fset:      &ctx.root.fset,
		Pkg:       typesPkg,
		TypesInfo: info,
	}

	pkg.loadKnownTypes(ctx)

	// Gather exported functions
	var funcs []*specFunc
	for _, file := range astFiles {
		// Gather directives
		var category string
		directives := make(map[*ast.Comment]ast.Directive)
		for _, cg := range file.Comments {
			for _, comment := range cg.List {
				if dir, ok := ast.ParseDirective(comment.Slash, comment.Text); ok && dir.Tool == "specgen" {
					switch dir.Name {
					case "category":
						// File-level directive
						if category != "" {
							ctx.at(dir.Pos()).errorf("multiple category directives in file")
						}
						category = dir.Args
					case "name", "commutative", "require":
						// Gather other directives to process with decls
						directives[comment] = dir
					case "experiment":
						// do nothing for now; this indicates that the operation should remain a GOEXPERIMENT
					default:
						ctx.at(dir.Pos()).errorf("unknown //specgen directive")
					}
				}
			}
		}

		for _, decl := range file.Decls {
			switch decl := decl.(type) {
			case *ast.FuncDecl:
				f := pkg.loadSpecFunc(ctx, decl, category, directives)
				if f == nil {
					break
				}
				if opts.Filter != nil && !opts.Filter(decl) {
					break
				}
				funcs = append(funcs, f)
			}
		}

		for _, dir := range directives {
			ctx.at(dir.Pos()).errorf("//%s:%s directive must be attached to a function", dir.Tool, dir.Name)
		}
	}

	pkg.Funcs = funcs
	return pkg
}

// loadKnownTypes looks up the spec types that specgen gives special meaning to,
// and records them in pkg.
func (pkg *specPackage) loadKnownTypes(ctx context) {
	lookupType := func(name string) types.Type {
		obj := pkg.Pkg.Scope().Lookup(name)
		if obj == nil {
			ctx.errorf("type %q missing from package %s", name, pkg.Pkg.Path())
			return nil
		}
		tn, ok := obj.(*types.TypeName)
		if !ok {
			ctx.at(obj.Pos()).errorf("%s expected to be a type", obj.String())
			return nil
		}
		return tn.Type()
	}

	// Gather types corresponding to shape constraints
	pkg.TypeElems = make(map[types.Type]specexpr.Basic)
	pkg.ElemTypes = make(map[specexpr.Basic]types.Type)
	if eltOrMask := lookupType("EltOrMask"); eltOrMask != nil {
		for _, elt := range typeSet(eltOrMask) {
			basic := shapeElemType(elt)
			pkg.TypeElems[elt] = basic
			pkg.ElemTypes[basic] = elt
		}
	}
	pkg.TypeWidths = make(map[types.Type]specexpr.Num)
	pkg.WidthTypes = make(map[specexpr.Num]types.Type)
	if width := lookupType("Width"); width != nil {
		for _, width := range typeSet(width) {
			val := shapeWidthVal(width)
			pkg.TypeWidths[width] = val
			pkg.WidthTypes[val] = width
		}
	}

	// Gather other known types
	pkg.VecType = lookupType("Vec")
	pkg.ArrayType = lookupType("Array")
	pkg.UintNType = lookupType("UintN")
	pkg.VariationType = lookupType("Variation")
}

// loadSpecFunc returns the specFunc for the spec function declaration d in a
// file whose //specgen:category is category, or nil if d does not describe an
// API function. It removes the directives attached to d from directives.
func (pkg *specPackage) loadSpecFunc(ctx context, d *ast.FuncDecl, category string, directives map[*ast.Comment]ast.Directive) *specFunc {
	if d.Recv != nil || !d.Name.IsExported() {
		return nil
	}
	switch d.Name.Name {
	case "ScalableWidth", "SetScalableWidth":
		return nil
	}

	obj := pkg.Pkg.Scope().Lookup(d.Name.Name)
	if obj == nil {
		return nil
	}
	fn, ok := obj.(*types.Func)
	if !ok {
		return nil
	}

	sig := fn.Type().(*types.Signature)

	var typeParams []*types.TypeParam
	tparams := sig.TypeParams()
	for tparam := range tparams.TypeParams() {
		typeParams = append(typeParams, tparam)
	}

	var params []*types.Var
	p := sig.Params()
	for v := range p.Variables() {
		params = append(params, v)
	}

	var results []*types.Var
	r := sig.Results()
	for v := range r.Variables() {
		results = append(results, v)
	}

	f := &specFunc{
		Pkg:        pkg,
		Name:       d.Name.Name,
		Pos:        d.Pos(),
		Sig:        sig,
		TypeParams: typeParams,
		Params:     params,
		Results:    results,
		Category:   category,
	}
	f.NameTmpl = specTemplate{raw: f.Name}
	doc, dirs := loadDoc(ctx, d.Doc, directives)
	f.Doc = doc
	for _, dir := range dirs {
		switch dir.Name {
		default:
			panic("directive lists out of sync")
		case "name":
			var err error
			f.NameTmpl, err = newSpecTemplate(dir.Args)
			if err != nil {
				ctx.at(dir.Pos()).errorf("malformed //specgen:name directive: %s", err)
			}
		case "commutative":
			if dir.Args != "" {
				ctx.at(dir.Pos()).errorf("malformed //specgen:commutative directive: expected no argument")
			}
			f.Commutative = true
		case "require":
			args, err := dir.ParseArgs()
			if err != nil {
				ctx.at(dir.Pos()).errorf("malformed //specgen:require directive: %s", err)
				break
			}
			for _, arg := range args {
				expr, err := specexpr.ParseExpr(arg.Arg)
				if err != nil {
					ctx.at(arg.Pos).errorf("failed to parse require argument %q: %s", arg.Arg, err)
					continue
				}
				f.Requirements = append(f.Requirements, expr)
			}
		}
	}

	return f
}

// loadDoc parses the doc comment doc as a spec doc template. It returns the
// template and the specgen directives attached to doc, removing them from
// directives.
func loadDoc(ctx context, doc *ast.CommentGroup, directives map[*ast.Comment]ast.Directive) (specTemplate, []ast.Directive) {
	if doc == nil {
		return specTemplate{}, nil
	}

	tmpl, err := newSpecTemplate(doc.Text())
	if err != nil {
		ctx.at(doc.Pos()).errorf("malformed doc comment: %s", err)
	}

	var dirs []ast.Directive
	for _, comment := range doc.List {
		if dir, ok := directives[comment]; ok {
			delete(directives, comment)
			dirs = append(dirs, dir)
		}
	}
	return tmpl, dirs
}
