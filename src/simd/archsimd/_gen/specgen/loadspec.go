// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package specgen

import (
	"fmt"
	"go/ast"
	"go/build"
	"go/importer"
	"go/parser"
	"go/token"
	"go/types"
	"path/filepath"
	"simd/archsimd/_gen/specgen/specexpr"
	"strings"
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

	VecType   types.Type // Uninstantiated Vec type
	ArrayType types.Type // Uninstantiated Array type
	UintNType types.Type // Uninstantiated UintN type
}

// specFunc represents an exported function in the spec source.
type specFunc struct {
	Pkg          *specPackage
	Name         string       // Source name
	NameTmpl     specTemplate // API name template from `//specgen:name` directive, or same as Name.
	Pos          token.Pos
	Doc          specTemplate
	Category     string
	Commutative  bool
	Sig          *types.Signature
	TypeParams   []*types.TypeParam
	Params       []*types.Var
	Results      []*types.Var
	Requirements []specexpr.Expr
}

// specTemplate is a template string, with placeholders of the form `{var}`,
// which will be replaced with variable values from the solver.
type specTemplate struct {
	tmpl   string   // raw template string including patterns
	fields [][2]int // start:end ranges of fields, including '{}'s, in ascending order
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

	var pkg specPackage

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
					default:
						ctx.at(dir.Pos()).errorf("unknown //specgen directive")
					}
				}
			}
		}

		for _, decl := range file.Decls {
			d, ok := decl.(*ast.FuncDecl)
			if !ok || !d.Name.IsExported() {
				continue
			}

			obj := typesPkg.Scope().Lookup(d.Name.Name)
			if obj == nil {
				continue
			}
			fn, ok := obj.(*types.Func)
			if !ok {
				continue
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
				Pkg:        &pkg,
				Name:       d.Name.Name,
				Pos:        decl.Pos(),
				Sig:        sig,
				TypeParams: typeParams,
				Params:     params,
				Results:    results,
				Category:   category,
			}
			f.NameTmpl = specTemplate{tmpl: f.Name}
			if d.Doc != nil {
				var err error
				f.Doc, err = newSpecTemplate(d.Doc.Text())
				if err != nil {
					ctx.at(d.Doc.Pos()).errorf("malformed doc comment: %s", err)
				}
				for _, comment := range d.Doc.List {
					if dir, ok := directives[comment]; ok {
						delete(directives, comment)
						switch dir.Name {
						default:
							panic("directive lists out of sync")
						case "name":
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
				}
			}

			if opts.Filter != nil && !opts.Filter(d) {
				continue
			}

			funcs = append(funcs, f)
		}

		for _, dir := range directives {
			ctx.at(dir.Pos()).errorf("//%s:%s directive must be attached to a function", dir.Tool, dir.Name)
		}
	}

	lookupType := func(name string) types.Type {
		obj := typesPkg.Scope().Lookup(name)
		if obj == nil {
			ctx.errorf("type %q missing from package %s", name, typesPkg.Path())
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
	typeElems := make(map[types.Type]specexpr.Basic)
	elemTypes := make(map[specexpr.Basic]types.Type)
	if eltOrMask := lookupType("EltOrMask"); eltOrMask != nil {
		for _, elt := range typeSet(eltOrMask) {
			basic := shapeElemType(elt)
			typeElems[elt] = basic
			elemTypes[basic] = elt
		}
	}
	typeWidths := make(map[types.Type]specexpr.Num)
	widthTypes := make(map[specexpr.Num]types.Type)
	if width := lookupType("Width"); width != nil {
		for _, width := range typeSet(width) {
			val := shapeWidthVal(width)
			typeWidths[width] = val
			widthTypes[val] = width
		}
	}

	// Gather other known types
	vecType := lookupType("Vec")
	arrayType := lookupType("Array")
	uintNType := lookupType("UintN")

	pkg = specPackage{
		Fset:       &ctx.root.fset,
		Pkg:        typesPkg,
		TypesInfo:  info,
		Funcs:      funcs,
		TypeElems:  typeElems,
		TypeWidths: typeWidths,
		ElemTypes:  elemTypes,
		WidthTypes: widthTypes,
		VecType:    vecType,
		ArrayType:  arrayType,
		UintNType:  uintNType,
	}
	return &pkg
}

// newSpecTemplate parses spec template.
func newSpecTemplate(tmpl string) (specTemplate, error) {
	if !strings.ContainsAny(tmpl, "{}") {
		return specTemplate{tmpl, nil}, nil
	}

	var fields [][2]int
	for i := 0; i < len(tmpl); i++ {
		switch tmpl[i] {
		case '{':
			j := i + strings.IndexByte(tmpl[i:], '}') + 1
			if j <= i {
				return specTemplate{}, fmt.Errorf("unclosed '{' in template %q", tmpl)
			}
			fields = append(fields, [2]int{i, j})
			i = j - 1
		case '}':
			return specTemplate{}, fmt.Errorf("unmatched '}' in template %q", tmpl)
		}
	}
	return specTemplate{
		tmpl:   tmpl,
		fields: fields,
	}, nil
}

// expand replaces placeholders in template s by calling the lookup function to
// resolve their values.
func (s *specTemplate) expand(lookup func(string) string) string {
	if len(s.fields) == 0 {
		return s.tmpl
	}
	var buf strings.Builder
	pos := 0
	for _, field := range s.fields {
		buf.WriteString(s.tmpl[pos:field[0]])
		val := lookup(s.tmpl[field[0]+1 : field[1]-1])
		buf.WriteString(val)
		pos = field[1]
	}
	buf.WriteString(s.tmpl[pos:])
	return buf.String()
}
