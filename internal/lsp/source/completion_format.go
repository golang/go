// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package source

import (
	"bytes"
	"fmt"
	"go/ast"
	"go/printer"
	"go/types"
	"strings"

	"golang.org/x/tools/internal/lsp/snippet"
)

// formatCompletion creates a completion item for a given types.Object.
func (c *completer) item(obj types.Object, score float64) CompletionItem {
	// Handle builtin types separately.
	if obj.Parent() == types.Universe {
		return c.formatBuiltin(obj, score)
	}

	var (
		label              = obj.Name()
		detail             = types.TypeString(obj.Type(), c.qf)
		insert             = label
		kind               CompletionItemKind
		plainSnippet       *snippet.Builder
		placeholderSnippet *snippet.Builder
	)

	switch o := obj.(type) {
	case *types.TypeName:
		detail, kind = formatType(o.Type(), c.qf)
	case *types.Const:
		if obj.Parent() == types.Universe {
			detail = ""
		} else {
			val := o.Val().ExactString()
			if !strings.ContainsRune(val, '\n') { // skip any multiline constants
				label += " = " + val
			}
		}
		kind = ConstantCompletionItem
	case *types.Var:
		if _, ok := o.Type().(*types.Struct); ok {
			detail = "struct{...}" // for anonymous structs
		}
		if o.IsField() {
			kind = FieldCompletionItem
			plainSnippet, placeholderSnippet = c.structFieldSnippets(label, detail)
		} else if c.isParameter(o) {
			kind = ParameterCompletionItem
		} else {
			kind = VariableCompletionItem
		}
	case *types.Func:
		sig, ok := o.Type().(*types.Signature)
		if !ok {
			break
		}
		params := formatParams(sig.Params(), sig.Variadic(), c.qf)
		results, writeParens := formatResults(sig.Results(), c.qf)
		label, detail = formatFunction(obj.Name(), params, results, writeParens)
		plainSnippet, placeholderSnippet = c.functionCallSnippets(obj.Name(), params)
		kind = FunctionCompletionItem
		if sig.Recv() != nil {
			kind = MethodCompletionItem
		}
	case *types.PkgName:
		kind = PackageCompletionItem
		detail = fmt.Sprintf("\"%s\"", o.Imported().Path())
	}
	detail = strings.TrimPrefix(detail, "untyped ")

	return CompletionItem{
		Label:              label,
		InsertText:         insert,
		Detail:             detail,
		Kind:               kind,
		Score:              score,
		Snippet:            plainSnippet,
		PlaceholderSnippet: placeholderSnippet,
	}
}

// isParameter returns true if the given *types.Var is a parameter
// of the enclosingFunction.
func (c *completer) isParameter(v *types.Var) bool {
	if c.enclosingFunction == nil {
		return false
	}
	for i := 0; i < c.enclosingFunction.Params().Len(); i++ {
		if c.enclosingFunction.Params().At(i) == v {
			return true
		}
	}
	return false
}

func (c *completer) formatBuiltin(obj types.Object, score float64) CompletionItem {
	item := CompletionItem{
		Label:      obj.Name(),
		InsertText: obj.Name(),
		Score:      score,
	}
	switch obj.(type) {
	case *types.Const:
		item.Kind = ConstantCompletionItem
	case *types.Builtin:
		fn := c.view.BuiltinPackage().Scope.Lookup(obj.Name())
		decl, ok := fn.Decl.(*ast.FuncDecl)
		if !ok {
			break
		}
		params, _ := c.formatFieldList(decl.Type.Params)
		results, writeResultParens := c.formatFieldList(decl.Type.Results)
		item.Label, item.Detail = formatFunction(obj.Name(), params, results, writeResultParens)
		item.Snippet, item.PlaceholderSnippet = c.functionCallSnippets(obj.Name(), params)
		item.Kind = FunctionCompletionItem
	case *types.TypeName:
		if types.IsInterface(obj.Type()) {
			item.Kind = InterfaceCompletionItem
		} else {
			item.Kind = TypeCompletionItem
		}
	case *types.Nil:
		item.Kind = VariableCompletionItem
	}
	return item
}

var replacer = strings.NewReplacer(
	`ComplexType`, `complex128`,
	`FloatType`, `float64`,
	`IntegerType`, `int`,
)

func (c *completer) formatFieldList(list *ast.FieldList) ([]string, bool) {
	if list == nil {
		return nil, false
	}
	var writeResultParens bool
	var result []string
	for i := 0; i < len(list.List); i++ {
		if i >= 1 {
			writeResultParens = true
		}
		p := list.List[i]
		cfg := printer.Config{Mode: printer.UseSpaces | printer.TabIndent, Tabwidth: 4}
		b := &bytes.Buffer{}
		if err := cfg.Fprint(b, c.view.FileSet(), p.Type); err != nil {
			c.view.Logger().Errorf(c.ctx, "unable to print type %v", p.Type)
			continue
		}
		typ := replacer.Replace(b.String())
		if len(p.Names) == 0 {
			result = append(result, fmt.Sprintf("%s", typ))
		}
		for _, name := range p.Names {
			if name.Name != "" {
				if i == 0 {
					writeResultParens = true
				}
				result = append(result, fmt.Sprintf("%s %s", name.Name, typ))
			} else {
				result = append(result, fmt.Sprintf("%s", typ))
			}
		}
	}
	return result, writeResultParens
}

// qualifier returns a function that appropriately formats a types.PkgName
// appearing in a *ast.File.
func qualifier(f *ast.File, pkg *types.Package, info *types.Info) types.Qualifier {
	// Construct mapping of import paths to their defined or implicit names.
	imports := make(map[*types.Package]string)
	for _, imp := range f.Imports {
		var obj types.Object
		if imp.Name != nil {
			obj = info.Defs[imp.Name]
		} else {
			obj = info.Implicits[imp]
		}
		if pkgname, ok := obj.(*types.PkgName); ok {
			imports[pkgname.Imported()] = pkgname.Name()
		}
	}
	// Define qualifier to replace full package paths with names of the imports.
	return func(p *types.Package) string {
		if p == pkg {
			return ""
		}
		if name, ok := imports[p]; ok {
			return name
		}
		return p.Name()
	}
}
