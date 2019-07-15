// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package source

import (
	"bytes"
	"context"
	"fmt"
	"go/ast"
	"go/printer"
	"go/types"
	"strings"

	"golang.org/x/tools/internal/lsp/snippet"
	"golang.org/x/tools/internal/lsp/telemetry/log"
	"golang.org/x/tools/internal/lsp/telemetry/tag"
	"golang.org/x/tools/internal/span"
)

// formatCompletion creates a completion item for a given candidate.
func (c *completer) item(cand candidate) (CompletionItem, error) {
	obj := cand.obj

	// Handle builtin types separately.
	if obj.Parent() == types.Universe {
		return c.formatBuiltin(cand)
	}

	var (
		label              = c.deepState.chainString(obj.Name())
		detail             = types.TypeString(obj.Type(), c.qf)
		insert             = label
		kind               CompletionItemKind
		plainSnippet       *snippet.Builder
		placeholderSnippet *snippet.Builder
	)

	// expandFuncCall mutates the completion label, detail, and snippets
	// to that of an invocation of sig.
	expandFuncCall := func(sig *types.Signature) {
		params := formatParams(sig.Params(), sig.Variadic(), c.qf)
		plainSnippet, placeholderSnippet = c.functionCallSnippets(label, params)
		results, writeParens := formatResults(sig.Results(), c.qf)
		detail = "func" + formatFunction(params, results, writeParens)
	}

	switch obj := obj.(type) {
	case *types.TypeName:
		detail, kind = formatType(obj.Type(), c.qf)
	case *types.Const:
		kind = ConstantCompletionItem
	case *types.Var:
		if _, ok := obj.Type().(*types.Struct); ok {
			detail = "struct{...}" // for anonymous structs
		}
		if obj.IsField() {
			kind = FieldCompletionItem
			plainSnippet, placeholderSnippet = c.structFieldSnippets(label, detail)
		} else if c.isParameter(obj) {
			kind = ParameterCompletionItem
		} else {
			kind = VariableCompletionItem
		}

		if sig, ok := obj.Type().Underlying().(*types.Signature); ok && cand.expandFuncCall {
			expandFuncCall(sig)
		}
	case *types.Func:
		sig, ok := obj.Type().Underlying().(*types.Signature)
		if !ok {
			break
		}
		kind = FunctionCompletionItem
		if sig != nil && sig.Recv() != nil {
			kind = MethodCompletionItem
		}

		if cand.expandFuncCall {
			expandFuncCall(sig)
		}
	case *types.PkgName:
		kind = PackageCompletionItem
		detail = fmt.Sprintf("%q", obj.Imported().Path())
	}

	detail = strings.TrimPrefix(detail, "untyped ")
	item := CompletionItem{
		Label:              label,
		InsertText:         insert,
		Detail:             detail,
		Kind:               kind,
		Score:              cand.score,
		Depth:              len(c.deepState.chain),
		plainSnippet:       plainSnippet,
		placeholderSnippet: placeholderSnippet,
	}
	if c.opts.WantDocumentaton {
		declRange, err := objToRange(c.ctx, c.view.Session().Cache().FileSet(), obj)
		if err != nil {
			log.Error(c.ctx, "failed to get declaration range for object", err, tag.Of("Name", obj.Name()))
			goto Return
		}
		pos := declRange.FileSet.Position(declRange.Start)
		if !pos.IsValid() {
			log.Error(c.ctx, "invalid declaration position", err, tag.Of("Label", item.Label))
			goto Return
		}
		uri := span.FileURI(pos.Filename)
		f, err := c.view.GetFile(c.ctx, uri)
		if err != nil {
			log.Error(c.ctx, "unable to get file", err, tag.Of("URI", uri))
			goto Return
		}
		gof, ok := f.(GoFile)
		if !ok {
			log.Error(c.ctx, "declaration in a Go file", err, tag.Of("Label", item.Label))
			goto Return
		}
		ident, err := Identifier(c.ctx, c.view, gof, declRange.Start)
		if err != nil {
			log.Error(c.ctx, "no identifier", err, tag.Of("Name", obj.Name()))
			goto Return
		}
		documentation, err := ident.Documentation(c.ctx, SynopsisDocumentation)
		if err != nil {
			log.Error(c.ctx, "no documentation", err, tag.Of("Name", obj.Name()))
			goto Return
		}
		item.Documentation = documentation
	}
Return:
	return item, nil
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

func (c *completer) formatBuiltin(cand candidate) (CompletionItem, error) {
	obj := cand.obj
	item := CompletionItem{
		Label:      obj.Name(),
		InsertText: obj.Name(),
		Score:      cand.score,
	}
	switch obj.(type) {
	case *types.Const:
		item.Kind = ConstantCompletionItem
	case *types.Builtin:
		item.Kind = FunctionCompletionItem
		decl, ok := lookupBuiltinDecl(c.view, obj.Name()).(*ast.FuncDecl)
		if !ok {
			break
		}
		params, _ := formatFieldList(c.ctx, c.view, decl.Type.Params)
		results, writeResultParens := formatFieldList(c.ctx, c.view, decl.Type.Results)
		item.Label = obj.Name()
		item.Detail = "func" + formatFunction(params, results, writeResultParens)
		item.plainSnippet, item.placeholderSnippet = c.functionCallSnippets(obj.Name(), params)
	case *types.TypeName:
		if types.IsInterface(obj.Type()) {
			item.Kind = InterfaceCompletionItem
		} else {
			item.Kind = TypeCompletionItem
		}
	case *types.Nil:
		item.Kind = VariableCompletionItem
	}
	return item, nil
}

var replacer = strings.NewReplacer(
	`ComplexType`, `complex128`,
	`FloatType`, `float64`,
	`IntegerType`, `int`,
)

func formatFieldList(ctx context.Context, v View, list *ast.FieldList) ([]string, bool) {
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
		if err := cfg.Fprint(b, v.Session().Cache().FileSet(), p.Type); err != nil {
			log.Error(ctx, "unable to print type", nil, tag.Of("Type", p.Type))
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
