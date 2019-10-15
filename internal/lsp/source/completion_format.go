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

	"golang.org/x/tools/internal/imports"
	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/lsp/snippet"
	"golang.org/x/tools/internal/span"
	"golang.org/x/tools/internal/telemetry/log"
	"golang.org/x/tools/internal/telemetry/tag"
	errors "golang.org/x/xerrors"
)

// formatCompletion creates a completion item for a given candidate.
func (c *completer) item(cand candidate) (CompletionItem, error) {
	obj := cand.obj

	// Handle builtin types separately.
	if obj.Parent() == types.Universe {
		return c.formatBuiltin(cand), nil
	}

	var (
		label         = cand.name
		detail        = types.TypeString(obj.Type(), c.qf)
		insert        = label
		kind          = protocol.TextCompletion
		snip          *snippet.Builder
		protocolEdits []protocol.TextEdit
	)

	// expandFuncCall mutates the completion label, detail, and snippet
	// to that of an invocation of sig.
	expandFuncCall := func(sig *types.Signature) {
		params := formatParams(sig.Params(), sig.Variadic(), c.qf)
		snip = c.functionCallSnippet(label, params)
		results, writeParens := formatResults(sig.Results(), c.qf)
		detail = "func" + formatFunction(params, results, writeParens)
	}

	switch obj := obj.(type) {
	case *types.TypeName:
		detail, kind = formatType(obj.Type(), c.qf)
	case *types.Const:
		kind = protocol.ConstantCompletion
	case *types.Var:
		if _, ok := obj.Type().(*types.Struct); ok {
			detail = "struct{...}" // for anonymous structs
		}
		if obj.IsField() {
			kind = protocol.FieldCompletion
			snip = c.structFieldSnippet(label, detail)
		} else {
			kind = protocol.VariableCompletion
		}

		if sig, ok := obj.Type().Underlying().(*types.Signature); ok && cand.expandFuncCall {
			expandFuncCall(sig)
		}
	case *types.Func:
		sig, ok := obj.Type().Underlying().(*types.Signature)
		if !ok {
			break
		}
		kind = protocol.FunctionCompletion
		if sig != nil && sig.Recv() != nil {
			kind = protocol.MethodCompletion
		}

		if cand.expandFuncCall {
			expandFuncCall(sig)
		}
	case *types.PkgName:
		kind = protocol.ModuleCompletion
		detail = fmt.Sprintf("%q", obj.Imported().Path())
	case *types.Label:
		kind = protocol.ConstantCompletion
		detail = "label"
	}

	// If this candidate needs an additional import statement,
	// add the additional text edits needed.
	addlEdits, err := c.importEdits(cand.imp)
	if err != nil {
		return CompletionItem{}, err
	}
	protocolEdits = append(protocolEdits, addlEdits...)

	detail = strings.TrimPrefix(detail, "untyped ")
	item := CompletionItem{
		Label:               label,
		InsertText:          insert,
		AdditionalTextEdits: protocolEdits,
		Detail:              detail,
		Kind:                kind,
		Score:               cand.score,
		Depth:               len(c.deepState.chain),
		snippet:             snip,
	}
	// If the user doesn't want documentation for completion items.
	if !c.opts.Documentation {
		return item, nil
	}
	pos := c.view.Session().Cache().FileSet().Position(obj.Pos())

	// We ignore errors here, because some types, like "unsafe" or "error",
	// may not have valid positions that we can use to get documentation.
	if !pos.IsValid() {
		return item, nil
	}
	uri := span.FileURI(pos.Filename)
	ph, pkg, err := c.view.FindFileInPackage(c.ctx, uri, c.pkg)
	if err != nil {
		return CompletionItem{}, err
	}
	file, _, _, err := ph.Cached()
	if err != nil {
		return CompletionItem{}, err
	}
	if !(file.Pos() <= obj.Pos() && obj.Pos() <= file.End()) {
		return CompletionItem{}, errors.Errorf("no file for %s", obj.Name())
	}
	ident, err := findIdentifier(c.ctx, c.snapshot, pkg, file, obj.Pos())
	if err != nil {
		return CompletionItem{}, err
	}
	hover, err := ident.Hover(c.ctx)
	if err != nil {
		return CompletionItem{}, err
	}
	item.Documentation = hover.Synopsis
	if c.opts.FullDocumentation {
		item.Documentation = hover.FullDocumentation
	}
	return item, nil
}

// importEdits produces the text eddits necessary to add the given import to the current file.
func (c *completer) importEdits(imp *imports.ImportInfo) ([]protocol.TextEdit, error) {
	if imp == nil {
		return nil, nil
	}

	edit, err := addNamedImport(c.view.Session().Cache().FileSet(), c.file, imp.Name, imp.ImportPath)
	if err != nil {
		return nil, err
	}

	return ToProtocolEdits(c.mapper, edit)
}

func (c *completer) formatBuiltin(cand candidate) CompletionItem {
	obj := cand.obj
	item := CompletionItem{
		Label:      obj.Name(),
		InsertText: obj.Name(),
		Score:      cand.score,
	}
	switch obj.(type) {
	case *types.Const:
		item.Kind = protocol.ConstantCompletion
	case *types.Builtin:
		item.Kind = protocol.FunctionCompletion
		builtin := c.view.BuiltinPackage().Lookup(obj.Name())
		if obj == nil {
			break
		}
		decl, ok := builtin.Decl.(*ast.FuncDecl)
		if !ok {
			break
		}
		params, _ := formatFieldList(c.ctx, c.view, decl.Type.Params)
		results, writeResultParens := formatFieldList(c.ctx, c.view, decl.Type.Results)
		item.Label = obj.Name()
		item.Detail = "func" + formatFunction(params, results, writeResultParens)
		item.snippet = c.functionCallSnippet(obj.Name(), params)
	case *types.TypeName:
		if types.IsInterface(obj.Type()) {
			item.Kind = protocol.InterfaceCompletion
		} else {
			item.Kind = protocol.ClassCompletion
		}
	case *types.Nil:
		item.Kind = protocol.VariableCompletion
	}
	return item
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
			result = append(result, typ)
		}
		for _, name := range p.Names {
			if name.Name != "" {
				if i == 0 {
					writeResultParens = true
				}
				result = append(result, fmt.Sprintf("%s %s", name.Name, typ))
			} else {
				result = append(result, typ)
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
