// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package source

import (
	"context"
	"fmt"
	"go/ast"
	"go/types"
	"strings"

	"golang.org/x/tools/internal/event"
	"golang.org/x/tools/internal/imports"
	"golang.org/x/tools/internal/lsp/debug/tag"
	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/lsp/snippet"
	"golang.org/x/tools/internal/span"
)

// formatCompletion creates a completion item for a given candidate.
func (c *completer) item(ctx context.Context, cand candidate) (CompletionItem, error) {
	obj := cand.obj

	// Handle builtin types separately.
	if obj.Parent() == types.Universe {
		return c.formatBuiltin(ctx, cand)
	}

	var (
		label         = cand.name
		detail        = types.TypeString(obj.Type(), c.qf)
		insert        = label
		kind          = protocol.TextCompletion
		snip          *snippet.Builder
		protocolEdits []protocol.TextEdit
	)
	if obj.Type() == nil {
		detail = ""
	}

	// expandFuncCall mutates the completion label, detail, and snippet
	// to that of an invocation of sig.
	expandFuncCall := func(sig *types.Signature) error {
		s, err := newSignature(ctx, c.snapshot, c.pkg, c.file, "", sig, nil, c.qf)
		if err != nil {
			return err
		}
		snip = c.functionCallSnippet(label, s.params)
		detail = "func" + s.format()
		return nil
	}

	switch obj := obj.(type) {
	case *types.TypeName:
		detail, kind = formatType(obj.Type(), c.qf)
	case *types.Const:
		kind = protocol.ConstantCompletion
	case *types.Var:
		if _, ok := obj.Type().(*types.Struct); ok {
			detail = "struct{...}" // for anonymous structs
		} else if obj.IsField() {
			detail = formatVarType(ctx, c.snapshot, c.pkg, c.file, obj, c.qf)
		}
		if obj.IsField() {
			kind = protocol.FieldCompletion
			snip = c.structFieldSnippet(label, detail)
		} else {
			kind = protocol.VariableCompletion
		}
		if obj.Type() == nil {
			break
		}

		if sig, ok := obj.Type().Underlying().(*types.Signature); ok && cand.expandFuncCall {
			if err := expandFuncCall(sig); err != nil {
				return CompletionItem{}, err
			}
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
			if err := expandFuncCall(sig); err != nil {
				return CompletionItem{}, err
			}
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
	if cand.imp != nil {
		addlEdits, err := c.importEdits(ctx, cand.imp)
		if err != nil {
			return CompletionItem{}, err
		}

		protocolEdits = append(protocolEdits, addlEdits...)
		if kind != protocol.ModuleCompletion {
			if detail != "" {
				detail += " "
			}
			detail += fmt.Sprintf("(from %q)", cand.imp.importPath)
		}
	}

	// Prepend "&" or "*" operator as appropriate.
	var prefixOp string
	if cand.takeAddress {
		prefixOp = "&"
	} else if cand.makePointer {
		prefixOp = "*"
	} else if cand.dereference > 0 {
		prefixOp = strings.Repeat("*", cand.dereference)
	}

	if prefixOp != "" {
		// If we are in a selector, add an edit to place prefix before selector.
		if sel := enclosingSelector(c.path, c.pos); sel != nil {
			edits, err := prependEdit(c.snapshot.FileSet(), c.mapper, sel, prefixOp)
			if err != nil {
				return CompletionItem{}, err
			}
			protocolEdits = append(protocolEdits, edits...)
		} else {
			// If there is no selector, just stick the prefix at the start.
			insert = prefixOp + insert
		}

		label = prefixOp + label
	}

	// Add variadic "..." if we are filling in a variadic param.
	if cand.variadic {
		insert += "..."
		if snip != nil {
			snip.WriteText("...")
		}
	}

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
		obj:                 obj,
	}
	// If the user doesn't want documentation for completion items.
	if !c.opts.documentation {
		return item, nil
	}
	pos := c.snapshot.FileSet().Position(obj.Pos())

	// We ignore errors here, because some types, like "unsafe" or "error",
	// may not have valid positions that we can use to get documentation.
	if !pos.IsValid() {
		return item, nil
	}
	uri := span.URIFromPath(pos.Filename)

	// Find the source file of the candidate, starting from a package
	// that should have it in its dependencies.
	searchPkg := c.pkg
	if cand.imp != nil && cand.imp.pkg != nil {
		searchPkg = cand.imp.pkg
	}

	pgf, pkg, err := findPosInPackage(c.snapshot, searchPkg, obj.Pos())
	if err != nil {
		return item, nil
	}

	posToDecl, err := c.snapshot.PosToDecl(ctx, pgf)
	if err != nil {
		return CompletionItem{}, err
	}
	decl := posToDecl[obj.Pos()]
	if decl == nil {
		return item, nil
	}

	hover, err := hoverInfo(pkg, obj, decl)
	if err != nil {
		event.Error(ctx, "failed to find Hover", err, tag.URI.Of(uri))
		return item, nil
	}
	item.Documentation = hover.Synopsis
	if c.opts.fullDocumentation {
		item.Documentation = hover.FullDocumentation
	}
	return item, nil
}

// importEdits produces the text edits necessary to add the given import to the current file.
func (c *completer) importEdits(ctx context.Context, imp *importInfo) ([]protocol.TextEdit, error) {
	if imp == nil {
		return nil, nil
	}

	pgf, err := c.pkg.File(span.URIFromPath(c.filename))
	if err != nil {
		return nil, err
	}

	return computeOneImportFixEdits(ctx, c.snapshot, pgf, &imports.ImportFix{
		StmtInfo: imports.ImportInfo{
			ImportPath: imp.importPath,
			Name:       imp.name,
		},
		// IdentName is unused on this path and is difficult to get.
		FixType: imports.AddImport,
	})
}

func (c *completer) formatBuiltin(ctx context.Context, cand candidate) (CompletionItem, error) {
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
		sig, err := newBuiltinSignature(ctx, c.snapshot, obj.Name())
		if err != nil {
			return CompletionItem{}, err
		}
		item.Detail = "func" + sig.format()
		item.snippet = c.functionCallSnippet(obj.Name(), sig.params)
	case *types.TypeName:
		if types.IsInterface(obj.Type()) {
			item.Kind = protocol.InterfaceCompletion
		} else {
			item.Kind = protocol.ClassCompletion
		}
	case *types.Nil:
		item.Kind = protocol.VariableCompletion
	}
	return item, nil
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
