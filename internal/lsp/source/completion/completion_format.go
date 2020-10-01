// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package completion

import (
	"context"
	"fmt"
	"go/types"
	"strings"

	"golang.org/x/tools/internal/event"
	"golang.org/x/tools/internal/imports"
	"golang.org/x/tools/internal/lsp/debug/tag"
	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/lsp/snippet"
	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/span"
	errors "golang.org/x/xerrors"
)

// item formats a candidate to a CompletionItem.
func (c *completer) item(ctx context.Context, cand candidate) (CompletionItem, error) {
	obj := cand.obj

	// if the object isn't a valid match against the surrounding, return early.
	matchScore := c.matcher.Score(cand.name)
	if matchScore <= 0 {
		return CompletionItem{}, errors.New("not a surrounding match")
	}
	cand.score *= float64(matchScore)

	// Ignore deep candidates that wont be in the MaxDeepCompletions anyway.
	if len(cand.path) != 0 && !c.deepState.isHighScore(cand.score) {
		return CompletionItem{}, errors.New("not a high scoring candidate")
	}

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
	expandFuncCall := func(sig *types.Signature) {
		s := source.NewSignature(ctx, c.snapshot, c.pkg, c.file, "", sig, nil, c.qf)
		snip = c.functionCallSnippet(label, s.Params())
		detail = "func" + s.Format()
	}

	switch obj := obj.(type) {
	case *types.TypeName:
		detail, kind = source.FormatType(obj.Type(), c.qf)
	case *types.Const:
		kind = protocol.ConstantCompletion
	case *types.Var:
		if _, ok := obj.Type().(*types.Struct); ok {
			detail = "struct{...}" // for anonymous structs
		} else if obj.IsField() {
			detail = source.FormatVarType(ctx, c.snapshot, c.pkg, c.file, obj, c.qf)
		}
		if obj.IsField() {
			kind = protocol.FieldCompletion
			snip = c.structFieldSnippet(cand, label, detail)
		} else {
			kind = protocol.VariableCompletion
		}
		if obj.Type() == nil {
			break
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
	if cand.imp != nil {
		addlEdits, err := c.importEdits(cand.imp)
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
	// override computed detail with provided detail, if something is provided.
	if cand.detail != "" {
		detail = cand.detail
	}
	item := CompletionItem{
		Label:               label,
		InsertText:          insert,
		AdditionalTextEdits: protocolEdits,
		Detail:              detail,
		Kind:                kind,
		Score:               cand.score,
		Depth:               len(cand.path),
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

	pgf, pkg, err := source.FindPosInPackage(c.snapshot, searchPkg, obj.Pos())
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

	hover, err := source.HoverInfo(pkg, obj, decl)
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
func (c *completer) importEdits(imp *importInfo) ([]protocol.TextEdit, error) {
	if imp == nil {
		return nil, nil
	}

	pgf, err := c.pkg.File(span.URIFromPath(c.filename))
	if err != nil {
		return nil, err
	}

	return source.ComputeOneImportFixEdits(c.snapshot, pgf, &imports.ImportFix{
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
		sig, err := source.NewBuiltinSignature(ctx, c.snapshot, obj.Name())
		if err != nil {
			return CompletionItem{}, err
		}
		item.Detail = "func" + sig.Format()
		item.snippet = c.functionCallSnippet(obj.Name(), sig.Params())
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
