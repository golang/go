// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package completion

import (
	"context"
	"fmt"
	"go/ast"
	"go/doc"
	"go/types"
	"strings"

	"golang.org/x/tools/internal/event"
	"golang.org/x/tools/internal/imports"
	"golang.org/x/tools/internal/lsp/debug/tag"
	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/lsp/snippet"
	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/span"
	"golang.org/x/tools/internal/typeparams"
	errors "golang.org/x/xerrors"
)

var (
	errNoMatch  = errors.New("not a surrounding match")
	errLowScore = errors.New("not a high scoring candidate")
)

// item formats a candidate to a CompletionItem.
func (c *completer) item(ctx context.Context, cand candidate) (CompletionItem, error) {
	obj := cand.obj

	// if the object isn't a valid match against the surrounding, return early.
	matchScore := c.matcher.Score(cand.name)
	if matchScore <= 0 {
		return CompletionItem{}, errNoMatch
	}
	cand.score *= float64(matchScore)

	// Ignore deep candidates that wont be in the MaxDeepCompletions anyway.
	if len(cand.path) != 0 && !c.deepState.isHighScore(cand.score) {
		return CompletionItem{}, errLowScore
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
		snip          snippet.Builder
		protocolEdits []protocol.TextEdit
	)
	if obj.Type() == nil {
		detail = ""
	}
	if isTypeName(obj) && c.wantTypeParams() {
		x := cand.obj.(*types.TypeName)
		if named, ok := x.Type().(*types.Named); ok {
			tp := typeparams.ForNamed(named)
			label += source.FormatTypeParams(tp)
			insert = label // maintain invariant above (label == insert)
		}
	}

	snip.WriteText(insert)

	switch obj := obj.(type) {
	case *types.TypeName:
		detail, kind = source.FormatType(obj.Type(), c.qf)
	case *types.Const:
		kind = protocol.ConstantCompletion
	case *types.Var:
		if _, ok := obj.Type().(*types.Struct); ok {
			detail = "struct{...}" // for anonymous structs
		} else if obj.IsField() {
			detail = source.FormatVarType(ctx, c.snapshot, c.pkg, obj, c.qf)
		}
		if obj.IsField() {
			kind = protocol.FieldCompletion
			c.structFieldSnippet(cand, detail, &snip)
		} else {
			kind = protocol.VariableCompletion
		}
		if obj.Type() == nil {
			break
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
	case *types.PkgName:
		kind = protocol.ModuleCompletion
		detail = fmt.Sprintf("%q", obj.Imported().Path())
	case *types.Label:
		kind = protocol.ConstantCompletion
		detail = "label"
	}

	var prefix string
	for _, mod := range cand.mods {
		switch mod {
		case reference:
			prefix = "&" + prefix
		case dereference:
			prefix = "*" + prefix
		case chanRead:
			prefix = "<-" + prefix
		}
	}

	var (
		suffix   string
		funcType = obj.Type()
	)
Suffixes:
	for _, mod := range cand.mods {
		switch mod {
		case invoke:
			if sig, ok := funcType.Underlying().(*types.Signature); ok {
				s := source.NewSignature(ctx, c.snapshot, c.pkg, sig, nil, c.qf)
				c.functionCallSnippet("", s.TypeParams(), s.Params(), &snip)
				if sig.Results().Len() == 1 {
					funcType = sig.Results().At(0).Type()
				}
				detail = "func" + s.Format()
			}

			if !c.opts.snippets {
				// Without snippets the candidate will not include "()". Don't
				// add further suffixes since they will be invalid. For
				// example, with snippets "foo()..." would become "foo..."
				// without snippets if we added the dotDotDot.
				break Suffixes
			}
		case takeSlice:
			suffix += "[:]"
		case takeDotDotDot:
			suffix += "..."
		case index:
			snip.WriteText("[")
			snip.WritePlaceholder(nil)
			snip.WriteText("]")
		}
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

	if cand.convertTo != nil {
		typeName := types.TypeString(cand.convertTo, c.qf)

		switch cand.convertTo.(type) {
		// We need extra parens when casting to these types. For example,
		// we need "(*int)(foo)", not "*int(foo)".
		case *types.Pointer, *types.Signature:
			typeName = "(" + typeName + ")"
		}

		prefix = typeName + "(" + prefix
		suffix = ")"
	}

	if prefix != "" {
		// If we are in a selector, add an edit to place prefix before selector.
		if sel := enclosingSelector(c.path, c.pos); sel != nil {
			edits, err := c.editText(sel.Pos(), sel.Pos(), prefix)
			if err != nil {
				return CompletionItem{}, err
			}
			protocolEdits = append(protocolEdits, edits...)
		} else {
			// If there is no selector, just stick the prefix at the start.
			insert = prefix + insert
			snip.PrependText(prefix)
		}
	}

	if suffix != "" {
		insert += suffix
		snip.WriteText(suffix)
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
		snippet:             &snip,
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

	// Find the source file of the candidate.
	pkg, err := source.FindPackageFromPos(ctx, c.snapshot, obj.Pos())
	if err != nil {
		return item, nil
	}

	decl, err := c.snapshot.PosToDecl(ctx, pkg, obj.Pos())
	if err != nil {
		return CompletionItem{}, err
	}
	hover, err := source.FindHoverContext(ctx, c.snapshot, pkg, obj, decl, nil)
	if err != nil {
		event.Error(ctx, "failed to find Hover", err, tag.URI.Of(uri))
		return item, nil
	}
	if c.opts.fullDocumentation {
		item.Documentation = hover.Comment.Text()
	} else {
		item.Documentation = doc.Synopsis(hover.Comment.Text())
	}
	// The desired pattern is `^// Deprecated`, but the prefix has been removed
	if strings.HasPrefix(hover.Comment.Text(), "Deprecated") {
		if c.snapshot.View().Options().CompletionTags {
			item.Tags = []protocol.CompletionItemTag{protocol.ComplDeprecated}
		} else if c.snapshot.View().Options().CompletionDeprecated {
			item.Deprecated = true
		}
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
		item.snippet = &snippet.Builder{}
		c.functionCallSnippet(obj.Name(), sig.TypeParams(), sig.Params(), item.snippet)
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

// decide if the type params (if any) should be part of the completion
// which only possible for types.Named and types.Signature
// (so far, only in receivers, e.g.; func (s *GENERIC[K, V])..., which is a types.Named)
func (c *completer) wantTypeParams() bool {
	// Need to be lexically in a receiver, and a child of an IndexListExpr
	// (but IndexListExpr only exists with go1.18)
	start := c.path[0].Pos()
	for i, nd := range c.path {
		if fd, ok := nd.(*ast.FuncDecl); ok {
			if i > 0 && fd.Recv != nil && start < fd.Recv.End() {
				return true
			} else {
				return false
			}
		}
	}
	return false
}
