// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package source

import (
	"context"
	"fmt"
	"go/ast"
	"go/constant"
	"go/token"
	"go/types"
	"strings"

	"golang.org/x/tools/internal/event"
	"golang.org/x/tools/internal/lsp/lsppos"
	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/typeparams"
)

const (
	maxLabelLength = 28
)

func InlayHint(ctx context.Context, snapshot Snapshot, fh FileHandle, _ protocol.Range) ([]protocol.InlayHint, error) {
	ctx, done := event.Start(ctx, "source.InlayHint")
	defer done()

	pkg, pgf, err := GetParsedFile(ctx, snapshot, fh, NarrowestPackage)
	if err != nil {
		return nil, fmt.Errorf("getting file for InlayHint: %w", err)
	}

	tmap := lsppos.NewTokenMapper(pgf.Src, pgf.Tok)
	info := pkg.GetTypesInfo()
	q := Qualifier(pgf.File, pkg.GetTypes(), info)

	var hints []protocol.InlayHint
	ast.Inspect(pgf.File, func(node ast.Node) bool {
		switch n := node.(type) {
		case *ast.CallExpr:
			hints = append(hints, parameterNames(n, tmap, info)...)
			hints = append(hints, funcTypeParams(n, tmap, info)...)
		case *ast.AssignStmt:
			hints = append(hints, assignVariableTypes(n, tmap, info, &q)...)
		case *ast.RangeStmt:
			hints = append(hints, rangeVariableTypes(n, tmap, info, &q)...)
		case *ast.GenDecl:
			hints = append(hints, constantValues(n, tmap, info)...)
		case *ast.CompositeLit:
			hints = append(hints, compositeLiterals(n, tmap, info, &q)...)
		}
		return true
	})
	return hints, nil
}

func parameterNames(node *ast.CallExpr, tmap *lsppos.TokenMapper, info *types.Info) []protocol.InlayHint {
	signature, ok := info.TypeOf(node.Fun).(*types.Signature)
	if !ok {
		return nil
	}

	var hints []protocol.InlayHint
	for i, v := range node.Args {
		start, ok := tmap.Position(v.Pos())
		if !ok {
			continue
		}
		params := signature.Params()
		// When a function has variadic params, we skip args after
		// params.Len().
		if i > params.Len()-1 {
			break
		}
		value := params.At(i).Name()
		// param.Name is empty for built-ins like append
		if value == "" {
			continue
		}
		if signature.Variadic() && i == params.Len()-1 {
			value = value + "..."
		}
		hints = append(hints, protocol.InlayHint{
			Position:     &start,
			Label:        buildLabel(value + ":"),
			Kind:         protocol.Parameter,
			PaddingRight: true,
		})
	}
	return hints
}

func funcTypeParams(node *ast.CallExpr, tmap *lsppos.TokenMapper, info *types.Info) []protocol.InlayHint {
	id, ok := node.Fun.(*ast.Ident)
	if !ok {
		return nil
	}
	inst := typeparams.GetInstances(info)[id]
	if inst.TypeArgs == nil {
		return nil
	}
	start, ok := tmap.Position(id.End())
	if !ok {
		return nil
	}
	var args []string
	for i := 0; i < inst.TypeArgs.Len(); i++ {
		args = append(args, inst.TypeArgs.At(i).String())
	}
	if len(args) == 0 {
		return nil
	}
	return []protocol.InlayHint{{
		Position: &start,
		Label:    buildLabel("[" + strings.Join(args, ", ") + "]"),
		Kind:     protocol.Type,
	}}
}

func assignVariableTypes(node *ast.AssignStmt, tmap *lsppos.TokenMapper, info *types.Info, q *types.Qualifier) []protocol.InlayHint {
	if node.Tok != token.DEFINE {
		return nil
	}
	var hints []protocol.InlayHint
	for _, v := range node.Lhs {
		if h := variableType(v, tmap, info, q); h != nil {
			hints = append(hints, *h)
		}
	}
	return hints
}

func rangeVariableTypes(node *ast.RangeStmt, tmap *lsppos.TokenMapper, info *types.Info, q *types.Qualifier) []protocol.InlayHint {
	var hints []protocol.InlayHint
	if h := variableType(node.Key, tmap, info, q); h != nil {
		hints = append(hints, *h)
	}
	if h := variableType(node.Value, tmap, info, q); h != nil {
		hints = append(hints, *h)
	}
	return hints
}

func variableType(e ast.Expr, tmap *lsppos.TokenMapper, info *types.Info, q *types.Qualifier) *protocol.InlayHint {
	typ := info.TypeOf(e)
	if typ == nil {
		return nil
	}
	end, ok := tmap.Position(e.End())
	if !ok {
		return nil
	}
	return &protocol.InlayHint{
		Position:    &end,
		Label:       buildLabel(types.TypeString(typ, *q)),
		Kind:        protocol.Type,
		PaddingLeft: true,
	}
}

func constantValues(node *ast.GenDecl, tmap *lsppos.TokenMapper, info *types.Info) []protocol.InlayHint {
	if node.Tok != token.CONST {
		return nil
	}

	var hints []protocol.InlayHint
	for _, v := range node.Specs {
		spec, ok := v.(*ast.ValueSpec)
		if !ok {
			continue
		}
		end, ok := tmap.Position(v.End())
		if !ok {
			continue
		}
		// Show hints when values are missing or at least one value is not
		// a basic literal.
		showHints := len(spec.Values) == 0
		checkValues := len(spec.Names) == len(spec.Values)
		var values []string
		for i, w := range spec.Names {
			obj, ok := info.ObjectOf(w).(*types.Const)
			if !ok || obj.Val().Kind() == constant.Unknown {
				return nil
			}
			if checkValues {
				switch spec.Values[i].(type) {
				case *ast.BadExpr:
					return nil
				case *ast.BasicLit:
				default:
					if obj.Val().Kind() != constant.Bool {
						showHints = true
					}
				}
			}
			values = append(values, fmt.Sprintf("%v", obj.Val()))
		}
		if !showHints || len(values) == 0 {
			continue
		}
		hints = append(hints, protocol.InlayHint{
			Position:    &end,
			Label:       buildLabel("= " + strings.Join(values, ", ")),
			PaddingLeft: true,
		})
	}
	return hints
}

func compositeLiterals(node *ast.CompositeLit, tmap *lsppos.TokenMapper, info *types.Info, q *types.Qualifier) []protocol.InlayHint {
	typ := info.TypeOf(node)
	if typ == nil {
		return nil
	}

	prefix := ""
	if t, ok := typ.(*types.Pointer); ok {
		typ = t.Elem()
		prefix = "&"
	}

	strct, ok := typ.Underlying().(*types.Struct)
	if !ok {
		return nil
	}

	var hints []protocol.InlayHint
	if node.Type == nil {
		// The type for this struct is implicit, add an inlay hint.
		if start, ok := tmap.Position(node.Lbrace); ok {
			hints = append(hints, protocol.InlayHint{
				Position: &start,
				Label:    buildLabel(fmt.Sprintf("%s%s", prefix, types.TypeString(typ, *q))),
				Kind:     protocol.Type,
			})
		}
	}

	for i, v := range node.Elts {
		if _, ok := v.(*ast.KeyValueExpr); !ok {
			start, ok := tmap.Position(v.Pos())
			if !ok {
				continue
			}
			if i > strct.NumFields()-1 {
				break
			}
			hints = append(hints, protocol.InlayHint{
				Position:     &start,
				Label:        buildLabel(strct.Field(i).Name() + ":"),
				Kind:         protocol.Parameter,
				PaddingRight: true,
			})
		}
	}
	return hints
}

func buildLabel(s string) []protocol.InlayHintLabelPart {
	label := protocol.InlayHintLabelPart{
		Value: s,
	}
	if len(s) > maxLabelLength+len("...") {
		label.Value = s[:maxLabelLength] + "..."
	}
	return []protocol.InlayHintLabelPart{label}
}
