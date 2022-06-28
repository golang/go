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

type InlayHintFunc func(node ast.Node, tmap *lsppos.TokenMapper, info *types.Info, q *types.Qualifier) []protocol.InlayHint

type Hint struct {
	Name string
	Doc  string
	Run  InlayHintFunc
}

const (
	ParameterNames             = "parameterNames"
	AssignVariableTypes        = "assignVariableTypes"
	ConstantValues             = "constantValues"
	RangeVariableTypes         = "rangeVariableTypes"
	CompositeLiteralTypes      = "compositeLiteralTypes"
	CompositeLiteralFieldNames = "compositeLiteralFields"
	FunctionTypeParameters     = "functionTypeParameters"
)

var AllInlayHints = map[string]*Hint{
	AssignVariableTypes: {
		Name: AssignVariableTypes,
		Doc: `Enable/disable inlay hints for variable types in assign statements:

	i/* int/*, j/* int/* := 0, len(r)-1`,
		Run: assignVariableTypes,
	},
	ParameterNames: {
		Name: ParameterNames,
		Doc: `Enable/disable inlay hints for parameter names:

	parseInt(/* str: */ "123", /* radix: */ 8)`,
		Run: parameterNames,
	},
	ConstantValues: {
		Name: ConstantValues,
		Doc: `Enable/disable inlay hints for constant values:

	const (
		KindNone   Kind = iota/* = 0*/
		KindPrint/*  = 1*/
		KindPrintf/* = 2*/
		KindErrorf/* = 3*/
	)`,
		Run: constantValues,
	},
	RangeVariableTypes: {
		Name: RangeVariableTypes,
		Doc: `Enable/disable inlay hints for variable types in range statements:

	for k/* int*/, v/* string/* := range []string{} {
		fmt.Println(k, v)
	}`,
		Run: rangeVariableTypes,
	},
	CompositeLiteralTypes: {
		Name: CompositeLiteralTypes,
		Doc: `Enable/disable inlay hints for composite literal types:

	for _, c := range []struct {
		in, want string
	}{
		/*struct{ in string; want string }*/{"Hello, world", "dlrow ,olleH"},
	}`,
		Run: compositeLiteralTypes,
	},
	CompositeLiteralFieldNames: {
		Name: CompositeLiteralFieldNames,
		Doc: `Enable/disable inlay hints for composite literal field names:

	{in: "Hello, world", want: "dlrow ,olleH"}`,
		Run: compositeLiteralFields,
	},
	FunctionTypeParameters: {
		Name: FunctionTypeParameters,
		Doc: `Enable/disable inlay hints for implicit type parameters on generic functions:

	myFoo/*[int, string]*/(1, "hello")`,
		Run: funcTypeParams,
	},
}

func InlayHint(ctx context.Context, snapshot Snapshot, fh FileHandle, pRng protocol.Range) ([]protocol.InlayHint, error) {
	ctx, done := event.Start(ctx, "source.InlayHint")
	defer done()

	pkg, pgf, err := GetParsedFile(ctx, snapshot, fh, NarrowestPackage)
	if err != nil {
		return nil, fmt.Errorf("getting file for InlayHint: %w", err)
	}

	// Collect a list of the inlay hints that are enabled.
	inlayHintOptions := snapshot.View().Options().InlayHintOptions
	var enabledHints []InlayHintFunc
	for hint, enabled := range inlayHintOptions.Hints {
		if !enabled {
			continue
		}
		if h, ok := AllInlayHints[hint]; ok {
			enabledHints = append(enabledHints, h.Run)
		}
	}
	if len(enabledHints) == 0 {
		return nil, nil
	}

	tmap := lsppos.NewTokenMapper(pgf.Src, pgf.Tok)
	info := pkg.GetTypesInfo()
	q := Qualifier(pgf.File, pkg.GetTypes(), info)

	// Set the range to the full file if the range is not valid.
	start, end := pgf.File.Pos(), pgf.File.End()
	if pRng.Start.Line < pRng.End.Line || pRng.Start.Character < pRng.End.Character {
		// Adjust start and end for the specified range.
		rng, err := pgf.Mapper.RangeToSpanRange(pRng)
		if err != nil {
			return nil, err
		}
		start, end = rng.Start, rng.End
	}

	var hints []protocol.InlayHint
	ast.Inspect(pgf.File, func(node ast.Node) bool {
		// If not in range, we can stop looking.
		if node == nil || node.End() < start || node.Pos() > end {
			return false
		}
		for _, fn := range enabledHints {
			hints = append(hints, fn(node, tmap, info, &q)...)
		}
		return true
	})
	return hints, nil
}

func parameterNames(node ast.Node, tmap *lsppos.TokenMapper, info *types.Info, _ *types.Qualifier) []protocol.InlayHint {
	callExpr, ok := node.(*ast.CallExpr)
	if !ok {
		return nil
	}
	signature, ok := info.TypeOf(callExpr.Fun).(*types.Signature)
	if !ok {
		return nil
	}

	var hints []protocol.InlayHint
	for i, v := range callExpr.Args {
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

func funcTypeParams(node ast.Node, tmap *lsppos.TokenMapper, info *types.Info, _ *types.Qualifier) []protocol.InlayHint {
	ce, ok := node.(*ast.CallExpr)
	if !ok {
		return nil
	}
	id, ok := ce.Fun.(*ast.Ident)
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

func assignVariableTypes(node ast.Node, tmap *lsppos.TokenMapper, info *types.Info, q *types.Qualifier) []protocol.InlayHint {
	stmt, ok := node.(*ast.AssignStmt)
	if !ok || stmt.Tok != token.DEFINE {
		return nil
	}

	var hints []protocol.InlayHint
	for _, v := range stmt.Lhs {
		if h := variableType(v, tmap, info, q); h != nil {
			hints = append(hints, *h)
		}
	}
	return hints
}

func rangeVariableTypes(node ast.Node, tmap *lsppos.TokenMapper, info *types.Info, q *types.Qualifier) []protocol.InlayHint {
	rStmt, ok := node.(*ast.RangeStmt)
	if !ok {
		return nil
	}
	var hints []protocol.InlayHint
	if h := variableType(rStmt.Key, tmap, info, q); h != nil {
		hints = append(hints, *h)
	}
	if h := variableType(rStmt.Value, tmap, info, q); h != nil {
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

func constantValues(node ast.Node, tmap *lsppos.TokenMapper, info *types.Info, _ *types.Qualifier) []protocol.InlayHint {
	genDecl, ok := node.(*ast.GenDecl)
	if !ok || genDecl.Tok != token.CONST {
		return nil
	}

	var hints []protocol.InlayHint
	for _, v := range genDecl.Specs {
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

func compositeLiteralFields(node ast.Node, tmap *lsppos.TokenMapper, info *types.Info, q *types.Qualifier) []protocol.InlayHint {
	compLit, ok := node.(*ast.CompositeLit)
	if !ok {
		return nil
	}
	typ := info.TypeOf(compLit)
	if typ == nil {
		return nil
	}
	if t, ok := typ.(*types.Pointer); ok {
		typ = t.Elem()
	}
	strct, ok := typ.Underlying().(*types.Struct)
	if !ok {
		return nil
	}

	var hints []protocol.InlayHint
	var allEdits []protocol.TextEdit
	for i, v := range compLit.Elts {
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
			allEdits = append(allEdits, protocol.TextEdit{
				Range:   protocol.Range{Start: start, End: start},
				NewText: strct.Field(i).Name() + ": ",
			})
		}
	}
	// It is not allowed to have a mix of keyed and unkeyed fields, so
	// have the text edits add keys to all fields.
	for i := range hints {
		hints[i].TextEdits = allEdits
	}
	return hints
}

func compositeLiteralTypes(node ast.Node, tmap *lsppos.TokenMapper, info *types.Info, q *types.Qualifier) []protocol.InlayHint {
	compLit, ok := node.(*ast.CompositeLit)
	if !ok {
		return nil
	}
	typ := info.TypeOf(compLit)
	if typ == nil {
		return nil
	}
	if compLit.Type != nil {
		return nil
	}
	prefix := ""
	if t, ok := typ.(*types.Pointer); ok {
		typ = t.Elem()
		prefix = "&"
	}
	// The type for this composite literal is implicit, add an inlay hint.
	start, ok := tmap.Position(compLit.Lbrace)
	if !ok {
		return nil
	}
	return []protocol.InlayHint{{
		Position: &start,
		Label:    buildLabel(fmt.Sprintf("%s%s", prefix, types.TypeString(typ, *q))),
		Kind:     protocol.Type,
	}}
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
