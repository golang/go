// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"fmt"
	"go/token"
	"strings"

	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
)

// This file lowers the M6-S6 lambda nodes (SPEC Sec 1.4, T-104.6): a
// LambdaExpr becomes a function literal with parameter types from the
// recorded target signature. An expression body returns its value; a
// `?` body expands to an explicit error branch (the statement lowering
// already ran, so no TryExpr may remain); a block body carries over.
// Native tree and eject text come from one function-type spelling.

// lowerLambdas rewrites every lambda of one file, innermost first: an
// outer body copies its source through textOf, which folds the splices
// an inner lambda already recorded instead of overlapping them.
func (fl *fileLower) lowerLambdas() {
	var lams []*syntax.LambdaExpr
	syntax.Inspect(fl.file, func(n syntax.Node) bool {
		if lam, ok := n.(*syntax.LambdaExpr); ok {
			lams = append(lams, lam)
		}
		return true
	})
	for i := len(lams) - 1; i >= 0; i-- {
		fl.lowerLambdaExpr(lams[i])
	}
	fl.flushTypeImports()
}

// lambdaBodyEnd returns the end of the expression starting at from: the
// end of its last token. Brackets nest, and a comma, semicolon, or
// closing bracket at the body's own depth ends it, as does an automatic
// semicolon there. The node end cannot serve: EndPos of a call sits at
// its function and EndPos of a composite literal can miss its closing
// brace, so the splice end comes from the source tokens instead.
func lambdaBodyEnd(src []byte, from int) int {
	end := from
	depth := 0
	scanSegment(src, from, len(src), func(t tok) bool {
		if t.comment {
			return true
		}
		if t.auto {
			return depth > 0
		}
		switch t.kind {
		case token.LPAREN, token.LBRACK, token.LBRACE:
			depth++
		case token.RPAREN, token.RBRACK, token.RBRACE, token.COMMA, token.SEMICOLON:
			if depth == 0 {
				return false
			}
			if t.kind != token.COMMA && t.kind != token.SEMICOLON {
				depth--
			}
		}
		end = t.end
		return true
	})
	return end
}

// lowerLambdaExpr rewrites one lambda to a function literal.
func (fl *fileLower) lowerLambdaExpr(lam *syntax.LambdaExpr) {
	sig, ok := fl.pkg.info.TypeOf(lam).(*types2.Signature)
	if !ok {
		return
	}
	pos := lam.Pos()
	var params strings.Builder
	for i, name := range lam.Params {
		if i > 0 {
			params.WriteString(", ")
		}
		spelled, ok := fl.spellLowered(sig.Params().At(i).Type())
		if !ok {
			spelled, ok = fl.spellType(sig.Params().At(i).Type())
			if !ok {
				return
			}
		}
		params.WriteString(name.Value + " " + spelled)
	}
	results, resultsText := fl.lambdaResultSpell(pos, sig)
	if resultsText == "" && sig.Results().Len() > 0 {
		return
	}
	_ = results
	body, bodyText := fl.lambdaBody(lam, sig, pos)
	fullText := "func(" + params.String() + ")" + resultsText + " { " + bodyText + " }"
	ft, ok := typeExprAt("func("+params.String()+")"+resultsText, pos)
	if !ok {
		return
	}
	functype, ok := ft.(*syntax.FuncType)
	if !ok {
		return
	}
	lit := &syntax.FuncLit{Type: functype, Body: body}
	lit.SetPos(pos)
	end := fl.off(syntax.EndPos(lam))
	if block, ok := lam.Body.(*syntax.BlockStmt); ok {
		end = fl.off(block.Rbrace) + 1
	} else if expr, ok := lam.Body.(syntax.Expr); ok {
		end = lambdaBodyEnd(fl.src, fl.off(syntax.StartPos(expr)))
	}
	fl.add(fl.off(syntax.StartPos(lam)), end, fullText)
	replaceTupleExpr(fl.file, lam, lit)
}

// lambdaResultSpell spells the result list and builds zero values.
func (fl *fileLower) lambdaResultSpell(pos syntax.Pos, sig *types2.Signature) ([]syntax.Expr, string) {
	n := sig.Results().Len()
	if n == 0 {
		return nil, ""
	}
	out := make([]syntax.Expr, n)
	texts := make([]string, n)
	for i := range out {
		spelled, ok := fl.spellLowered(sig.Results().At(i).Type())
		if !ok {
			spelled, ok = fl.spellType(sig.Results().At(i).Type())
			if !ok {
				return nil, ""
			}
		}
		texts[i] = spelled
		tn, ok := typeExprAt(spelled, pos)
		if !ok {
			return nil, ""
		}
		out[i] = tn
	}
	if n == 1 {
		return out, " " + texts[0]
	}
	return out, " (" + strings.Join(texts, ", ") + ")"
}

// lambdaBody builds the function body and its text for a lambda.
func (fl *fileLower) lambdaBody(lam *syntax.LambdaExpr, sig *types2.Signature, pos syntax.Pos) (*syntax.BlockStmt, string) {
	if block, ok := lam.Body.(*syntax.BlockStmt); ok {
		text := fl.textOf(span{fl.off(syntax.StartPos(block)) + 1, fl.off(block.Rbrace)})
		return block, text
	}
	expr, ok := lam.Body.(syntax.Expr)
	if !ok {
		return blockOf(pos, nil), ""
	}
	if t, ok := syntax.Unparen(expr).(*syntax.TryExpr); ok {
		return fl.lambdaTryBody(sig, pos, t)
	}
	start := fl.off(syntax.StartPos(expr))
	text := fl.textOf(span{start, lambdaBodyEnd(fl.src, start)})
	return blockOf(pos, []syntax.Stmt{returnOf(pos, []syntax.Expr{expr})}), "return " + text
}

// lambdaTryBody expands a `?` lambda body to an explicit error branch:
// `r, e := call(); if e != nil { return zeros, e }; return r`.
func (fl *fileLower) lambdaTryBody(sig *types2.Signature, pos syntax.Pos, t *syntax.TryExpr) (*syntax.BlockStmt, string) {
	call, ok := syntax.Unparen(t.X).(*syntax.CallExpr)
	if !ok {
		text := fl.textOf(span{fl.off(syntax.StartPos(t)), fl.off(syntax.EndPos(t))})
		return blockOf(pos, []syntax.Stmt{returnOf(pos, []syntax.Expr{t})}), "return " + text
	}
	nres := sig.Results().Len()
	var names []string
	for i := range nres - 1 {
		names = append(names, fmt.Sprintf("_logos_lam%d_%d_%d", pos.Line(), pos.Col(), i))
	}
	errName := fmt.Sprintf("_logos_lam%d_%d_e", pos.Line(), pos.Col())
	callText := fl.textOf(fl.tupleCallSpan(call))
	var text strings.Builder
	var lhs []syntax.Expr
	var lhsTexts []string
	for _, nm := range names {
		lhs = append(lhs, nameOf(pos, nm))
		lhsTexts = append(lhsTexts, nm)
	}
	lhs = append(lhs, nameOf(pos, errName))
	lhsTexts = append(lhsTexts, errName)
	text.WriteString(strings.Join(lhsTexts, ", ") + " := " + callText + "; ")
	as := defineOf(pos, lhs, []syntax.Expr{call})
	var zeros []syntax.Expr
	var zeroTexts []string
	for i := range nres - 1 {
		zt := fl.lambdaZero(sig.Results().At(i).Type(), pos)
		zeros = append(zeros, zt)
		zeroTexts = append(zeroTexts, renderExpr(zt))
	}
	zeros = append(zeros, nameOf(pos, errName))
	zeroTexts = append(zeroTexts, errName)
	branch := ifOf(pos, nil, notNil(pos, nameOf(pos, errName)), []syntax.Stmt{returnOf(pos, zeros)})
	text.WriteString("if " + errName + " != nil { return " + strings.Join(zeroTexts, ", ") + " }; ")
	var retVals []syntax.Expr
	var retTexts []string
	for _, nm := range names {
		retVals = append(retVals, nameOf(pos, nm))
		retTexts = append(retTexts, nm)
	}
	ret := returnOf(pos, retVals)
	text.WriteString("return")
	if len(retTexts) > 0 {
		text.WriteString(" " + strings.Join(retTexts, ", "))
	}
	return blockOf(pos, []syntax.Stmt{as, branch, ret}), text.String()
}

// lambdaZero builds the zero value of a result type.
func (fl *fileLower) lambdaZero(typ types2.Type, pos syntax.Pos) syntax.Expr {
	if b, ok := typ.(*types2.Basic); ok {
		return zeroBasic(b, pos)
	}
	switch types2.Unalias(typ).(type) {
	case *types2.Pointer, *types2.Slice, *types2.Map, *types2.Chan, *types2.Signature, *types2.Interface:
		return nilOf(pos)
	default:
		spelled, ok := fl.spellLowered(typ)
		if !ok {
			spelled, _ = fl.spellType(typ)
		}
		tn, ok := typeExprAt(spelled, pos)
		if !ok {
			return nilOf(pos)
		}
		return at(&syntax.CompositeLit{Type: tn, Rbrace: pos}, pos)
	}
}
