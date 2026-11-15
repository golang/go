// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"fmt"
	"go/token"
	"strings"

	"cmd/compile/internal/base"
	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
)

func (fl *fileLower) lowerLambdas() {
	var lams []*syntax.LambdaExpr
	syntax.Inspect(fl.file, func(n syntax.Node) bool {
		if lam, ok := n.(*syntax.LambdaExpr); ok {
			lams = append(lams, lam)
		}
		return true
	})
	for i := len(lams) - 1; i >= 0; i-- {
		if lit := fl.lowerLambdaExpr(lams[i]); lit != nil {
			fl.stageReplacement(lams[i], lit)
		}
	}
	fl.flushInnerFirst()
	fl.flushTypeImports()
}

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
		case token.COMMA, token.SEMICOLON, token.COLON:
			if depth == 0 {
				return false
			}
		case token.RPAREN, token.RBRACK, token.RBRACE:
			if depth == 0 {
				return false
			}
			depth--
		}
		end = t.end
		return true
	})
	return end
}

func (fl *fileLower) lowerLambdaExpr(lam *syntax.LambdaExpr) *syntax.FuncLit {
	sig, ok := fl.pkg.info.TypeOf(lam).(*types2.Signature)
	if !ok {
		return nil
	}
	pos := lam.Pos()
	var params []otext
	for i, name := range lam.Params {
		if i > 0 {
			params = append(params, gen(", "))
		}
		spelled, ok := fl.spellLowered(sig.Params().At(i).Type())
		if !ok {
			spelled, ok = fl.spellType(sig.Params().At(i).Type())
			if !ok {
				return nil
			}
		}
		params = append(params, fl.srcUser(fl.off(name.Pos()), fl.off(name.Pos())+len(name.Value)), gen(" "+spelled))
	}
	resultsText := fl.lambdaResultSpell(pos, sig)
	if resultsText == "" && sig.Results().Len() > 0 {
		return nil
	}
	var hoisted string
	if block, ok := lam.Body.(*syntax.BlockStmt); ok {
		var ok2 bool
		resultsText, hoisted, ok2 = fl.lowerLambdaFrame(pos, sig, block, cat(params...).s, resultsText)
		if !ok2 {
			return nil
		}
	}
	body, bodyText := fl.lambdaBody(lam, sig, pos)
	fullText := cat(gen("func("), cat(params...), gen(")"+resultsText+" {"+hoisted+" "), bodyText, gen(" }"))
	ft, ok := typeExprAt("func("+cat(params...).s+")"+resultsText, pos)
	if !ok {
		return nil
	}
	functype, ok := ft.(*syntax.FuncType)
	if !ok {
		return nil
	}
	lit := &syntax.FuncLit{Type: functype, Body: body}
	lit.SetPos(pos)
	fl.addT(fl.off(syntax.StartPos(lam)), fl.lambdaEnd(lam), fullText)
	return lit
}

func (fl *fileLower) lambdaEnd(lam *syntax.LambdaExpr) int {
	switch body := lam.Body.(type) {
	case *syntax.BlockStmt:
		return fl.off(body.Rbrace) + 1
	case syntax.Expr:
		return fl.lambdaCaptureSpan(lam, body).end
	}
	return fl.off(syntax.EndPos(lam))
}

func (fl *fileLower) lambdaResultSpell(pos syntax.Pos, sig *types2.Signature) string {
	n := sig.Results().Len()
	if n == 0 {
		return ""
	}
	texts := make([]string, n)
	for i := range texts {
		spelled, ok := fl.spellLowered(sig.Results().At(i).Type())
		if !ok {
			spelled, ok = fl.spellType(sig.Results().At(i).Type())
			if !ok {
				return ""
			}
		}
		texts[i] = spelled
	}
	if n == 1 {
		return " " + texts[0]
	}
	return " (" + strings.Join(texts, ", ") + ")"
}

func (fl *fileLower) callBodySpan(call *syntax.CallExpr) span {
	split, ok := splitCall(fl.src, fl.off(call.Pos()))
	if !ok {
		base.Fatalf("logos: call at %s has no closing parenthesis", call.Pos().String())
	}
	return span{fl.off(leftmostOf(call)), split.end}
}

func (fl *fileLower) lambdaCaptureSpan(lam *syntax.LambdaExpr, body syntax.Expr) span {
	if stashed, ok := fl.optBodySpan[lam]; ok && stashed.end > stashed.start {
		switch syntax.Unparen(body).(type) {
		case *syntax.FuncLit, *syntax.LambdaExpr:
		default:
			return stashed
		}
	}
	start := fl.exprStart(body)
	return span{start, lambdaBodyEnd(fl.src, start)}
}

func (fl *fileLower) lambdaBody(lam *syntax.LambdaExpr, sig *types2.Signature, pos syntax.Pos) (*syntax.BlockStmt, otext) {
	fl.preLowerCaptureBody(lam)
	if block, ok := lam.Body.(*syntax.BlockStmt); ok {
		text := fl.textOf(span{fl.off(syntax.StartPos(block)) + 1, fl.off(block.Rbrace)})
		return block, text
	}
	expr, ok := lam.Body.(syntax.Expr)
	if !ok {
		return blockOf(pos, nil), otext{}
	}
	if t, ok := syntax.Unparen(expr).(*syntax.TryExpr); ok {
		return fl.lambdaTryBody(sig, pos, t)
	}
	espan := fl.lambdaCaptureSpan(lam, expr)
	text := fl.drainBodyWrapper(expr, fl.lazyText(espan))
	if native, done := fl.stagedReplacements[expr]; done {
		expr = native
	}
	return blockOf(pos, []syntax.Stmt{returnOf(pos, []syntax.Expr{expr})}), cat(gen("return "), text)
}

func (fl *fileLower) lambdaTryBody(sig *types2.Signature, pos syntax.Pos, t *syntax.TryExpr) (*syntax.BlockStmt, otext) {
	call, ok := syntax.Unparen(t.X).(*syntax.CallExpr)
	if !ok {
		text := fl.textOf(span{fl.off(syntax.StartPos(t)), fl.off(syntax.EndPos(t))})
		return blockOf(pos, []syntax.Stmt{returnOf(pos, []syntax.Expr{t})}), cat(gen("return "), text)
	}
	nres := sig.Results().Len()
	var names []string
	for i := range nres - 1 {
		names = append(names, fmt.Sprintf("_logos_lam%d_%d_%d", pos.Line(), pos.Col(), i))
	}
	errName := fmt.Sprintf("_logos_lam%d_%d_e", pos.Line(), pos.Col())
	callText := fl.textOf(fl.callBodySpan(call))
	var parts []otext
	var lhs []syntax.Expr
	var lhsTexts []string
	for _, nm := range names {
		lhs = append(lhs, nameOf(pos, nm))
		lhsTexts = append(lhsTexts, nm)
	}
	lhs = append(lhs, nameOf(pos, errName))
	lhsTexts = append(lhsTexts, errName)
	parts = append(parts, cat(gen(strings.Join(lhsTexts, ", ")+" := "), callText, gen("; ")))
	as := defineOf(pos, lhs, []syntax.Expr{call})
	var zeros []syntax.Expr
	var zeroTexts []string
	for i := range nres - 1 {
		zt := fl.zeroValue(sig.Results().At(i).Type(), pos)
		zeros = append(zeros, zt)
		zeroTexts = append(zeroTexts, renderExpr(zt))
	}
	zeros = append(zeros, nameOf(pos, errName))
	zeroTexts = append(zeroTexts, errName)
	branch := ifOf(pos, nil, notNil(pos, nameOf(pos, errName)), []syntax.Stmt{returnOf(pos, zeros)})
	parts = append(parts, gen("if "+errName+" != nil { return "+strings.Join(zeroTexts, ", ")+" }; "))
	var retVals []syntax.Expr
	var retTexts []string
	for _, nm := range names {
		retVals = append(retVals, nameOf(pos, nm))
		retTexts = append(retTexts, nm)
	}
	ret := returnOf(pos, retVals)
	parts = append(parts, gen("return"))
	if len(retTexts) > 0 {
		parts = append(parts, gen(" "+strings.Join(retTexts, ", ")))
	}
	return blockOf(pos, []syntax.Stmt{as, branch, ret}), cat(parts...)
}

func (fl *fileLower) lowerLambdaFrame(pos syntax.Pos, sig *types2.Signature, block *syntax.BlockStmt, paramText, resultsText string) (string, string, bool) {
	deferTry, errdefer, scopeDefer := cleanupsIn(block)
	if (deferTry || errdefer || scopeDefer) && sig.Results().Len() > 0 {
		named, ok := fl.lambdaNamedResults(pos, sig)
		if !ok {
			return "", "", false
		}
		resultsText = named
	}
	ft, ok := typeExprAt("func("+paramText+")"+resultsText, pos)
	if !ok {
		return "", "", false
	}
	functype, ok := ft.(*syntax.FuncType)
	if !ok {
		return "", "", false
	}
	fc := newFuncLower(fl, functype, sig, block)
	fc.lower()
	return resultsText, fc.prependHoists(), true
}

func (fl *fileLower) lambdaNamedResults(pos syntax.Pos, sig *types2.Signature) (string, bool) {
	n := sig.Results().Len()
	texts := make([]string, n)
	for i := range texts {
		spelled, ok := fl.spellLowered(sig.Results().At(i).Type())
		if !ok {
			spelled, ok = fl.spellType(sig.Results().At(i).Type())
			if !ok {
				return "", false
			}
		}
		texts[i] = fmt.Sprintf("_logos_lr%d %s", i, spelled)
	}
	return " (" + strings.Join(texts, ", ") + ")", true
}
