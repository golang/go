// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"fmt"
	"reflect"
	"slices"
	"strconv"

	"cmd/compile/internal/base"
	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
)

const (
	interpConvPkg   = "_logos_strconv"
	interpUnsafePkg = "_logos_unsafe"
	interpFmtPkg    = "_logos_fmt"
	interpBuf       = "_logos_b"
	interpVal       = "_logos_v%d"
	interpStr       = "_logos_s%d"
)

const (
	interpIntBound   = 20
	interpFloatBound = 24
	interpBoolBound  = 5
)

func InterpFixedBytes(parts []string, kinds []types2.LogosInterpKind) int {
	n := 0
	for _, part := range parts {
		n += len(part)
	}
	for _, k := range kinds {
		n += interpFixedWidth(k)
	}
	return n
}

func interpFixedWidth(k types2.LogosInterpKind) int {
	switch k {
	case types2.LogosInterpInt, types2.LogosInterpUint:
		return interpIntBound
	case types2.LogosInterpFloat32, types2.LogosInterpFloat64:
		return interpFloatBound
	case types2.LogosInterpBool:
		return interpBoolBound
	}
	return 0
}

var interpKinds = make(map[*syntax.InterpExpr][]types2.LogosInterpKind)

func (fl *fileLower) collectInterpKinds() {
	syntax.Inspect(fl.file, func(n syntax.Node) bool {
		e, ok := n.(*syntax.InterpExpr)
		if !ok || len(e.Exprs) == 0 {
			return true
		}
		if _, done := interpKinds[e]; done {
			return true
		}
		kinds := make([]types2.LogosInterpKind, len(e.Exprs))
		for i, op := range e.Exprs {
			t := fl.pkg.info.TypeOf(op)
			k, ok := types2.LogosInterpKindOf(t)
			if !ok {
				if base.Errors() > 0 {
					return true
				}
				base.Fatalf("logos: interpolation operand at %s has unexpected type %s", op.Pos().String(), t)
			}
			kinds[i] = k
		}
		interpKinds[e] = kinds
		return true
	})
}

func interpLitsUnder(root syntax.Node) []*syntax.InterpExpr {
	var lits []*syntax.InterpExpr
	syntax.Inspect(root, func(n syntax.Node) bool {
		if e, ok := n.(*syntax.InterpExpr); ok {
			lits = append(lits, e)
		}
		return true
	})
	slices.SortFunc(lits, func(a, b *syntax.InterpExpr) int {
		return len(a.Raw) - len(b.Raw)
	})
	return lits
}

func (fl *fileLower) lowerInterp() {
	needConv, needUnsafe := fl.interpNeedConv, fl.interpNeedUnsafe
	for _, e := range interpLitsUnder(fl.file) {
		kinds := interpKinds[e]
		delete(interpKinds, e)
		if kinds == nil && len(e.Exprs) > 0 && base.Errors() > 0 {
			continue
		}
		need, _ := fl.lowerInterpExpr(e, kinds)
		needConv = needConv || need
		if len(e.Exprs) > 0 {
			needUnsafe = true
		}
	}
	fl.flushInnerFirst()
	fl.flushInterpImports(needConv, needUnsafe)
}

func replaceFieldExprs(n syntax.Node, interfaceField func(reflect.Type) bool, repl func(syntax.Expr) syntax.Expr) {
	v := reflect.ValueOf(n)
	if v.Kind() != reflect.Pointer || v.IsNil() {
		return
	}
	v = v.Elem()
	if v.Kind() != reflect.Struct {
		return
	}
	typ := v.Type()
	for i := range v.NumField() {
		if typ.Field(i).IsExported() {
			replaceFieldExpr(v.Field(i), interfaceField, repl)
		}
	}
}

func replaceFieldExpr(f reflect.Value, interfaceField func(reflect.Type) bool, repl func(syntax.Expr) syntax.Expr) {
	switch f.Kind() {
	case reflect.Interface:
		if f.IsNil() || !f.CanSet() || !interfaceField(f.Type()) {
			return
		}
		e, ok := f.Elem().Interface().(syntax.Expr)
		if !ok {
			return
		}
		f.Set(reflect.ValueOf(repl(e)))
	case reflect.Slice, reflect.Array:
		for i := range f.Len() {
			replaceFieldExpr(f.Index(i), interfaceField, repl)
		}
	}
}

func anyInterfaceFields(t reflect.Type) bool {
	return true
}

func (fl *fileLower) interpHelper(op syntax.Expr, kind types2.LogosInterpKind) string {
	helper := "InterpError"
	if kind == types2.LogosInterpStringer {
		helper = "InterpStringer"
	}
	t := fl.pkg.info.TypeOf(op)
	if t == nil {
		return helper
	}
	if c := constAsView(t); c != nil {
		t = c.Elem
	}
	if _, ok := types2.Unalias(t).(*types2.Interface); ok {
		return helper
	}
	if n, ok := types2.Unalias(t).(*types2.Named); ok {
		if _, ok := n.Underlying().(*types2.Interface); ok {
			return helper
		}
	}
	return helper + "Value"
}

func (fl *fileLower) lowerInterpExpr(e *syntax.InterpExpr, kinds []types2.LogosInterpKind) (bool, otext) {
	pos := e.Pos()
	start := fl.off(pos)
	end := start + len(e.Raw)
	if len(e.Exprs) == 0 {
		text := fmt.Sprintf("%q", e.Parts[0])
		fl.add(start, end, text)
		fl.stageReplacement(e, litOf(pos, text, syntax.StringLit))
		return false, gen(text)
	}
	if len(kinds) != len(e.Exprs) || len(e.Opens) != len(e.Exprs) || len(e.Closes) != len(e.Exprs) {
		base.Fatalf("logos: interpolation at %s has %d operands but %d kinds, %d opens, %d closes",
			pos.String(), len(e.Exprs), len(kinds), len(e.Opens), len(e.Closes))
	}
	opTexts := make([]otext, len(e.Exprs))
	for i := range e.Exprs {
		opTexts[i] = fl.interpOperandText(e.Opens[i], e.Closes[i])
	}
	needConv := false
	for _, k := range kinds {
		switch k {
		case types2.LogosInterpInt, types2.LogosInterpUint,
			types2.LogosInterpFloat32, types2.LogosInterpFloat64,
			types2.LogosInterpBool:
			needConv = true
		}
	}
	valName := func(i int) string { return fmt.Sprintf(interpVal, i) }
	strName := func(i int) string { return fmt.Sprintf(interpStr, i) }
	var stmts []syntax.Stmt
	var lines []otext
	for i := range e.Exprs {
		op := e.Exprs[i]
		if native, ok := fl.stagedReplacements[op]; ok {
			op = native
		}
		v := valName(i)
		stmts = append(stmts, defineOf(pos, []syntax.Expr{nameOf(pos, v)}, []syntax.Expr{op}))
		lines = append(lines, cat(gen(v+" := "), opTexts[i]))
		if kinds[i] == types2.LogosInterpError || kinds[i] == types2.LogosInterpStringer {
			helper := fl.interpHelper(op, kinds[i])
			needConv = true
			s := strName(i)
			fl.interpNeedFmt = true
			format := selOf(pos, nameOf(pos, interpFmtPkg), "Sprint")
			stmts = append(stmts, defineOf(pos,
				[]syntax.Expr{nameOf(pos, s)},
				[]syntax.Expr{callOf(pos, selOf(pos, nameOf(pos, rtAlias), helper), []syntax.Expr{nameOf(pos, v), format})}))
			lines = append(lines, gen(s+" := "+rtAlias+"."+helper+"("+v+", "+interpFmtPkg+".Sprint)"))
		}
	}
	static := 0
	for _, part := range e.Parts {
		static += len(part)
	}
	boundText := strconv.Itoa(static)
	var bound syntax.Expr = litOf(pos, boundText, syntax.IntLit)
	for i, k := range kinds {
		var termText string
		var term syntax.Expr
		switch k {
		case types2.LogosInterpString, types2.LogosInterpBytes,
			types2.LogosInterpError, types2.LogosInterpStringer:
			ref := valName(i)
			if k == types2.LogosInterpError || k == types2.LogosInterpStringer {
				ref = strName(i)
			}
			termText = "len(" + ref + ")"
			term = callOf(pos, nameOf(pos, "len"), []syntax.Expr{nameOf(pos, ref)})
		case types2.LogosInterpInt, types2.LogosInterpUint:
			termText = strconv.Itoa(interpIntBound)
			term = litOf(pos, termText, syntax.IntLit)
		case types2.LogosInterpFloat32, types2.LogosInterpFloat64:
			termText = strconv.Itoa(interpFloatBound)
			term = litOf(pos, termText, syntax.IntLit)
		case types2.LogosInterpBool:
			termText = strconv.Itoa(interpBoolBound)
			term = litOf(pos, termText, syntax.IntLit)
		}
		boundText += "+" + termText
		bound = at(&syntax.Operation{Op: syntax.Add, X: bound, Y: term}, pos)
	}
	stmts = append(stmts, defineOf(pos, []syntax.Expr{nameOf(pos, interpBuf)}, []syntax.Expr{
		callOf(pos, nameOf(pos, "make"), []syntax.Expr{
			at(&syntax.SliceType{Elem: nameOf(pos, "byte")}, pos),
			litOf(pos, "0", syntax.IntLit),
			bound,
		}),
	}))
	lines = append(lines, gen(interpBuf+" := make([]byte, 0, "+boundText+")"))
	emitAppend := func(arg syntax.Expr, argText otext, spread bool) {
		call := callOf(pos, nameOf(pos, "append"), []syntax.Expr{nameOf(pos, interpBuf), arg})
		call.HasDots = spread
		stmts = append(stmts, assignOf(pos, []syntax.Expr{nameOf(pos, interpBuf)}, []syntax.Expr{call}))
		if spread {
			argText = cat(argText, gen("..."))
		}
		lines = append(lines, cat(gen(interpBuf+" = append("+interpBuf+", "), argText, gen(")")))
	}
	emitConv := func(name string, args []syntax.Expr, argTexts []otext) {
		call := callOf(pos, selOf(pos, nameOf(pos, rtAlias), name), args)
		stmts = append(stmts, assignOf(pos, []syntax.Expr{nameOf(pos, interpBuf)}, []syntax.Expr{call}))
		cparts := make([]otext, 0, 2*len(argTexts)+1)
		cparts = append(cparts, gen(interpBuf+" = "+rtAlias+"."+name+"("))
		for i, a := range argTexts {
			if i > 0 {
				cparts = append(cparts, gen(", "))
			}
			cparts = append(cparts, a)
		}
		cparts = append(cparts, gen(")"))
		lines = append(lines, cat(cparts...))
	}
	conv1 := func(name, conv, v, last string) {
		emitConv(name,
			[]syntax.Expr{nameOf(pos, interpBuf), callOf(pos, nameOf(pos, conv), []syntax.Expr{nameOf(pos, v)}), litOf(pos, last, syntax.IntLit)},
			[]otext{gen(interpBuf), gen(conv + "(" + v + ")"), gen(last)})
	}
	for i, part := range e.Parts {
		if part != "" {
			quoted := fmt.Sprintf("%q", part)
			emitAppend(litOf(pos, quoted, syntax.StringLit), gen(quoted), true)
		}
		if i >= len(e.Exprs) {
			continue
		}
		v := valName(i)
		switch kinds[i] {
		case types2.LogosInterpString, types2.LogosInterpBytes:
			emitAppend(nameOf(pos, v), gen(v), true)
		case types2.LogosInterpError, types2.LogosInterpStringer:
			s := strName(i)
			emitAppend(nameOf(pos, s), gen(s), true)
		case types2.LogosInterpInt:
			conv1("AppendInt", "int64", v, "10")
		case types2.LogosInterpUint:
			conv1("AppendUint", "uint64", v, "10")
		case types2.LogosInterpFloat32, types2.LogosInterpFloat64:
			bits := "64"
			if kinds[i] == types2.LogosInterpFloat32 {
				bits = "32"
			}
			emitConv("AppendFloat",
				[]syntax.Expr{
					nameOf(pos, interpBuf),
					callOf(pos, nameOf(pos, "float64"), []syntax.Expr{nameOf(pos, v)}),
					litOf(pos, "'g'", syntax.RuneLit),
					at(&syntax.Operation{Op: syntax.Sub, X: litOf(pos, "1", syntax.IntLit)}, pos),
					litOf(pos, bits, syntax.IntLit),
				},
				[]otext{gen(interpBuf), gen("float64(" + v + ")"), gen("'g'"), gen("-1"), gen(bits)})
		case types2.LogosInterpBool:
			emitConv("AppendBool",
				[]syntax.Expr{nameOf(pos, interpBuf), nameOf(pos, v)},
				[]otext{gen(interpBuf), gen(v)})
		}
	}
	lines = append(lines, gen("return "+interpUnsafePkg+".String("+interpUnsafePkg+".SliceData("+interpBuf+"), len("+interpBuf+"))"))
	stmts = append(stmts, returnOf(pos, []syntax.Expr{
		callOf(pos, selOf(pos, nameOf(pos, interpUnsafePkg), "String"), []syntax.Expr{
			callOf(pos, selOf(pos, nameOf(pos, interpUnsafePkg), "SliceData"), []syntax.Expr{nameOf(pos, interpBuf)}),
			callOf(pos, nameOf(pos, "len"), []syntax.Expr{nameOf(pos, interpBuf)}),
		}),
	}))
	signature := at(&syntax.FuncType{
		ParamList:  []*syntax.Field{},
		ResultList: []*syntax.Field{at(&syntax.Field{Type: nameOf(pos, "string")}, pos)},
	}, pos)
	lit := at(&syntax.FuncLit{Type: signature, Body: blockOf(pos, stmts)}, pos)
	native := callOf(pos, lit, nil)
	fl.stageReplacement(e, native)
	if fl.optBodyType == nil {
		fl.optBodyType = make(map[syntax.Expr]types2.Type)
	}
	fl.optBodyType[native] = types2.Typ[types2.String]
	parts := make([]otext, 0, 2*len(lines)+2)
	parts = append(parts, gen("func() string {\n"))
	for i, line := range lines {
		if i > 0 {
			parts = append(parts, gen("\n"))
		}
		parts = append(parts, line)
	}
	parts = append(parts, gen("\n}()"))
	text := cat(parts...)
	fl.addT(start, end, text)
	return needConv, text
}

func (fl *fileLower) interpOperandText(open, close syntax.Pos) otext {
	text := fl.textOf(span{fl.off(open), fl.off(close) + 1})
	return text.slice(1, len(text.s)-1)
}

func (fl *fileLower) flushInterpImports(needConv, needUnsafe bool) {
	if !needConv && !needUnsafe {
		return
	}
	pos := fl.file.Pos()
	var decls []syntax.Decl
	var text string
	if needConv && !fl.needRT {
		decls = append(decls, rtImport(pos))
		if !fl.needRTText {
			text += "\n\nimport " + rtAlias + " " + fmt.Sprintf("%q", rtImportPath)
		}
		fl.needRT = true
	}
	if needUnsafe {
		decls = append(decls, at(&syntax.ImportDecl{
			LocalPkgName: nameOf(pos, interpUnsafePkg),
			Path:         litOf(pos, `"unsafe"`, syntax.StringLit),
		}, pos))
		text += "\nimport " + interpUnsafePkg + ` "unsafe"`
	}
	if fl.interpNeedFmt {
		decls = append(decls, at(&syntax.ImportDecl{
			LocalPkgName: nameOf(pos, interpFmtPkg),
			Path:         litOf(pos, `"fmt"`, syntax.StringLit),
		}, pos))
		text += "\nimport " + interpFmtPkg + ` "fmt"`
	}
	fl.file.DeclList = append(decls, fl.file.DeclList...)
	name := fl.file.PkgName.Value
	nameOff := fl.off(fl.file.PkgName.Pos())
	fl.add(nameOff, nameOff+len(name), name+text)
}
