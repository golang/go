// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"fmt"
	"reflect"
	"slices"
	"strconv"
	"strings"

	"cmd/compile/internal/base"
	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
)

// This file lowers M6-S1 interpolated strings (SPEC Sec 1.4, T-104.4).
// After this pass no *syntax.InterpExpr remains: each one becomes an
// immediately invoked `func() string` literal, so the backend only ever
// sees plain Go.
//
// An interpolation with operands lowers to
//
//	func() string {
//		_logos_v0 := <op0>
//		...
//		_logos_s1 := _logos_v1.Error() // error/Stringer operands only
//		_logos_b := make([]byte, 0, <bound>)
//		_logos_b = append(_logos_b, "lit0"...)
//		_logos_b = strconv.AppendInt(_logos_b, int64(_logos_v0), 10)
//		...
//		return unsafe.String(unsafe.SliceData(_logos_b), len(_logos_b))
//	}()
//
// Every operand binds to a temporary once, left to right (#64, #126), so
// sizing and appending share one evaluation. Error and fmt.Stringer
// operands convert to strings before sizing, and the bound counts those
// strings: literal bytes plus len of each string/byte operand plus 20
// per integer, 24 per float, and 5 per bool (#34). One make means one
// allocation, never a regrow. Operand text is data, never a format
// string (#177). A literal without operands becomes its plain string.
//
// The pass runs last in lowerFile, after every other splice producer, so
// each operand's source comes from textOf, which drains the already
// recorded inner splices into the generated text. The span runs brace to
// brace, so zero-width boundary inserts of option and switch wrappers
// sitting exactly at an operand edge land strictly inside and are
// absorbed rather than left overlapping the whole-span replacement. The
// native tree reuses the live operand nodes, which earlier passes
// already lowered in place.

const (
	// interpConvPkg and interpUnsafePkg are the synthesized import
	// qualifiers for the strconv and unsafe packages. The reserved
	// prefix keeps them clear of every user identifier, so generated
	// code resolves regardless of file-level shadowing.
	interpConvPkg   = "_logos_strconv"
	interpUnsafePkg = "_logos_unsafe"
	// interpBuf is the buffer temporary; interpVal and interpStr are
	// the per-operand value and converted-string prefixes. Each
	// interpolation is its own function literal scope, so the names
	// restart every time.
	interpBuf = "_logos_b"
	interpVal = "_logos_v%d"
	interpStr = "_logos_s%d"
)

// Bytes the buffer bound adds for each operand whose text has a fixed
// worst-case width: SPEC T-104.4 counts 20 per integer, 24 per float, and 5
// per bool. Every other operand adds its own len, known only at run time.
const (
	interpIntBound   = 20
	interpFloatBound = 24
	interpBoolBound  = 5
)

// InterpFixedBytes returns the part of the buffer bound of an interpolation
// that no run-time length changes: the literal bytes of parts plus the fixed
// width of each integer, float, and bool operand in kinds. The editor hover
// renders it as `Fixed: <n>B + dynamic`.
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

// interpFixedWidth returns the fixed width the bound adds for an operand of
// kind k, and 0 for a kind that adds its run-time length.
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

// interpKinds holds the pre-lowering operand kinds per interpolation,
// keyed by node. collectInterpKinds fills it at the top of lowerFile
// while every operand node is still the checked original: later passes
// (notably lowerOptions) swap wrappers into operand slots whose fresh
// nodes carry no recorded type. It runs a second time when lowerConstFor
// returns, after every comptime copy exists and before any pass rewrites
// an operand slot: the generic copies and the unrolled const for
// iterations are fresh nodes that the copy checks typed.
var interpKinds = make(map[*syntax.InterpExpr][]types2.LogosInterpKind)

// collectInterpKinds records every interpolation's operand kinds while
// the tree still holds exactly what the checker saw. An operand without
// a recorded type, or of a type no interpolation accepts, is an internal
// error: the main check rejects both in the source, and the copy checks
// bind every alias to a complete package, and the function copy checks
// report their diagnostics before the copy joins the tree. While any
// diagnostic is pending the fact may be missing by construction, so the
// interpolation is skipped and Lower exits on the diagnostic.
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
					return true // A reported diagnostic explains the missing fact, and Lower exits on it before survivors are rejected.
				}
				base.Fatalf("logos: interpolation operand at %s has unexpected type %s", op.Pos().String(), t)
			}
			kinds[i] = k
		}
		interpKinds[e] = kinds
		return true
	})
}

// lowerInterp lowers every interpolated string of one file, innermost
// first: an outer whole-span replacement would otherwise overlap the
// splices of any inner interpolation it contains. Kinds come from the
// collectInterpKinds pre-pass; operand texts drain through textOf, which
// absorbs the inner whole-span splice of any nested literal.
func (fl *fileLower) lowerInterp() {
	var lits []*syntax.InterpExpr
	syntax.Inspect(fl.file, func(n syntax.Node) bool {
		if e, ok := n.(*syntax.InterpExpr); ok {
			lits = append(lits, e)
		}
		return true
	})
	slices.SortFunc(lits, func(a, b *syntax.InterpExpr) int {
		return len(a.Raw) - len(b.Raw)
	})
	// Scoped capture pre-lowering (D4, D7) credits its import needs
	// here, so the flush below still runs once even when every
	// interpolation was pre-lowered.
	needConv, needUnsafe := fl.interpNeedConv, fl.interpNeedUnsafe
	for _, e := range lits {
		kinds := interpKinds[e]
		delete(interpKinds, e)
		if kinds == nil && len(e.Exprs) > 0 && base.Errors() > 0 {
			continue // collectInterpKinds skipped it under a reported diagnostic.
		}
		need, _ := fl.lowerInterpExpr(e, kinds)
		needConv = needConv || need
		if len(e.Exprs) > 0 {
			needUnsafe = true
		}
	}
	fl.flushInterpImports(needConv, needUnsafe)
}

// replaceInterpExprs replaces the *syntax.InterpExpr that n holds
// directly, as a field or as an element of a list field, with repl of
// it. It mirrors replaceSwitchExprs: the caller walks the whole tree
// (lowerInterpLit) and lowers innermost first, so replacements never
// nest inside a pending whole-span splice.
func replaceInterpExprs(n syntax.Node, repl func(*syntax.InterpExpr) syntax.Expr) {
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
			replaceInterpField(v.Field(i), repl)
		}
	}
}

// replaceInterpField replaces an *syntax.InterpExpr held by the
// interface value f, or by an element of the slice or array f.
func replaceInterpField(f reflect.Value, repl func(*syntax.InterpExpr) syntax.Expr) {
	switch f.Kind() {
	case reflect.Interface:
		if f.IsNil() || !f.CanSet() {
			return
		}
		if e, ok := f.Elem().Interface().(*syntax.InterpExpr); ok {
			f.Set(reflect.ValueOf(repl(e)))
		}
	case reflect.Slice, reflect.Array:
		for i := range f.Len() {
			replaceInterpField(f.Index(i), repl)
		}
	}
}

// lowerInterpExpr lowers one interpolation, returning whether it needed
// a strconv import and the generated function literal text. The whole
// `$"..."` source span becomes one splice holding that text, and the
// node is replaced in the tree by the matching native call. kinds holds
// the pre-lowering operand kinds; operand texts drain from the live
// tree, absorbing whatever inner splices earlier passes recorded.
func (fl *fileLower) lowerInterpExpr(e *syntax.InterpExpr, kinds []types2.LogosInterpKind) (bool, string) {
	pos := e.Pos()
	start := fl.off(pos)
	end := start + len(e.Raw)
	if len(e.Exprs) == 0 {
		text := fmt.Sprintf("%q", e.Parts[0])
		fl.add(start, end, text)
		fl.replaceInterpLit(e, litOf(pos, text, syntax.StringLit))
		return false, text
	}
	if len(kinds) != len(e.Exprs) || len(e.Opens) != len(e.Exprs) || len(e.Closes) != len(e.Exprs) {
		base.Fatalf("logos: interpolation at %s has %d operands but %d kinds, %d opens, %d closes",
			pos.String(), len(e.Exprs), len(kinds), len(e.Opens), len(e.Closes))
	}
	opTexts := make([]string, len(e.Exprs))
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
	var lines []string
	for i := range e.Exprs {
		// The node comes from the live tree (earlier passes may have
		// rewritten it, and an inner literal already swapped its
		// native form in); kinds and texts were fixed up front.
		op := e.Exprs[i]
		v := valName(i)
		stmts = append(stmts, defineOf(pos, []syntax.Expr{nameOf(pos, v)}, []syntax.Expr{op}))
		lines = append(lines, v+" := "+opTexts[i])
		if kinds[i] == types2.LogosInterpError || kinds[i] == types2.LogosInterpStringer {
			method := "Error"
			if kinds[i] == types2.LogosInterpStringer {
				method = "String"
			}
			s := strName(i)
			stmts = append(stmts, defineOf(pos,
				[]syntax.Expr{nameOf(pos, s)},
				[]syntax.Expr{callOf(pos, selOf(pos, nameOf(pos, v), method), nil)}))
			lines = append(lines, s+" := "+v+"."+method+"()")
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
	lines = append(lines, interpBuf+" := make([]byte, 0, "+boundText+")")
	emitAppend := func(arg syntax.Expr, argText string, spread bool) {
		call := callOf(pos, nameOf(pos, "append"), []syntax.Expr{nameOf(pos, interpBuf), arg})
		call.HasDots = spread
		stmts = append(stmts, assignOf(pos, []syntax.Expr{nameOf(pos, interpBuf)}, []syntax.Expr{call}))
		if spread {
			argText += "..."
		}
		lines = append(lines, interpBuf+" = append("+interpBuf+", "+argText+")")
	}
	emitConv := func(name string, args []syntax.Expr, argTexts []string) {
		call := callOf(pos, selOf(pos, nameOf(pos, interpConvPkg), name), args)
		stmts = append(stmts, assignOf(pos, []syntax.Expr{nameOf(pos, interpBuf)}, []syntax.Expr{call}))
		lines = append(lines, interpBuf+" = "+interpConvPkg+"."+name+"("+strings.Join(argTexts, ", ")+")")
	}
	conv1 := func(name, conv, v, last string) {
		emitConv(name,
			[]syntax.Expr{nameOf(pos, interpBuf), callOf(pos, nameOf(pos, conv), []syntax.Expr{nameOf(pos, v)}), litOf(pos, last, syntax.IntLit)},
			[]string{interpBuf, conv + "(" + v + ")", last})
	}
	for i, part := range e.Parts {
		if part != "" {
			quoted := fmt.Sprintf("%q", part)
			emitAppend(litOf(pos, quoted, syntax.StringLit), quoted, true)
		}
		if i >= len(e.Exprs) {
			continue
		}
		v := valName(i)
		switch kinds[i] {
		case types2.LogosInterpString, types2.LogosInterpBytes:
			emitAppend(nameOf(pos, v), v, true)
		case types2.LogosInterpError, types2.LogosInterpStringer:
			s := strName(i)
			emitAppend(nameOf(pos, s), s, true)
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
				[]string{interpBuf, "float64(" + v + ")", "'g'", "-1", bits})
		case types2.LogosInterpBool:
			emitConv("AppendBool",
				[]syntax.Expr{nameOf(pos, interpBuf), nameOf(pos, v)},
				[]string{interpBuf, v})
		}
	}
	lines = append(lines, "return "+interpUnsafePkg+".String("+interpUnsafePkg+".SliceData("+interpBuf+"), len("+interpBuf+"))")
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
	fl.replaceInterpLit(e, native)
	// D4: an interpolation is always string-typed, but the replacement
	// is synthetic; stash the type for chain prodType over capture
	// bodies.
	if fl.optBodyType == nil {
		fl.optBodyType = make(map[syntax.Expr]types2.Type)
	}
	fl.optBodyType[native] = types2.Typ[types2.String]
	text := "func() string {\n" + strings.Join(lines, "\n") + "\n}()"
	fl.add(start, end, text)
	return needConv, text
}

// replaceInterpLit swaps one interpolation for its lowered native form
// wherever it sits in the file: replaceInterpExprs only sees direct
// children, so this walks the whole tree like lowerUnions does.
func (fl *fileLower) replaceInterpLit(e *syntax.InterpExpr, native syntax.Expr) {
	syntax.Inspect(fl.file, func(n syntax.Node) bool {
		if n == nil {
			return true
		}
		replaceInterpExprs(n, func(target *syntax.InterpExpr) syntax.Expr {
			if target != e {
				return target
			}
			return native
		})
		return true
	})
}

// interpOperandText spells one `{...}` operand's lowered source,
// draining the inner splices earlier passes recorded. The span runs
// from the opening brace to one past the closing brace, so it needs no
// node positions at all: later passes may have rewritten the operand
// nodes (notably lowerOptions), but the braces never move. Zero-width
// boundary inserts sitting exactly at an operand edge land strictly
// inside and are absorbed.
func (fl *fileLower) interpOperandText(open, close syntax.Pos) string {
	text := fl.textOf(span{fl.off(open), fl.off(close) + 1})
	return text[1 : len(text)-1]
}

// flushInterpImports prepends the synthesized strconv and unsafe imports
// the lowered interpolations need, with matching eject-text edits. The
// text replaces the package-name span (re-emitting the name) rather
// than joining the shared header offset: same-offset inserts keep input
// order, so anything recorded earlier (rt header, type imports) would
// print ahead of the native prepend, while the name span sorts strictly
// before those splices, keeping both orders [mine, ...] together.
func (fl *fileLower) flushInterpImports(needConv, needUnsafe bool) {
	if !needConv && !needUnsafe {
		return
	}
	pos := fl.file.Pos()
	var decls []syntax.Decl
	var text string
	if needConv {
		decls = append(decls, at(&syntax.ImportDecl{
			LocalPkgName: nameOf(pos, interpConvPkg),
			Path:         litOf(pos, `"strconv"`, syntax.StringLit),
		}, pos))
		text += "\nimport " + interpConvPkg + ` "strconv"`
	}
	if needUnsafe {
		decls = append(decls, at(&syntax.ImportDecl{
			LocalPkgName: nameOf(pos, interpUnsafePkg),
			Path:         litOf(pos, `"unsafe"`, syntax.StringLit),
		}, pos))
		text += "\nimport " + interpUnsafePkg + ` "unsafe"`
	}
	fl.file.DeclList = append(decls, fl.file.DeclList...)
	name := fl.file.PkgName.Value
	nameOff := fl.off(fl.file.PkgName.Pos())
	fl.add(nameOff, nameOff+len(name), name+text)
}
