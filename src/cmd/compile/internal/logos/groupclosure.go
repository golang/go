// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"fmt"
	"slices"
	"strings"

	"cmd/compile/internal/syntax"
)

type groupClosureTextKind uint8

const (
	groupClosureSignature groupClosureTextKind = iota
	groupClosureSuccess
	groupClosureVoidReturn
	groupClosureExpression
	groupClosureInvocation
	groupClosurePreludeText
)

type groupClosure struct {
	registry   *groupClosureRegistry
	id         int
	literal    *syntax.FuncLit
	call       *syntax.CallExpr
	source     span
	resultText []string
	values     []string
	err        string
	lifted     bool
	marks      [5]string
	root       *groupClosurePrelude
	preludes   int
	original   otext
	outputs    []string
}

type groupClosureText struct {
	closure *groupClosure
	kind    groupClosureTextKind
	normal  string
	prelude *groupClosurePrelude
}

type groupClosureRegistry struct {
	file       *fileLower
	calls      map[*syntax.CallExpr]*groupClosure
	literals   map[*syntax.FuncLit]*groupClosure
	texts      map[string]groupClosureText
	statements map[syntax.Stmt]*groupClosurePrelude
	owner      *funcLower
	seq        int
}

type groupClosurePrelude struct {
	closure *groupClosure
	text    otext
}

func (c *groupClosure) prefix(statement syntax.Stmt) string {
	if c == nil {
		return ""
	}
	prelude := &groupClosurePrelude{closure: c}
	mark := fmt.Sprintf("\x00groupclosure%d_p%d\x00", c.id, c.preludes)
	c.preludes++
	c.registry.texts[mark] = groupClosureText{closure: c, kind: groupClosurePreludeText, prelude: prelude}
	if statement == nil {
		c.root = prelude
	} else {
		c.registry.statements[statement] = prelude
	}
	return mark
}

func (fl *fileLower) groupClosurePrelude(statement syntax.Stmt) *groupClosurePrelude {
	if fl.groupClosures == nil {
		return nil
	}
	return fl.groupClosures.statements[statement]
}

func (h *groupHoist) atStatement(statement syntax.Stmt) {
	h.prelude = h.fc.fl.groupClosurePrelude(statement)
	if h.prelude != nil {
		h.from = h.prelude.closure.source.start
	}
}

func (h *groupHoist) scoped(statement syntax.Stmt) []syntax.Stmt {
	if h.prelude != nil {
		return append(h.native, statement)
	}
	return h.fc.scoped(statement, h.native)
}

func (fl *fileLower) newGroupClosure(original syntax.Node) *groupClosure {
	if !fl.groupNodesHaveRecv(original) {
		return nil
	}
	if fl.groupClosures == nil {
		fl.groupClosures = &groupClosureRegistry{
			file:       fl,
			calls:      make(map[*syntax.CallExpr]*groupClosure),
			literals:   make(map[*syntax.FuncLit]*groupClosure),
			texts:      make(map[string]groupClosureText),
			statements: make(map[syntax.Stmt]*groupClosurePrelude),
		}
	}
	r := fl.groupClosures
	c := &groupClosure{registry: r, id: r.seq}
	r.seq++
	return c
}

func (c *groupClosure) text(kind groupClosureTextKind, normal string) string {
	if c == nil {
		return normal
	}
	if c.marks[kind] == "" {
		mark := fmt.Sprintf("\x00groupclosure%d_%d\x00", c.id, kind)
		c.marks[kind] = mark
		c.registry.texts[mark] = groupClosureText{closure: c, kind: kind, normal: normal}
	}
	return c.marks[kind]
}

func (c *groupClosure) head(normal string, results []string) string {
	if c != nil {
		c.resultText = results
	}
	return c.text(groupClosureSignature, normal)
}

func (c *groupClosure) success() string {
	return c.text(groupClosureSuccess, "")
}

func (c *groupClosure) voidReturn() string {
	return c.text(groupClosureVoidReturn, "")
}

func (c *groupClosure) bind(call *syntax.CallExpr, source span, id int) {
	if c == nil {
		return
	}
	c.call, c.source = call, source
	c.literal = call.Fun.(*syntax.FuncLit)
	c.registry.calls[call] = c
	c.registry.literals[c.literal] = c
	if c.root != nil {
		c.registry.statements[c.literal.Body.List[0]] = c.root
	}
	fl := c.registry.file
	head, tail := fl.drainBoundariesSince(source, id)
	owner, mark := fl.openMove(source)
	c.original = cat(head, gen(mark), tail)
	*owner = append(*owner, splice{start: source.start, end: source.end, text: c.text(groupClosureExpression, "")})
}

func (c *groupClosure) armEnd(end int) {
	c.registry.file.addAfterExpression(end, c.success())
}

func (fl *fileLower) groupClosureFor(literal *syntax.FuncLit) *groupClosure {
	if fl.groupClosures == nil {
		return nil
	}
	return fl.groupClosures.literals[literal]
}

func (fl *fileLower) groupClosureCall(e syntax.Expr) *groupClosure {
	if fl.groupClosures == nil {
		return nil
	}
	call, ok := syntax.Unparen(e).(*syntax.CallExpr)
	if !ok {
		return nil
	}
	return fl.groupClosures.calls[call]
}

func (fl *fileLower) groupClosureText(mark string) (otext, bool) {
	if fl.groupClosures == nil {
		return otext{}, false
	}
	t, ok := fl.groupClosures.texts[mark]
	if !ok {
		return otext{}, false
	}
	c := t.closure
	switch t.kind {
	case groupClosureInvocation:
		return fl.settleMarks(c.original), true
	case groupClosureExpression:
		if c.lifted {
			return gen(strings.Join(c.outputs, ", ")), true
		}
		return fl.settleMarks(c.original), true
	}
	if !c.lifted {
		return gen(t.normal), true
	}
	switch t.kind {
	case groupClosurePreludeText:
		return fl.settleMarks(t.prelude.text), true
	case groupClosureSignature:
		results := make([]string, 0, len(c.resultText)+1)
		for i, typ := range c.resultText {
			results = append(results, c.values[i]+" "+typ)
		}
		results = append(results, c.err+" error")
		return gen("func() (" + strings.Join(results, ", ") + ")"), true
	case groupClosureSuccess:
		return gen(", nil"), true
	case groupClosureVoidReturn:
		return gen("return nil; "), true
	}
	panic("invalid generated group closure text")
}

func (fl *fileLower) resolveGroupClosureText() {
	if fl.groupClosures == nil {
		return
	}
	resolve := func(list []splice) {
		for i := range list {
			t := substNul(otext{s: list[i].text, chunks: list[i].chunks}, fl.groupClosureText)
			list[i].text, list[i].chunks = t.s, t.chunks
		}
	}
	resolve(fl.splices)
	for _, region := range fl.regions {
		resolve(region.splices)
	}
	for _, move := range fl.moves {
		resolve(move.inner)
	}
}

func (fc *funcLower) groupClosureOwner() func() {
	if fc.fl.groupClosures == nil {
		return func() {}
	}
	r := fc.fl.groupClosures
	saved := r.owner
	r.owner = fc
	return func() { r.owner = saved }
}

func (fl *fileLower) groupClosurePending(literal *syntax.FuncLit) bool {
	c := fl.groupClosureFor(literal)
	if c == nil || c.lifted || c.registry.owner == nil {
		return false
	}
	_, child := c.registry.owner.groupChanTop()
	return child
}

func groupFlattenArgs(call *syntax.CallExpr) {
	for i, arg := range call.ArgList {
		if _, ok := arg.(*syntax.ListExpr); !ok {
			continue
		}
		args := slices.Clone(call.ArgList[:i])
		for _, rest := range call.ArgList[i:] {
			args = append(args, syntax.UnpackListExpr(rest)...)
		}
		call.ArgList = args
		return
	}
}

type groupClosureUse uint8

const (
	groupClosureValue groupClosureUse = iota
	groupClosureDiscard
)

func (h *groupHoist) statement(e syntax.Expr) syntax.Expr {
	if closure := h.fc.fl.groupClosureCall(e); closure != nil {
		return h.closure(closure, groupClosureDiscard)
	}
	return h.expr(e)
}

func (h *groupHoist) closure(c *groupClosure, use groupClosureUse) syntax.Expr {
	fc, fl := h.fc, h.fc.fl
	pos := h.scopePos(c.call.Pos())
	c.lifted = true
	fields := c.literal.Type.ResultList
	c.values = make([]string, len(fields))
	for i, field := range fields {
		c.values[i] = fc.groupTempName("r")
		fields[i] = at(&syntax.Field{Name: nameOf(pos, c.values[i]), Type: field.Type}, pos)
	}
	c.err = fc.groupTempName("cancel")
	c.literal.Type.ResultList = append(fields, at(&syntax.Field{Name: nameOf(pos, c.err), Type: nameOf(pos, "error")}, pos))
	syntax.Inspect(c.literal.Body, func(n syntax.Node) bool {
		if _, nested := n.(*syntax.FuncLit); nested {
			return false
		}
		if ret, ok := n.(*syntax.ReturnStmt); ok {
			results := append(syntax.UnpackListExpr(ret.Results), nilOf(ret.Pos()))
			ret.Results = listOf(ret.Pos(), results)
		}
		return true
	})
	if len(c.values) == 0 {
		c.literal.Body.List = append(c.literal.Body.List, returnOf(pos, []syntax.Expr{nilOf(pos)}))
	}
	child := newFuncLower(fl, c.literal.Type, nil, c.literal.Body)
	child.ft, child.specs = c.literal.Type, fl.resultSpecs(c.literal.Type, nil)
	child.errName = c.err
	child.errSeq, child.funSeq, child.argSeq = fc.errSeq, fc.funSeq, fc.argSeq
	child.tmpSeq, child.scopeSeq, child.groupSeq = fc.tmpSeq, fc.scopeSeq, fc.groupSeq
	child.groupStack = []groupFrame{{g: h.top.g, gctx: h.top.gctx, child: true, closure: c}}
	child.scopeParents = fc.scopeParents
	fc.groupMarkDone(c.literal)
	child.run()
	fc.groupTakeChild(child)

	values := make([]string, len(c.values))
	for i := range values {
		values[i] = "_"
		if use == groupClosureValue {
			values[i] = fc.groupTempName("t")
		}
	}
	if use == groupClosureValue {
		c.outputs = values
	}
	err := fc.groupTempName("cancel")
	names := append(slices.Clone(values), err)
	init := defineOf(pos, tempNodes(pos, names), []syntax.Expr{c.call})
	results := fc.groupDoneResults(pos, h.top)
	results[len(results)-1] = nameOf(pos, err)
	check := ifOf(pos, nil, notNil(pos, nameOf(pos, err)), []syntax.Stmt{returnOf(pos, results)})
	var exit []string
	for _, result := range results {
		exit = append(exit, renderExpr(result))
	}
	callText := gen(c.text(groupClosureInvocation, ""))
	h.add([]syntax.Stmt{init, check},
		cat(gen(strings.Join(names, ", ")+" := "), callText),
		gen("if "+err+" != nil { return "+strings.Join(exit, ", ")+" }"))
	if use == groupClosureDiscard || len(values) == 0 {
		return nil
	}
	return listOf(pos, tempNodes(pos, values))
}
