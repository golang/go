// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lint

import (
	"slices"

	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
	"internal/buildcfg"
)

// concurrentMultiwordRace is concurrency::concurrent_multiword_race (SPEC
// Sec 1.2, #13, #148): a local variable of multi-word type that a child
// captures and either side writes after the spawn. A child is the function
// literal of a go statement, a loGo/s group child block, or a lambda or
// function literal of a group stage segment, together with the function
// literals it calls through a local variable, a slice literal, or the
// source of its stage, and the literals those call in turn at any depth;
// a callee whose literal is not visible in the declaration is not
// followed. A variable is
// multi-word when its value spans more than one machine word: a string
// (two), a slice (three), an interface or union (two), a complex number
// (two), and a struct or array whose value-bearing storage, padding and
// zero-sized fields excluded, spans more than one target machine word. A value
// of a logos/sync type is exempt, and so is a const view, which nothing
// writes through. A write is an assignment, an increment, or a range
// assignment to the variable itself or to a field or array element held in
// it; storing through a pointer, a slice element, or a map entry writes other
// storage. The spawn starts the write window, which ends with the
// innermost enclosing go group block, because the group joins its children
// there, and otherwise with the enclosing declaration. For a variable
// declared before a loop that encloses the spawn and ends inside the
// window, the window starts at the first code of each iteration, which is
// its condition, else its post statement, else its body, instead, because
// the next iteration evaluates them while the previous child still reads
// and a one-time init stays outside. A write by the child is inside the
// window. The function value and the arguments of a go statement run in
// the calling goroutine, so a variable that only they touch is not
// shared. The rule reads syntax and checker objects only, so it holds for
// Go packages through the go statement leg and for loGo/s packages
// through both legs.
var concurrentMultiwordRace = Rule{
	ID:    "concurrency::concurrent_multiword_race",
	Group: Concurrency,
	Go:    true,
	Check: checkConcurrentMultiwordRace,
}

// spawn is one child of a declaration: its statement, the node whose span is
// the child's own scope (nil for a go statement whose target is a stored
// literal, which runs only that literal), for a segment child the end of the
// chain or loop that joins it, and the function literals the child calls
// through a local variable, a slice literal, or a stage source, which run in
// the child.
type spawn struct {
	node   syntax.Node
	child  syntax.Node
	join   syntax.Pos
	thunks []*syntax.FuncLit
}

// varWrite is one assignment to a variable.
type varWrite struct {
	v   *types2.Var
	pos syntax.Pos
}

// raceCheck is the analysis of one top-level declaration.
type raceCheck struct {
	pass    *Pass
	end     syntax.Pos
	writes  []varWrite
	groups  []*syntax.GroupStmt
	spawns  []spawn
	goCalls []*syntax.CallStmt
	loops   []*syntax.ForStmt
	bound   map[*types2.Var][]syntax.Expr
}

// checkConcurrentMultiwordRace is the Check of concurrentMultiwordRace.
func checkConcurrentMultiwordRace(pass *Pass) {
	for _, f := range pass.Files {
		for _, d := range f.DeclList {
			c := &raceCheck{pass: pass, end: syntax.EndPos(d), bound: make(map[*types2.Var][]syntax.Expr)}
			c.collect(d)
			c.followThunks()
			c.report()
		}
	}
}

// collect records the writes, groups, and children under root.
func (c *raceCheck) collect(root syntax.Node) {
	var path []syntax.Node
	syntax.Inspect(root, func(n syntax.Node) bool {
		if n == nil {
			path = path[:len(path)-1]
			return true
		}
		path = append(path, n)
		c.noteWrites(n)
		c.noteBindings(n)
		switch n := n.(type) {
		case *syntax.GroupStmt:
			c.groups = append(c.groups, n)
		case *syntax.GroupChild:
			c.spawns = append(c.spawns, spawn{node: n, child: n})
		case *syntax.CallStmt:
			if lit := goLiteral(n); lit != nil {
				c.spawns = append(c.spawns, spawn{node: n, child: lit})
			} else if n.Tok == syntax.Go {
				c.goCalls = append(c.goCalls, n)
			}
		case *syntax.ForStmt:
			c.loops = append(c.loops, n)
		case *syntax.CallExpr:
			c.noteSegment(n, path)
		}
		return true
	})
}

func (c *raceCheck) noteSegment(call *syntax.CallExpr, path []syntax.Node) {
	stage := stageOf(c.pass, call)
	if stage == nil {
		return
	}
	source := stageSource(stage)
	join := syntax.EndPos(joinNode(path))
	for _, fn := range funcArgs(call) {
		c.bindParam(fn, source)
		c.spawns = append(c.spawns, spawn{node: fn, child: fn, join: join})
	}
}

func stageSource(stage *syntax.CallExpr) syntax.Expr {
	sel := syntax.Unparen(stage.Fun).(*syntax.SelectorExpr)
	if sel.Sel.Value == "unordered" {
		sel = syntax.Unparen(callOf(sel.X).Fun).(*syntax.SelectorExpr)
	}
	return sel.X
}

func (c *raceCheck) bindParam(fn, source syntax.Expr) {
	var param *syntax.Name
	switch fn := fn.(type) {
	case *syntax.LambdaExpr:
		if len(fn.Params) > 0 {
			param = fn.Params[0]
		}
	case *syntax.FuncLit:
		if len(fn.Type.ParamList) > 0 {
			param = fn.Type.ParamList[0].Name
		}
	}
	if param != nil {
		c.bind(param, source)
	}
}

func (c *raceCheck) noteBindings(n syntax.Node) {
	switch n := n.(type) {
	case *syntax.AssignStmt:
		if n.Rhs == nil || (n.Op != 0 && n.Op != syntax.Def) {
			return
		}
		lhs, rhs := exprList(n.Lhs), exprList(n.Rhs)
		if len(lhs) != len(rhs) {
			return
		}
		for i := range lhs {
			c.bind(lhs[i], rhs[i])
		}
	case *syntax.VarDecl:
		if n.Values == nil {
			return
		}
		vals := exprList(n.Values)
		if len(vals) != len(n.NameList) {
			return
		}
		for i, name := range n.NameList {
			c.bind(name, vals[i])
		}
	case *syntax.RangeClause:
		if l, ok := n.Lhs.(*syntax.ListExpr); ok && len(l.ElemList) == 2 {
			c.bind(l.ElemList[1], n.X)
		}
	}
}

func (c *raceCheck) bind(lhs, rhs syntax.Expr) {
	if name, ok := syntax.Unparen(lhs).(*syntax.Name); ok {
		if v := c.varOf(name); v != nil {
			c.bound[v] = append(c.bound[v], rhs)
		}
	}
}

func (c *raceCheck) varOf(name *syntax.Name) *types2.Var {
	obj := c.pass.Info.Uses[name]
	if obj == nil {
		obj = c.pass.Info.Defs[name]
	}
	v, _ := obj.(*types2.Var)
	return v
}

func (c *raceCheck) followThunks() {
	for i := range c.spawns {
		s := &c.spawns[i]
		s.thunks = c.reachable(s.child, nil)
	}
	for _, g := range c.goCalls {
		call := callOf(g.Call)
		if call == nil {
			continue
		}
		if lits := c.litsOf(call.Fun, make(map[*types2.Var]bool)); len(lits) > 0 {
			c.spawns = append(c.spawns, spawn{node: g, thunks: c.reachable(nil, lits)})
		}
	}
}

func (c *raceCheck) reachable(root syntax.Node, seeds []*syntax.FuncLit) []*syntax.FuncLit {
	out := slices.Clone(seeds)
	var work []syntax.Node
	if root != nil {
		work = append(work, root)
	}
	for _, lit := range seeds {
		work = append(work, lit)
	}
	for len(work) > 0 {
		node := work[len(work)-1]
		work = work[:len(work)-1]
		syntax.Inspect(node, func(n syntax.Node) bool {
			call, ok := n.(*syntax.CallExpr)
			if !ok {
				return true
			}
			for _, lit := range c.litsOf(call.Fun, make(map[*types2.Var]bool)) {
				inside := root != nil && within(syntax.StartPos(lit), syntax.StartPos(root), syntax.EndPos(root))
				if !inside && !slices.Contains(out, lit) {
					out = append(out, lit)
					work = append(work, lit)
				}
			}
			return true
		})
	}
	return out
}

func (c *raceCheck) litsOf(e syntax.Expr, seen map[*types2.Var]bool) []*syntax.FuncLit {
	switch e := syntax.Unparen(e).(type) {
	case *syntax.FuncLit:
		return []*syntax.FuncLit{e}
	case *syntax.CompositeLit:
		var out []*syntax.FuncLit
		for _, el := range e.ElemList {
			if kv, ok := el.(*syntax.KeyValueExpr); ok {
				el = kv.Value
			}
			out = append(out, c.litsOf(el, seen)...)
		}
		return out
	case *syntax.IndexExpr:
		return c.litsOf(e.X, seen)
	case *syntax.Name:
		v := c.varOf(e)
		if v == nil || seen[v] {
			return nil
		}
		seen[v] = true
		var out []*syntax.FuncLit
		for _, x := range c.bound[v] {
			out = append(out, c.litsOf(x, seen)...)
		}
		return out
	}
	return nil
}

func joinNode(path []syntax.Node) syntax.Node {
	top := len(path) - 1
	for {
		next, ok := chainParent(path, top)
		if !ok {
			break
		}
		top = next
	}
	if p := ancestor(path, top); p >= 0 {
		if _, ok := path[p].(*syntax.RangeClause); ok {
			if f := ancestor(path, p); f >= 0 {
				return path[f]
			}
		}
	}
	return path[top]
}

func chainParent(path []syntax.Node, i int) (int, bool) {
	sel := ancestor(path, i)
	if sel < 0 {
		return 0, false
	}
	s, ok := path[sel].(*syntax.SelectorExpr)
	if !ok || syntax.Unparen(s.X) != path[i] {
		return 0, false
	}
	call := ancestor(path, sel)
	if call < 0 {
		return 0, false
	}
	c, ok := path[call].(*syntax.CallExpr)
	if !ok || syntax.Unparen(c.Fun) != s {
		return 0, false
	}
	return call, true
}

func ancestor(path []syntax.Node, i int) int {
	for i--; i >= 0; i-- {
		if _, ok := path[i].(*syntax.ParenExpr); !ok {
			return i
		}
	}
	return -1
}

// goLiteral returns the function literal that go statement s calls, or nil.
func goLiteral(s *syntax.CallStmt) *syntax.FuncLit {
	call := callOf(s.Call)
	if s.Tok != syntax.Go || call == nil {
		return nil
	}
	lit, _ := syntax.Unparen(call.Fun).(*syntax.FuncLit)
	return lit
}

// noteWrites records the variables that statement n assigns.
func (c *raceCheck) noteWrites(n syntax.Node) {
	var targets []syntax.Expr
	switch n := n.(type) {
	case *syntax.AssignStmt:
		targets = exprList(n.Lhs)
	case *syntax.RangeClause:
		if n.Lhs != nil && !n.Def {
			targets = exprList(n.Lhs)
		}
	}
	for _, t := range targets {
		if v := c.writtenVar(t); v != nil {
			c.writes = append(c.writes, varWrite{v: v, pos: syntax.StartPos(n)})
		}
	}
}

// writtenVar returns the variable whose own storage an assignment to e
// stores into, or nil.
func (c *raceCheck) writtenVar(e syntax.Expr) *types2.Var {
	for e != nil {
		if name, ok := e.(*syntax.Name); ok {
			v, _ := c.pass.Info.Uses[name].(*types2.Var)
			return v
		}
		e = c.inPlaceOperand(e)
	}
	return nil
}

// inPlaceOperand returns the operand x of a parenthesized expression, a
// field selection, or an array index e when e names storage inside x itself,
// and nil otherwise.
func (c *raceCheck) inPlaceOperand(e syntax.Expr) syntax.Expr {
	switch x := e.(type) {
	case *syntax.ParenExpr:
		return x.X
	case *syntax.SelectorExpr:
		sel := c.pass.Info.Selections[x]
		if sel != nil && sel.Kind() == types2.FieldVal && !sel.Indirect() {
			return x.X
		}
	case *syntax.IndexExpr:
		if c.isArray(TypeOf(x.X)) {
			return x.X
		}
	}
	return nil
}

// isArray reports whether t is an array type.
func (c *raceCheck) isArray(t types2.Type) bool {
	_, ok := underOrNil(t).(*types2.Array)
	return ok
}

// report flags every captured multi-word variable that a child shares with
// a write.
func (c *raceCheck) report() {
	for _, s := range c.spawns {
		for _, v := range c.shared(s) {
			if multiWord(v.Type()) && c.written(s, v) {
				c.flag(s, v)
			}
		}
	}
}

func (c *raceCheck) shared(s spawn) []*types2.Var {
	var out []*types2.Var
	if s.child != nil {
		out = c.captured(s.child)
	}
	for _, t := range s.thunks {
		for _, v := range c.captured(t) {
			if !slices.Contains(out, v) {
				out = append(out, v)
			}
		}
	}
	return out
}

// captured returns the local variables the child references that are
// declared outside it, in order of first reference.
func (c *raceCheck) captured(child syntax.Node) []*types2.Var {
	var out []*types2.Var
	start, end := syntax.StartPos(child), syntax.EndPos(child)
	syntax.Inspect(child, func(n syntax.Node) bool {
		name, ok := n.(*syntax.Name)
		if !ok {
			return true
		}
		v, _ := c.pass.Info.Uses[name].(*types2.Var)
		if v != nil && c.isLocal(v) && !within(v.Pos(), start, end) && !slices.Contains(out, v) {
			out = append(out, v)
		}
		return true
	})
	return out
}

// isLocal reports whether v is a variable of a function, not a field or a
// package-level variable.
func (c *raceCheck) isLocal(v *types2.Var) bool {
	scope := v.Parent()
	return !v.IsField() && scope != nil && scope != c.pass.Pkg.Scope()
}

// within reports whether p lies between start and end, inclusive.
func within(p, start, end syntax.Pos) bool {
	return p.IsKnown() && p.Cmp(start) >= 0 && p.Cmp(end) <= 0
}

// written reports whether v is written inside the window of s.
func (c *raceCheck) written(s spawn, v *types2.Var) bool {
	end := c.window(s)
	start := c.windowStart(s, v, end)
	return slices.ContainsFunc(c.writes, func(w varWrite) bool {
		inThunk := slices.ContainsFunc(s.thunks, func(t *syntax.FuncLit) bool {
			return within(w.pos, syntax.StartPos(t), syntax.EndPos(t))
		})
		return w.v == v && (within(w.pos, start, end) || inThunk)
	})
}

// windowStart returns where a write to v begins to race with the child of
// s: the spawn, or the first code of an iteration of the outermost loop
// that encloses the spawn, ends inside the window, and was entered after v
// was declared. The first code of an iteration is the range clause of a
// range loop, else the condition, else the post statement, else the body.
func (c *raceCheck) windowStart(s spawn, v *types2.Var, end syntax.Pos) syntax.Pos {
	spawnStart := syntax.StartPos(s.node)
	start := spawnStart
	for _, l := range c.loops {
		loopStart, loopEnd := syntax.StartPos(l), syntax.EndPos(l)
		if !within(spawnStart, loopStart, loopEnd) || loopEnd.Cmp(end) > 0 || v.Pos().Cmp(loopStart) > 0 {
			continue
		}
		first := iterationStart(l)
		if first.Cmp(start) < 0 {
			start = first
		}
	}
	return start
}

func iterationStart(l *syntax.ForStmt) syntax.Pos {
	if rc, ok := l.Init.(*syntax.RangeClause); ok {
		return syntax.StartPos(rc)
	}
	if l.Cond != nil {
		return syntax.StartPos(l.Cond)
	}
	if l.Post != nil {
		return syntax.StartPos(l.Post)
	}
	return syntax.StartPos(l.Body)
}

// window returns the end of the window in which a write races with the
// child of s: the end of the innermost group that holds s, or else the end
// of the declaration, and no later than the join of a segment child.
func (c *raceCheck) window(s spawn) syntax.Pos {
	end := c.end
	if s.join.IsKnown() && s.join.Cmp(end) < 0 {
		end = s.join
	}
	start := syntax.StartPos(s.node)
	for _, g := range c.groups {
		gend := syntax.EndPos(g)
		if within(start, syntax.StartPos(g), gend) && gend.Cmp(end) < 0 {
			end = gend
		}
	}
	return end
}

// flag reports that the child of s shares v with a write.
func (c *raceCheck) flag(s spawn, v *types2.Var) {
	typ := types2.TypeString(v.Type(), types2.RelativeTo(c.pass.Pkg))
	c.pass.Report(Diagnostic{
		Pos:     syntax.StartPos(s.node),
		End:     syntax.EndPos(s.node),
		Message: "variable " + v.Name() + " of multi-word type " + typ + " is shared with this child and written; guard it with logos/sync or pass a const view",
		Provenance: Provenance{
			OriginExpr:   v.Name(),
			OriginLine:   int(v.Pos().Line()),
			InferredType: typ,
		},
	})
}

// multiWord reports whether a variable of type t races when a child shares
// it with a write: its value spans more than one word and t is not a
// logos/sync type.
func multiWord(t types2.Type) bool {
	return !isLogosSyncType(t) && wordsOf(t) > 1
}

// isLogosSyncType reports whether t is a type declared by logos/sync.
func isLogosSyncType(t types2.Type) bool {
	named, ok := types2.Unalias(t).(*types2.Named)
	if !ok {
		return false
	}
	pkg := named.Obj().Pkg()
	return pkg != nil && pkg.Path() == logosSyncPath
}

// wordsOf returns the machine words a value of type t spans: its
// value-bearing bytes over the target word size, rounded up once.
func wordsOf(t types2.Type) int {
	word := targetWord()
	return int((valueBytes(t, word) + word - 1) / word)
}

// targetWord returns the target machine word size in bytes: the size the
// gc layout gives an int for the configured GOARCH, which is one word on
// every architecture, or eight when the layout is unknown.
func targetWord() int64 {
	sizes := types2.SizesFor("gc", buildcfg.GOARCH)
	if sizes == nil {
		return 8
	}
	if word := sizes.Sizeof(types2.Typ[types2.Int]); word > 0 {
		return word
	}
	return 8
}

// valueBytes returns the value-bearing bytes a value of type t holds:
// padding and zero-sized fields count nothing, so a struct or array sums
// the true sizes of its parts. A pointer, a map, a channel, a function, a
// const view, and a type parameter count one word.
func valueBytes(t types2.Type, word int64) int64 {
	if _, ok := types2.Unalias(t).(*types2.TypeParam); ok {
		return word
	}
	switch u := t.Underlying().(type) {
	case *types2.Basic:
		return basicBytes(u, word)
	case *types2.Slice:
		return 3 * word
	case *types2.Interface:
		return 2 * word
	case *types2.Array:
		return min(u.Len()*valueBytes(u.Elem(), word), 2*word)
	case *types2.Struct:
		total := int64(0)
		for i := range u.NumFields() {
			total += valueBytes(u.Field(i).Type(), word)
		}
		return total
	}
	return word
}

// basicBytes returns the value-bearing bytes of a basic type: a string
// takes its two-word header, a complex number its exact width, a narrow
// scalar its true size, and int, uint, and uintptr one word.
func basicBytes(b *types2.Basic, word int64) int64 {
	if b.Info()&types2.IsString != 0 {
		return 2 * word
	}
	switch b.Kind() {
	case types2.Complex128:
		return 16
	case types2.Complex64:
		return 8
	case types2.Bool, types2.Int8, types2.Uint8:
		return 1
	case types2.Int16, types2.Uint16:
		return 2
	case types2.Int32, types2.Uint32, types2.Float32:
		return 4
	case types2.Int64, types2.Uint64, types2.Float64:
		return 8
	}
	return word
}
