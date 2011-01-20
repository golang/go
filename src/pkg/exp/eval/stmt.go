// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package eval

import (
	"big"
	"log"
	"go/ast"
	"go/token"
)

const (
	returnPC = ^uint(0)
	badPC    = ^uint(1)
)

/*
 * Statement compiler
 */

type stmtCompiler struct {
	*blockCompiler
	pos token.Pos
	// This statement's label, or nil if it is not labeled.
	stmtLabel *label
}

func (a *stmtCompiler) diag(format string, args ...interface{}) {
	a.diagAt(a.pos, format, args...)
}

/*
 * Flow checker
 */

type flowEnt struct {
	// Whether this flow entry is conditional.  If true, flow can
	// continue to the next PC.
	cond bool
	// True if this will terminate flow (e.g., a return statement).
	// cond must be false and jumps must be nil if this is true.
	term bool
	// PC's that can be reached from this flow entry.
	jumps []*uint
	// Whether this flow entry has been visited by reachesEnd.
	visited bool
}

type flowBlock struct {
	// If this is a goto, the target label.
	target string
	// The inner-most block containing definitions.
	block *block
	// The numVars from each block leading to the root of the
	// scope, starting at block.
	numVars []int
}

type flowBuf struct {
	cb *codeBuf
	// ents is a map from PC's to flow entries.  Any PC missing
	// from this map is assumed to reach only PC+1.
	ents map[uint]*flowEnt
	// gotos is a map from goto positions to information on the
	// block at the point of the goto.
	gotos map[token.Pos]*flowBlock
	// labels is a map from label name to information on the block
	// at the point of the label.  labels are tracked by name,
	// since mutliple labels at the same PC can have different
	// blocks.
	labels map[string]*flowBlock
}

func newFlowBuf(cb *codeBuf) *flowBuf {
	return &flowBuf{cb, make(map[uint]*flowEnt), make(map[token.Pos]*flowBlock), make(map[string]*flowBlock)}
}

// put creates a flow control point for the next PC in the code buffer.
// This should be done before pushing the instruction into the code buffer.
func (f *flowBuf) put(cond bool, term bool, jumps []*uint) {
	pc := f.cb.nextPC()
	if ent, ok := f.ents[pc]; ok {
		log.Panicf("Flow entry already exists at PC %d: %+v", pc, ent)
	}
	f.ents[pc] = &flowEnt{cond, term, jumps, false}
}

// putTerm creates a flow control point at the next PC that
// unconditionally terminates execution.
func (f *flowBuf) putTerm() { f.put(false, true, nil) }

// put1 creates a flow control point at the next PC that jumps to one
// PC and, if cond is true, can also continue to the PC following the
// next PC.
func (f *flowBuf) put1(cond bool, jumpPC *uint) {
	f.put(cond, false, []*uint{jumpPC})
}

func newFlowBlock(target string, b *block) *flowBlock {
	// Find the inner-most block containing definitions
	for b.numVars == 0 && b.outer != nil && b.outer.scope == b.scope {
		b = b.outer
	}

	// Count parents leading to the root of the scope
	n := 0
	for bp := b; bp.scope == b.scope; bp = bp.outer {
		n++
	}

	// Capture numVars from each block to the root of the scope
	numVars := make([]int, n)
	i := 0
	for bp := b; i < n; bp = bp.outer {
		numVars[i] = bp.numVars
		i++
	}

	return &flowBlock{target, b, numVars}
}

// putGoto captures the block at a goto statement.  This should be
// called in addition to putting a flow control point.
func (f *flowBuf) putGoto(pos token.Pos, target string, b *block) {
	f.gotos[pos] = newFlowBlock(target, b)
}

// putLabel captures the block at a label.
func (f *flowBuf) putLabel(name string, b *block) {
	f.labels[name] = newFlowBlock("", b)
}

// reachesEnd returns true if the end of f's code buffer can be
// reached from the given program counter.  Error reporting is the
// caller's responsibility.
func (f *flowBuf) reachesEnd(pc uint) bool {
	endPC := f.cb.nextPC()
	if pc > endPC {
		log.Panicf("Reached bad PC %d past end PC %d", pc, endPC)
	}

	for ; pc < endPC; pc++ {
		ent, ok := f.ents[pc]
		if !ok {
			continue
		}

		if ent.visited {
			return false
		}
		ent.visited = true

		if ent.term {
			return false
		}

		// If anything can reach the end, we can reach the end
		// from pc.
		for _, j := range ent.jumps {
			if f.reachesEnd(*j) {
				return true
			}
		}
		// If the jump was conditional, we can reach the next
		// PC, so try reaching the end from it.
		if ent.cond {
			continue
		}
		return false
	}
	return true
}

// gotosObeyScopes returns true if no goto statement causes any
// variables to come into scope that were not in scope at the point of
// the goto.  Reports any errors using the given compiler.
func (f *flowBuf) gotosObeyScopes(a *compiler) {
	for pos, src := range f.gotos {
		tgt := f.labels[src.target]

		// The target block must be a parent of this block
		numVars := src.numVars
		b := src.block
		for len(numVars) > 0 && b != tgt.block {
			b = b.outer
			numVars = numVars[1:]
		}
		if b != tgt.block {
			// We jumped into a deeper block
			a.diagAt(pos, "goto causes variables to come into scope")
			return
		}

		// There must be no variables in the target block that
		// did not exist at the jump
		tgtNumVars := tgt.numVars
		for i := range numVars {
			if tgtNumVars[i] > numVars[i] {
				a.diagAt(pos, "goto causes variables to come into scope")
				return
			}
		}
	}
}

/*
 * Statement generation helpers
 */

func (a *stmtCompiler) defineVar(ident *ast.Ident, t Type) *Variable {
	v, prev := a.block.DefineVar(ident.Name, ident.Pos(), t)
	if prev != nil {
		if prev.Pos().IsValid() {
			a.diagAt(ident.Pos(), "variable %s redeclared in this block\n\tprevious declaration at %s", ident.Name, a.fset.Position(prev.Pos()))
		} else {
			a.diagAt(ident.Pos(), "variable %s redeclared in this block", ident.Name)
		}
		return nil
	}

	// Initialize the variable
	index := v.Index
	if v.Index >= 0 {
		a.push(func(v *Thread) { v.f.Vars[index] = t.Zero() })
	}
	return v
}

// TODO(austin) Move doAssign to here

/*
 * Statement compiler
 */

func (a *stmtCompiler) compile(s ast.Stmt) {
	if a.block.inner != nil {
		log.Panic("Child scope still entered")
	}

	notimpl := false
	switch s := s.(type) {
	case *ast.BadStmt:
		// Error already reported by parser.
		a.silentErrors++

	case *ast.DeclStmt:
		a.compileDeclStmt(s)

	case *ast.EmptyStmt:
		// Do nothing.

	case *ast.LabeledStmt:
		a.compileLabeledStmt(s)

	case *ast.ExprStmt:
		a.compileExprStmt(s)

	case *ast.IncDecStmt:
		a.compileIncDecStmt(s)

	case *ast.AssignStmt:
		a.compileAssignStmt(s)

	case *ast.GoStmt:
		notimpl = true

	case *ast.DeferStmt:
		notimpl = true

	case *ast.ReturnStmt:
		a.compileReturnStmt(s)

	case *ast.BranchStmt:
		a.compileBranchStmt(s)

	case *ast.BlockStmt:
		a.compileBlockStmt(s)

	case *ast.IfStmt:
		a.compileIfStmt(s)

	case *ast.CaseClause:
		a.diag("case clause outside switch")

	case *ast.SwitchStmt:
		a.compileSwitchStmt(s)

	case *ast.TypeCaseClause:
		notimpl = true

	case *ast.TypeSwitchStmt:
		notimpl = true

	case *ast.CommClause:
		notimpl = true

	case *ast.SelectStmt:
		notimpl = true

	case *ast.ForStmt:
		a.compileForStmt(s)

	case *ast.RangeStmt:
		notimpl = true

	default:
		log.Panicf("unexpected ast node type %T", s)
	}

	if notimpl {
		a.diag("%T statment node not implemented", s)
	}

	if a.block.inner != nil {
		log.Panic("Forgot to exit child scope")
	}
}

func (a *stmtCompiler) compileDeclStmt(s *ast.DeclStmt) {
	switch decl := s.Decl.(type) {
	case *ast.BadDecl:
		// Do nothing.  Already reported by parser.
		a.silentErrors++

	case *ast.FuncDecl:
		if !a.block.global {
			log.Panic("FuncDecl at statement level")
		}

	case *ast.GenDecl:
		if decl.Tok == token.IMPORT && !a.block.global {
			log.Panic("import at statement level")
		}

	default:
		log.Panicf("Unexpected Decl type %T", s.Decl)
	}
	a.compileDecl(s.Decl)
}

func (a *stmtCompiler) compileVarDecl(decl *ast.GenDecl) {
	for _, spec := range decl.Specs {
		spec := spec.(*ast.ValueSpec)
		if spec.Values == nil {
			// Declaration without assignment
			if spec.Type == nil {
				// Parser should have caught
				log.Panic("Type and Values nil")
			}
			t := a.compileType(a.block, spec.Type)
			// Define placeholders even if type compile failed
			for _, n := range spec.Names {
				a.defineVar(n, t)
			}
		} else {
			// Declaration with assignment
			lhs := make([]ast.Expr, len(spec.Names))
			for i, n := range spec.Names {
				lhs[i] = n
			}
			a.doAssign(lhs, spec.Values, decl.Tok, spec.Type)
		}
	}
}

func (a *stmtCompiler) compileDecl(decl ast.Decl) {
	switch d := decl.(type) {
	case *ast.BadDecl:
		// Do nothing.  Already reported by parser.
		a.silentErrors++

	case *ast.FuncDecl:
		decl := a.compileFuncType(a.block, d.Type)
		if decl == nil {
			return
		}
		// Declare and initialize v before compiling func
		// so that body can refer to itself.
		c, prev := a.block.DefineConst(d.Name.Name, a.pos, decl.Type, decl.Type.Zero())
		if prev != nil {
			pos := prev.Pos()
			if pos.IsValid() {
				a.diagAt(d.Name.Pos(), "identifier %s redeclared in this block\n\tprevious declaration at %s", d.Name.Name, a.fset.Position(pos))
			} else {
				a.diagAt(d.Name.Pos(), "identifier %s redeclared in this block", d.Name.Name)
			}
		}
		fn := a.compileFunc(a.block, decl, d.Body)
		if c == nil || fn == nil {
			return
		}
		var zeroThread Thread
		c.Value.(FuncValue).Set(nil, fn(&zeroThread))

	case *ast.GenDecl:
		switch d.Tok {
		case token.IMPORT:
			log.Panicf("%v not implemented", d.Tok)
		case token.CONST:
			log.Panicf("%v not implemented", d.Tok)
		case token.TYPE:
			a.compileTypeDecl(a.block, d)
		case token.VAR:
			a.compileVarDecl(d)
		}

	default:
		log.Panicf("Unexpected Decl type %T", decl)
	}
}

func (a *stmtCompiler) compileLabeledStmt(s *ast.LabeledStmt) {
	// Define label
	l, ok := a.labels[s.Label.Name]
	if ok {
		if l.resolved.IsValid() {
			a.diag("label %s redeclared in this block\n\tprevious declaration at %s", s.Label.Name, a.fset.Position(l.resolved))
		}
	} else {
		pc := badPC
		l = &label{name: s.Label.Name, gotoPC: &pc}
		a.labels[l.name] = l
	}
	l.desc = "regular label"
	l.resolved = s.Pos()

	// Set goto PC
	*l.gotoPC = a.nextPC()

	// Define flow entry so we can check for jumps over declarations.
	a.flow.putLabel(l.name, a.block)

	// Compile the statement.  Reuse our stmtCompiler for simplicity.
	sc := &stmtCompiler{a.blockCompiler, s.Stmt.Pos(), l}
	sc.compile(s.Stmt)
}

func (a *stmtCompiler) compileExprStmt(s *ast.ExprStmt) {
	bc := a.enterChild()
	defer bc.exit()

	e := a.compileExpr(bc.block, false, s.X)
	if e == nil {
		return
	}

	if e.exec == nil {
		a.diag("%s cannot be used as expression statement", e.desc)
		return
	}

	a.push(e.exec)
}

func (a *stmtCompiler) compileIncDecStmt(s *ast.IncDecStmt) {
	// Create temporary block for extractEffect
	bc := a.enterChild()
	defer bc.exit()

	l := a.compileExpr(bc.block, false, s.X)
	if l == nil {
		return
	}

	if l.evalAddr == nil {
		l.diag("cannot assign to %s", l.desc)
		return
	}
	if !(l.t.isInteger() || l.t.isFloat()) {
		l.diagOpType(s.Tok, l.t)
		return
	}

	var op token.Token
	var desc string
	switch s.Tok {
	case token.INC:
		op = token.ADD
		desc = "increment statement"
	case token.DEC:
		op = token.SUB
		desc = "decrement statement"
	default:
		log.Panicf("Unexpected IncDec token %v", s.Tok)
	}

	effect, l := l.extractEffect(bc.block, desc)

	one := l.newExpr(IdealIntType, "constant")
	one.pos = s.Pos()
	one.eval = func() *big.Int { return big.NewInt(1) }

	binop := l.compileBinaryExpr(op, l, one)
	if binop == nil {
		return
	}

	assign := a.compileAssign(s.Pos(), bc.block, l.t, []*expr{binop}, "", "")
	if assign == nil {
		log.Panicf("compileAssign type check failed")
	}

	lf := l.evalAddr
	a.push(func(v *Thread) {
		effect(v)
		assign(lf(v), v)
	})
}

func (a *stmtCompiler) doAssign(lhs []ast.Expr, rhs []ast.Expr, tok token.Token, declTypeExpr ast.Expr) {
	nerr := a.numError()

	// Compile right side first so we have the types when
	// compiling the left side and so we don't see definitions
	// made on the left side.
	rs := make([]*expr, len(rhs))
	for i, re := range rhs {
		rs[i] = a.compileExpr(a.block, false, re)
	}

	errOp := "assignment"
	if tok == token.DEFINE || tok == token.VAR {
		errOp = "declaration"
	}
	ac, ok := a.checkAssign(a.pos, rs, errOp, "value")
	ac.allowMapForms(len(lhs))

	// If this is a definition and the LHS is too big, we won't be
	// able to produce the usual error message because we can't
	// begin to infer the types of the LHS.
	if (tok == token.DEFINE || tok == token.VAR) && len(lhs) > len(ac.rmt.Elems) {
		a.diag("not enough values for definition")
	}

	// Compile left type if there is one
	var declType Type
	if declTypeExpr != nil {
		declType = a.compileType(a.block, declTypeExpr)
	}

	// Compile left side
	ls := make([]*expr, len(lhs))
	nDefs := 0
	for i, le := range lhs {
		// If this is a definition, get the identifier and its type
		var ident *ast.Ident
		var lt Type
		switch tok {
		case token.DEFINE:
			// Check that it's an identifier
			ident, ok = le.(*ast.Ident)
			if !ok {
				a.diagAt(le.Pos(), "left side of := must be a name")
				// Suppress new defitions errors
				nDefs++
				continue
			}

			// Is this simply an assignment?
			if _, ok := a.block.defs[ident.Name]; ok {
				ident = nil
				break
			}
			nDefs++

		case token.VAR:
			ident = le.(*ast.Ident)
		}

		// If it's a definition, get or infer its type.
		if ident != nil {
			// Compute the identifier's type from the RHS
			// type.  We use the computed MultiType so we
			// don't have to worry about unpacking.
			switch {
			case declTypeExpr != nil:
				// We have a declaration type, use it.
				// If declType is nil, we gave an
				// error when we compiled it.
				lt = declType

			case i >= len(ac.rmt.Elems):
				// Define a placeholder.  We already
				// gave the "not enough" error above.
				lt = nil

			case ac.rmt.Elems[i] == nil:
				// We gave the error when we compiled
				// the RHS.
				lt = nil

			case ac.rmt.Elems[i].isIdeal():
				// If the type is absent and the
				// corresponding expression is a
				// constant expression of ideal
				// integer or ideal float type, the
				// type of the declared variable is
				// int or float respectively.
				switch {
				case ac.rmt.Elems[i].isInteger():
					lt = IntType
				case ac.rmt.Elems[i].isFloat():
					lt = Float64Type
				default:
					log.Panicf("unexpected ideal type %v", rs[i].t)
				}

			default:
				lt = ac.rmt.Elems[i]
			}
		}

		// If it's a definition, define the identifier
		if ident != nil {
			if a.defineVar(ident, lt) == nil {
				continue
			}
		}

		// Compile LHS
		ls[i] = a.compileExpr(a.block, false, le)
		if ls[i] == nil {
			continue
		}

		if ls[i].evalMapValue != nil {
			// Map indexes are not generally addressable,
			// but they are assignable.
			//
			// TODO(austin) Now that the expression
			// compiler uses semantic values, this might
			// be easier to implement as a function call.
			sub := ls[i]
			ls[i] = ls[i].newExpr(sub.t, sub.desc)
			ls[i].evalMapValue = sub.evalMapValue
			mvf := sub.evalMapValue
			et := sub.t
			ls[i].evalAddr = func(t *Thread) Value {
				m, k := mvf(t)
				e := m.Elem(t, k)
				if e == nil {
					e = et.Zero()
					m.SetElem(t, k, e)
				}
				return e
			}
		} else if ls[i].evalAddr == nil {
			ls[i].diag("cannot assign to %s", ls[i].desc)
			continue
		}
	}

	// A short variable declaration may redeclare variables
	// provided they were originally declared in the same block
	// with the same type, and at least one of the variables is
	// new.
	if tok == token.DEFINE && nDefs == 0 {
		a.diag("at least one new variable must be declared")
		return
	}

	// If there have been errors, our arrays are full of nil's so
	// get out of here now.
	if nerr != a.numError() {
		return
	}

	// Check for 'a[x] = r, ok'
	if len(ls) == 1 && len(rs) == 2 && ls[0].evalMapValue != nil {
		a.diag("a[x] = r, ok form not implemented")
		return
	}

	// Create assigner
	var lt Type
	n := len(lhs)
	if n == 1 {
		lt = ls[0].t
	} else {
		lts := make([]Type, len(ls))
		for i, l := range ls {
			if l != nil {
				lts[i] = l.t
			}
		}
		lt = NewMultiType(lts)
	}
	bc := a.enterChild()
	defer bc.exit()
	assign := ac.compile(bc.block, lt)
	if assign == nil {
		return
	}

	// Compile
	if n == 1 {
		// Don't need temporaries and can avoid []Value.
		lf := ls[0].evalAddr
		a.push(func(t *Thread) { assign(lf(t), t) })
	} else if tok == token.VAR || (tok == token.DEFINE && nDefs == n) {
		// Don't need temporaries
		lfs := make([]func(*Thread) Value, n)
		for i, l := range ls {
			lfs[i] = l.evalAddr
		}
		a.push(func(t *Thread) {
			dest := make([]Value, n)
			for i, lf := range lfs {
				dest[i] = lf(t)
			}
			assign(multiV(dest), t)
		})
	} else {
		// Need temporaries
		lmt := lt.(*MultiType)
		lfs := make([]func(*Thread) Value, n)
		for i, l := range ls {
			lfs[i] = l.evalAddr
		}
		a.push(func(t *Thread) {
			temp := lmt.Zero().(multiV)
			assign(temp, t)
			// Copy to destination
			for i := 0; i < n; i++ {
				// TODO(austin) Need to evaluate LHS
				// before RHS
				lfs[i](t).Assign(t, temp[i])
			}
		})
	}
}

var assignOpToOp = map[token.Token]token.Token{
	token.ADD_ASSIGN: token.ADD,
	token.SUB_ASSIGN: token.SUB,
	token.MUL_ASSIGN: token.MUL,
	token.QUO_ASSIGN: token.QUO,
	token.REM_ASSIGN: token.REM,

	token.AND_ASSIGN:     token.AND,
	token.OR_ASSIGN:      token.OR,
	token.XOR_ASSIGN:     token.XOR,
	token.SHL_ASSIGN:     token.SHL,
	token.SHR_ASSIGN:     token.SHR,
	token.AND_NOT_ASSIGN: token.AND_NOT,
}

func (a *stmtCompiler) doAssignOp(s *ast.AssignStmt) {
	if len(s.Lhs) != 1 || len(s.Rhs) != 1 {
		a.diag("tuple assignment cannot be combined with an arithmetic operation")
		return
	}

	// Create temporary block for extractEffect
	bc := a.enterChild()
	defer bc.exit()

	l := a.compileExpr(bc.block, false, s.Lhs[0])
	r := a.compileExpr(bc.block, false, s.Rhs[0])
	if l == nil || r == nil {
		return
	}

	if l.evalAddr == nil {
		l.diag("cannot assign to %s", l.desc)
		return
	}

	effect, l := l.extractEffect(bc.block, "operator-assignment")

	binop := r.compileBinaryExpr(assignOpToOp[s.Tok], l, r)
	if binop == nil {
		return
	}

	assign := a.compileAssign(s.Pos(), bc.block, l.t, []*expr{binop}, "assignment", "value")
	if assign == nil {
		log.Panicf("compileAssign type check failed")
	}

	lf := l.evalAddr
	a.push(func(t *Thread) {
		effect(t)
		assign(lf(t), t)
	})
}

func (a *stmtCompiler) compileAssignStmt(s *ast.AssignStmt) {
	switch s.Tok {
	case token.ASSIGN, token.DEFINE:
		a.doAssign(s.Lhs, s.Rhs, s.Tok, nil)

	default:
		a.doAssignOp(s)
	}
}

func (a *stmtCompiler) compileReturnStmt(s *ast.ReturnStmt) {
	if a.fnType == nil {
		a.diag("cannot return at the top level")
		return
	}

	if len(s.Results) == 0 && (len(a.fnType.Out) == 0 || a.outVarsNamed) {
		// Simple case.  Simply exit from the function.
		a.flow.putTerm()
		a.push(func(v *Thread) { v.pc = returnPC })
		return
	}

	bc := a.enterChild()
	defer bc.exit()

	// Compile expressions
	bad := false
	rs := make([]*expr, len(s.Results))
	for i, re := range s.Results {
		rs[i] = a.compileExpr(bc.block, false, re)
		if rs[i] == nil {
			bad = true
		}
	}
	if bad {
		return
	}

	// Create assigner

	// However, if the expression list in the "return" statement
	// is a single call to a multi-valued function, the values
	// returned from the called function will be returned from
	// this one.
	assign := a.compileAssign(s.Pos(), bc.block, NewMultiType(a.fnType.Out), rs, "return", "value")

	// XXX(Spec) "The result types of the current function and the
	// called function must match."  Match is fuzzy.  It should
	// say that they must be assignment compatible.

	// Compile
	start := len(a.fnType.In)
	nout := len(a.fnType.Out)
	a.flow.putTerm()
	a.push(func(t *Thread) {
		assign(multiV(t.f.Vars[start:start+nout]), t)
		t.pc = returnPC
	})
}

func (a *stmtCompiler) findLexicalLabel(name *ast.Ident, pred func(*label) bool, errOp, errCtx string) *label {
	bc := a.blockCompiler
	for ; bc != nil; bc = bc.parent {
		if bc.label == nil {
			continue
		}
		l := bc.label
		if name == nil && pred(l) {
			return l
		}
		if name != nil && l.name == name.Name {
			if !pred(l) {
				a.diag("cannot %s to %s %s", errOp, l.desc, l.name)
				return nil
			}
			return l
		}
	}
	if name == nil {
		a.diag("%s outside %s", errOp, errCtx)
	} else {
		a.diag("%s label %s not defined", errOp, name.Name)
	}
	return nil
}

func (a *stmtCompiler) compileBranchStmt(s *ast.BranchStmt) {
	var pc *uint

	switch s.Tok {
	case token.BREAK:
		l := a.findLexicalLabel(s.Label, func(l *label) bool { return l.breakPC != nil }, "break", "for loop, switch, or select")
		if l == nil {
			return
		}
		pc = l.breakPC

	case token.CONTINUE:
		l := a.findLexicalLabel(s.Label, func(l *label) bool { return l.continuePC != nil }, "continue", "for loop")
		if l == nil {
			return
		}
		pc = l.continuePC

	case token.GOTO:
		l, ok := a.labels[s.Label.Name]
		if !ok {
			pc := badPC
			l = &label{name: s.Label.Name, desc: "unresolved label", gotoPC: &pc, used: s.Pos()}
			a.labels[l.name] = l
		}

		pc = l.gotoPC
		a.flow.putGoto(s.Pos(), l.name, a.block)

	case token.FALLTHROUGH:
		a.diag("fallthrough outside switch")
		return

	default:
		log.Panic("Unexpected branch token %v", s.Tok)
	}

	a.flow.put1(false, pc)
	a.push(func(v *Thread) { v.pc = *pc })
}

func (a *stmtCompiler) compileBlockStmt(s *ast.BlockStmt) {
	bc := a.enterChild()
	bc.compileStmts(s)
	bc.exit()
}

func (a *stmtCompiler) compileIfStmt(s *ast.IfStmt) {
	// The scope of any variables declared by [the init] statement
	// extends to the end of the "if" statement and the variables
	// are initialized once before the statement is entered.
	//
	// XXX(Spec) What this really wants to say is that there's an
	// implicit scope wrapping every if, for, and switch
	// statement.  This is subtly different from what it actually
	// says when there's a non-block else clause, because that
	// else claus has to execute in a scope that is *not* the
	// surrounding scope.
	bc := a.enterChild()
	defer bc.exit()

	// Compile init statement, if any
	if s.Init != nil {
		bc.compileStmt(s.Init)
	}

	elsePC := badPC
	endPC := badPC

	// Compile condition, if any.  If there is no condition, we
	// fall through to the body.
	if s.Cond != nil {
		e := bc.compileExpr(bc.block, false, s.Cond)
		switch {
		case e == nil:
			// Error reported by compileExpr
		case !e.t.isBoolean():
			e.diag("'if' condition must be boolean\n\t%v", e.t)
		default:
			eval := e.asBool()
			a.flow.put1(true, &elsePC)
			a.push(func(t *Thread) {
				if !eval(t) {
					t.pc = elsePC
				}
			})
		}
	}

	// Compile body
	body := bc.enterChild()
	body.compileStmts(s.Body)
	body.exit()

	// Compile else
	if s.Else != nil {
		// Skip over else if we executed the body
		a.flow.put1(false, &endPC)
		a.push(func(v *Thread) { v.pc = endPC })
		elsePC = a.nextPC()
		bc.compileStmt(s.Else)
	} else {
		elsePC = a.nextPC()
	}
	endPC = a.nextPC()
}

func (a *stmtCompiler) compileSwitchStmt(s *ast.SwitchStmt) {
	// Create implicit scope around switch
	bc := a.enterChild()
	defer bc.exit()

	// Compile init statement, if any
	if s.Init != nil {
		bc.compileStmt(s.Init)
	}

	// Compile condition, if any, and extract its effects
	var cond *expr
	condbc := bc.enterChild()
	if s.Tag != nil {
		e := condbc.compileExpr(condbc.block, false, s.Tag)
		if e != nil {
			var effect func(*Thread)
			effect, cond = e.extractEffect(condbc.block, "switch")
			a.push(effect)
		}
	}

	// Count cases
	ncases := 0
	hasDefault := false
	for _, c := range s.Body.List {
		clause, ok := c.(*ast.CaseClause)
		if !ok {
			a.diagAt(clause.Pos(), "switch statement must contain case clauses")
			continue
		}
		if clause.Values == nil {
			if hasDefault {
				a.diagAt(clause.Pos(), "switch statement contains more than one default case")
			}
			hasDefault = true
		} else {
			ncases += len(clause.Values)
		}
	}

	// Compile case expressions
	cases := make([]func(*Thread) bool, ncases)
	i := 0
	for _, c := range s.Body.List {
		clause, ok := c.(*ast.CaseClause)
		if !ok {
			continue
		}
		for _, v := range clause.Values {
			e := condbc.compileExpr(condbc.block, false, v)
			switch {
			case e == nil:
				// Error reported by compileExpr
			case cond == nil && !e.t.isBoolean():
				a.diagAt(v.Pos(), "'case' condition must be boolean")
			case cond == nil:
				cases[i] = e.asBool()
			case cond != nil:
				// Create comparison
				// TOOD(austin) This produces bad error messages
				compare := e.compileBinaryExpr(token.EQL, cond, e)
				if compare != nil {
					cases[i] = compare.asBool()
				}
			}
			i++
		}
	}

	// Emit condition
	casePCs := make([]*uint, ncases+1)
	endPC := badPC

	a.flow.put(false, false, casePCs)
	a.push(func(t *Thread) {
		for i, c := range cases {
			if c(t) {
				t.pc = *casePCs[i]
				return
			}
		}
		t.pc = *casePCs[ncases]
	})
	condbc.exit()

	// Compile cases
	i = 0
	for _, c := range s.Body.List {
		clause, ok := c.(*ast.CaseClause)
		if !ok {
			continue
		}

		// Save jump PC's
		pc := a.nextPC()
		if clause.Values != nil {
			for _ = range clause.Values {
				casePCs[i] = &pc
				i++
			}
		} else {
			// Default clause
			casePCs[ncases] = &pc
		}

		// Compile body
		fall := false
		for j, s := range clause.Body {
			if br, ok := s.(*ast.BranchStmt); ok && br.Tok == token.FALLTHROUGH {
				// println("Found fallthrough");
				// It may be used only as the final
				// non-empty statement in a case or
				// default clause in an expression
				// "switch" statement.
				for _, s2 := range clause.Body[j+1:] {
					// XXX(Spec) 6g also considers
					// empty blocks to be empty
					// statements.
					if _, ok := s2.(*ast.EmptyStmt); !ok {
						a.diagAt(s.Pos(), "fallthrough statement must be final statement in case")
						break
					}
				}
				fall = true
			} else {
				bc.compileStmt(s)
			}
		}
		// Jump out of switch, unless there was a fallthrough
		if !fall {
			a.flow.put1(false, &endPC)
			a.push(func(v *Thread) { v.pc = endPC })
		}
	}

	// Get end PC
	endPC = a.nextPC()
	if !hasDefault {
		casePCs[ncases] = &endPC
	}
}

func (a *stmtCompiler) compileForStmt(s *ast.ForStmt) {
	// Wrap the entire for in a block.
	bc := a.enterChild()
	defer bc.exit()

	// Compile init statement, if any
	if s.Init != nil {
		bc.compileStmt(s.Init)
	}

	bodyPC := badPC
	postPC := badPC
	checkPC := badPC
	endPC := badPC

	// Jump to condition check.  We generate slightly less code by
	// placing the condition check after the body.
	a.flow.put1(false, &checkPC)
	a.push(func(v *Thread) { v.pc = checkPC })

	// Compile body
	bodyPC = a.nextPC()
	body := bc.enterChild()
	if a.stmtLabel != nil {
		body.label = a.stmtLabel
	} else {
		body.label = &label{resolved: s.Pos()}
	}
	body.label.desc = "for loop"
	body.label.breakPC = &endPC
	body.label.continuePC = &postPC
	body.compileStmts(s.Body)
	body.exit()

	// Compile post, if any
	postPC = a.nextPC()
	if s.Post != nil {
		// TODO(austin) Does the parser disallow short
		// declarations in s.Post?
		bc.compileStmt(s.Post)
	}

	// Compile condition check, if any
	checkPC = a.nextPC()
	if s.Cond == nil {
		// If the condition is absent, it is equivalent to true.
		a.flow.put1(false, &bodyPC)
		a.push(func(v *Thread) { v.pc = bodyPC })
	} else {
		e := bc.compileExpr(bc.block, false, s.Cond)
		switch {
		case e == nil:
			// Error reported by compileExpr
		case !e.t.isBoolean():
			a.diag("'for' condition must be boolean\n\t%v", e.t)
		default:
			eval := e.asBool()
			a.flow.put1(true, &bodyPC)
			a.push(func(t *Thread) {
				if eval(t) {
					t.pc = bodyPC
				}
			})
		}
	}

	endPC = a.nextPC()
}

/*
 * Block compiler
 */

func (a *blockCompiler) compileStmt(s ast.Stmt) {
	sc := &stmtCompiler{a, s.Pos(), nil}
	sc.compile(s)
}

func (a *blockCompiler) compileStmts(block *ast.BlockStmt) {
	for _, sub := range block.List {
		a.compileStmt(sub)
	}
}

func (a *blockCompiler) enterChild() *blockCompiler {
	block := a.block.enterChild()
	return &blockCompiler{
		funcCompiler: a.funcCompiler,
		block:        block,
		parent:       a,
	}
}

func (a *blockCompiler) exit() { a.block.exit() }

/*
 * Function compiler
 */

func (a *compiler) compileFunc(b *block, decl *FuncDecl, body *ast.BlockStmt) func(*Thread) Func {
	// Create body scope
	//
	// The scope of a parameter or result is the body of the
	// corresponding function.
	bodyScope := b.ChildScope()
	defer bodyScope.exit()
	for i, t := range decl.Type.In {
		if decl.InNames[i] != nil {
			bodyScope.DefineVar(decl.InNames[i].Name, decl.InNames[i].Pos(), t)
		} else {
			bodyScope.DefineTemp(t)
		}
	}
	for i, t := range decl.Type.Out {
		if decl.OutNames[i] != nil {
			bodyScope.DefineVar(decl.OutNames[i].Name, decl.OutNames[i].Pos(), t)
		} else {
			bodyScope.DefineTemp(t)
		}
	}

	// Create block context
	cb := newCodeBuf()
	fc := &funcCompiler{
		compiler:     a,
		fnType:       decl.Type,
		outVarsNamed: len(decl.OutNames) > 0 && decl.OutNames[0] != nil,
		codeBuf:      cb,
		flow:         newFlowBuf(cb),
		labels:       make(map[string]*label),
	}
	bc := &blockCompiler{
		funcCompiler: fc,
		block:        bodyScope.block,
	}

	// Compile body
	nerr := a.numError()
	bc.compileStmts(body)
	fc.checkLabels()
	if nerr != a.numError() {
		return nil
	}

	// Check that the body returned if necessary.  We only check
	// this if there were no errors compiling the body.
	if len(decl.Type.Out) > 0 && fc.flow.reachesEnd(0) {
		// XXX(Spec) Not specified.
		a.diagAt(body.Rbrace, "function ends without a return statement")
		return nil
	}

	code := fc.get()
	maxVars := bodyScope.maxVars
	return func(t *Thread) Func { return &evalFunc{t.f, maxVars, code} }
}

// Checks that labels were resolved and that all jumps obey scoping
// rules.  Reports an error and set fc.err if any check fails.
func (a *funcCompiler) checkLabels() {
	nerr := a.numError()
	for _, l := range a.labels {
		if !l.resolved.IsValid() {
			a.diagAt(l.used, "label %s not defined", l.name)
		}
	}
	if nerr != a.numError() {
		// Don't check scopes if we have unresolved labels
		return
	}

	// Executing the "goto" statement must not cause any variables
	// to come into scope that were not already in scope at the
	// point of the goto.
	a.flow.gotosObeyScopes(a.compiler)
}
