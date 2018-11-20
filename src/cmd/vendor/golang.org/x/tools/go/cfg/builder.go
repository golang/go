// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cfg

// This file implements the CFG construction pass.

import (
	"fmt"
	"go/ast"
	"go/token"
)

type builder struct {
	cfg       *CFG
	mayReturn func(*ast.CallExpr) bool
	current   *Block
	lblocks   map[*ast.Object]*lblock // labeled blocks
	targets   *targets                // linked stack of branch targets
}

func (b *builder) stmt(_s ast.Stmt) {
	// The label of the current statement.  If non-nil, its _goto
	// target is always set; its _break and _continue are set only
	// within the body of switch/typeswitch/select/for/range.
	// It is effectively an additional default-nil parameter of stmt().
	var label *lblock
start:
	switch s := _s.(type) {
	case *ast.BadStmt,
		*ast.SendStmt,
		*ast.IncDecStmt,
		*ast.GoStmt,
		*ast.DeferStmt,
		*ast.EmptyStmt,
		*ast.AssignStmt:
		// No effect on control flow.
		b.add(s)

	case *ast.ExprStmt:
		b.add(s)
		if call, ok := s.X.(*ast.CallExpr); ok && !b.mayReturn(call) {
			// Calls to panic, os.Exit, etc, never return.
			b.current = b.newBlock("unreachable.call")
		}

	case *ast.DeclStmt:
		// Treat each var ValueSpec as a separate statement.
		d := s.Decl.(*ast.GenDecl)
		if d.Tok == token.VAR {
			for _, spec := range d.Specs {
				if spec, ok := spec.(*ast.ValueSpec); ok {
					b.add(spec)
				}
			}
		}

	case *ast.LabeledStmt:
		label = b.labeledBlock(s.Label)
		b.jump(label._goto)
		b.current = label._goto
		_s = s.Stmt
		goto start // effectively: tailcall stmt(g, s.Stmt, label)

	case *ast.ReturnStmt:
		b.add(s)
		b.current = b.newBlock("unreachable.return")

	case *ast.BranchStmt:
		b.branchStmt(s)

	case *ast.BlockStmt:
		b.stmtList(s.List)

	case *ast.IfStmt:
		if s.Init != nil {
			b.stmt(s.Init)
		}
		then := b.newBlock("if.then")
		done := b.newBlock("if.done")
		_else := done
		if s.Else != nil {
			_else = b.newBlock("if.else")
		}
		b.add(s.Cond)
		b.ifelse(then, _else)
		b.current = then
		b.stmt(s.Body)
		b.jump(done)

		if s.Else != nil {
			b.current = _else
			b.stmt(s.Else)
			b.jump(done)
		}

		b.current = done

	case *ast.SwitchStmt:
		b.switchStmt(s, label)

	case *ast.TypeSwitchStmt:
		b.typeSwitchStmt(s, label)

	case *ast.SelectStmt:
		b.selectStmt(s, label)

	case *ast.ForStmt:
		b.forStmt(s, label)

	case *ast.RangeStmt:
		b.rangeStmt(s, label)

	default:
		panic(fmt.Sprintf("unexpected statement kind: %T", s))
	}
}

func (b *builder) stmtList(list []ast.Stmt) {
	for _, s := range list {
		b.stmt(s)
	}
}

func (b *builder) branchStmt(s *ast.BranchStmt) {
	var block *Block
	switch s.Tok {
	case token.BREAK:
		if s.Label != nil {
			if lb := b.labeledBlock(s.Label); lb != nil {
				block = lb._break
			}
		} else {
			for t := b.targets; t != nil && block == nil; t = t.tail {
				block = t._break
			}
		}

	case token.CONTINUE:
		if s.Label != nil {
			if lb := b.labeledBlock(s.Label); lb != nil {
				block = lb._continue
			}
		} else {
			for t := b.targets; t != nil && block == nil; t = t.tail {
				block = t._continue
			}
		}

	case token.FALLTHROUGH:
		for t := b.targets; t != nil; t = t.tail {
			block = t._fallthrough
		}

	case token.GOTO:
		if s.Label != nil {
			block = b.labeledBlock(s.Label)._goto
		}
	}
	if block == nil {
		block = b.newBlock("undefined.branch")
	}
	b.jump(block)
	b.current = b.newBlock("unreachable.branch")
}

func (b *builder) switchStmt(s *ast.SwitchStmt, label *lblock) {
	if s.Init != nil {
		b.stmt(s.Init)
	}
	if s.Tag != nil {
		b.add(s.Tag)
	}
	done := b.newBlock("switch.done")
	if label != nil {
		label._break = done
	}
	// We pull the default case (if present) down to the end.
	// But each fallthrough label must point to the next
	// body block in source order, so we preallocate a
	// body block (fallthru) for the next case.
	// Unfortunately this makes for a confusing block order.
	var defaultBody *[]ast.Stmt
	var defaultFallthrough *Block
	var fallthru, defaultBlock *Block
	ncases := len(s.Body.List)
	for i, clause := range s.Body.List {
		body := fallthru
		if body == nil {
			body = b.newBlock("switch.body") // first case only
		}

		// Preallocate body block for the next case.
		fallthru = done
		if i+1 < ncases {
			fallthru = b.newBlock("switch.body")
		}

		cc := clause.(*ast.CaseClause)
		if cc.List == nil {
			// Default case.
			defaultBody = &cc.Body
			defaultFallthrough = fallthru
			defaultBlock = body
			continue
		}

		var nextCond *Block
		for _, cond := range cc.List {
			nextCond = b.newBlock("switch.next")
			b.add(cond) // one half of the tag==cond condition
			b.ifelse(body, nextCond)
			b.current = nextCond
		}
		b.current = body
		b.targets = &targets{
			tail:         b.targets,
			_break:       done,
			_fallthrough: fallthru,
		}
		b.stmtList(cc.Body)
		b.targets = b.targets.tail
		b.jump(done)
		b.current = nextCond
	}
	if defaultBlock != nil {
		b.jump(defaultBlock)
		b.current = defaultBlock
		b.targets = &targets{
			tail:         b.targets,
			_break:       done,
			_fallthrough: defaultFallthrough,
		}
		b.stmtList(*defaultBody)
		b.targets = b.targets.tail
	}
	b.jump(done)
	b.current = done
}

func (b *builder) typeSwitchStmt(s *ast.TypeSwitchStmt, label *lblock) {
	if s.Init != nil {
		b.stmt(s.Init)
	}
	if s.Assign != nil {
		b.add(s.Assign)
	}

	done := b.newBlock("typeswitch.done")
	if label != nil {
		label._break = done
	}
	var default_ *ast.CaseClause
	for _, clause := range s.Body.List {
		cc := clause.(*ast.CaseClause)
		if cc.List == nil {
			default_ = cc
			continue
		}
		body := b.newBlock("typeswitch.body")
		var next *Block
		for _, casetype := range cc.List {
			next = b.newBlock("typeswitch.next")
			// casetype is a type, so don't call b.add(casetype).
			// This block logically contains a type assertion,
			// x.(casetype), but it's unclear how to represent x.
			_ = casetype
			b.ifelse(body, next)
			b.current = next
		}
		b.current = body
		b.typeCaseBody(cc, done)
		b.current = next
	}
	if default_ != nil {
		b.typeCaseBody(default_, done)
	} else {
		b.jump(done)
	}
	b.current = done
}

func (b *builder) typeCaseBody(cc *ast.CaseClause, done *Block) {
	b.targets = &targets{
		tail:   b.targets,
		_break: done,
	}
	b.stmtList(cc.Body)
	b.targets = b.targets.tail
	b.jump(done)
}

func (b *builder) selectStmt(s *ast.SelectStmt, label *lblock) {
	// First evaluate channel expressions.
	// TODO(adonovan): fix: evaluate only channel exprs here.
	for _, clause := range s.Body.List {
		if comm := clause.(*ast.CommClause).Comm; comm != nil {
			b.stmt(comm)
		}
	}

	done := b.newBlock("select.done")
	if label != nil {
		label._break = done
	}

	var defaultBody *[]ast.Stmt
	for _, cc := range s.Body.List {
		clause := cc.(*ast.CommClause)
		if clause.Comm == nil {
			defaultBody = &clause.Body
			continue
		}
		body := b.newBlock("select.body")
		next := b.newBlock("select.next")
		b.ifelse(body, next)
		b.current = body
		b.targets = &targets{
			tail:   b.targets,
			_break: done,
		}
		switch comm := clause.Comm.(type) {
		case *ast.ExprStmt: // <-ch
			// nop
		case *ast.AssignStmt: // x := <-states[state].Chan
			b.add(comm.Lhs[0])
		}
		b.stmtList(clause.Body)
		b.targets = b.targets.tail
		b.jump(done)
		b.current = next
	}
	if defaultBody != nil {
		b.targets = &targets{
			tail:   b.targets,
			_break: done,
		}
		b.stmtList(*defaultBody)
		b.targets = b.targets.tail
		b.jump(done)
	}
	b.current = done
}

func (b *builder) forStmt(s *ast.ForStmt, label *lblock) {
	//	...init...
	//      jump loop
	// loop:
	//      if cond goto body else done
	// body:
	//      ...body...
	//      jump post
	// post:				 (target of continue)
	//      ...post...
	//      jump loop
	// done:                                 (target of break)
	if s.Init != nil {
		b.stmt(s.Init)
	}
	body := b.newBlock("for.body")
	done := b.newBlock("for.done") // target of 'break'
	loop := body                   // target of back-edge
	if s.Cond != nil {
		loop = b.newBlock("for.loop")
	}
	cont := loop // target of 'continue'
	if s.Post != nil {
		cont = b.newBlock("for.post")
	}
	if label != nil {
		label._break = done
		label._continue = cont
	}
	b.jump(loop)
	b.current = loop
	if loop != body {
		b.add(s.Cond)
		b.ifelse(body, done)
		b.current = body
	}
	b.targets = &targets{
		tail:      b.targets,
		_break:    done,
		_continue: cont,
	}
	b.stmt(s.Body)
	b.targets = b.targets.tail
	b.jump(cont)

	if s.Post != nil {
		b.current = cont
		b.stmt(s.Post)
		b.jump(loop) // back-edge
	}
	b.current = done
}

func (b *builder) rangeStmt(s *ast.RangeStmt, label *lblock) {
	b.add(s.X)

	if s.Key != nil {
		b.add(s.Key)
	}
	if s.Value != nil {
		b.add(s.Value)
	}

	//      ...
	// loop:                                   (target of continue)
	// 	if ... goto body else done
	// body:
	//      ...
	// 	jump loop
	// done:                                   (target of break)

	loop := b.newBlock("range.loop")
	b.jump(loop)
	b.current = loop

	body := b.newBlock("range.body")
	done := b.newBlock("range.done")
	b.ifelse(body, done)
	b.current = body

	if label != nil {
		label._break = done
		label._continue = loop
	}
	b.targets = &targets{
		tail:      b.targets,
		_break:    done,
		_continue: loop,
	}
	b.stmt(s.Body)
	b.targets = b.targets.tail
	b.jump(loop) // back-edge
	b.current = done
}

// -------- helpers --------

// Destinations associated with unlabeled for/switch/select stmts.
// We push/pop one of these as we enter/leave each construct and for
// each BranchStmt we scan for the innermost target of the right type.
//
type targets struct {
	tail         *targets // rest of stack
	_break       *Block
	_continue    *Block
	_fallthrough *Block
}

// Destinations associated with a labeled block.
// We populate these as labels are encountered in forward gotos or
// labeled statements.
//
type lblock struct {
	_goto     *Block
	_break    *Block
	_continue *Block
}

// labeledBlock returns the branch target associated with the
// specified label, creating it if needed.
//
func (b *builder) labeledBlock(label *ast.Ident) *lblock {
	lb := b.lblocks[label.Obj]
	if lb == nil {
		lb = &lblock{_goto: b.newBlock(label.Name)}
		if b.lblocks == nil {
			b.lblocks = make(map[*ast.Object]*lblock)
		}
		b.lblocks[label.Obj] = lb
	}
	return lb
}

// newBlock appends a new unconnected basic block to b.cfg's block
// slice and returns it.
// It does not automatically become the current block.
// comment is an optional string for more readable debugging output.
func (b *builder) newBlock(comment string) *Block {
	g := b.cfg
	block := &Block{
		Index:   int32(len(g.Blocks)),
		comment: comment,
	}
	block.Succs = block.succs2[:0]
	g.Blocks = append(g.Blocks, block)
	return block
}

func (b *builder) add(n ast.Node) {
	b.current.Nodes = append(b.current.Nodes, n)
}

// jump adds an edge from the current block to the target block,
// and sets b.current to nil.
func (b *builder) jump(target *Block) {
	b.current.Succs = append(b.current.Succs, target)
	b.current = nil
}

// ifelse emits edges from the current block to the t and f blocks,
// and sets b.current to nil.
func (b *builder) ifelse(t, f *Block) {
	b.current.Succs = append(b.current.Succs, t, f)
	b.current = nil
}
