// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package deadcode

import (
	"go/constant"
	"go/token"

	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
)

func Func(fn *ir.Func) {
	stmts(&fn.Body)

	if len(fn.Body) == 0 {
		return
	}

	for _, n := range fn.Body {
		if len(n.Init()) > 0 {
			return
		}
		switch n.Op() {
		case ir.OIF:
			n := n.(*ir.IfStmt)
			if !ir.IsConst(n.Cond, constant.Bool) || len(n.Body) > 0 || len(n.Else) > 0 {
				return
			}
		case ir.OFOR:
			n := n.(*ir.ForStmt)
			if !ir.IsConst(n.Cond, constant.Bool) || ir.BoolVal(n.Cond) {
				return
			}
		default:
			return
		}
	}

	ir.VisitList(fn.Body, markHiddenClosureDead)
	fn.Body = []ir.Node{ir.NewBlockStmt(base.Pos, nil)}
}

func stmts(nn *ir.Nodes) {
	var lastLabel = -1
	for i, n := range *nn {
		if n != nil && n.Op() == ir.OLABEL {
			lastLabel = i
		}
	}
	for i, n := range *nn {
		// Cut is set to true when all nodes after i'th position
		// should be removed.
		// In other words, it marks whole slice "tail" as dead.
		cut := false
		if n == nil {
			continue
		}
		if n.Op() == ir.OIF {
			n := n.(*ir.IfStmt)
			n.Cond = expr(n.Cond)
			if ir.IsConst(n.Cond, constant.Bool) {
				var body ir.Nodes
				if ir.BoolVal(n.Cond) {
					ir.VisitList(n.Else, markHiddenClosureDead)
					n.Else = ir.Nodes{}
					body = n.Body
				} else {
					ir.VisitList(n.Body, markHiddenClosureDead)
					n.Body = ir.Nodes{}
					body = n.Else
				}
				// If "then" or "else" branch ends with panic or return statement,
				// it is safe to remove all statements after this node.
				// isterminating is not used to avoid goto-related complications.
				// We must be careful not to deadcode-remove labels, as they
				// might be the target of a goto. See issue 28616.
				if body := body; len(body) != 0 {
					switch body[(len(body) - 1)].Op() {
					case ir.ORETURN, ir.OTAILCALL, ir.OPANIC:
						if i > lastLabel {
							cut = true
						}
					}
				}
			}
		}
		if n.Op() == ir.OSWITCH {
			n := n.(*ir.SwitchStmt)
			// Use a closure wrapper here so we can use "return" to abort the analysis.
			func() {
				if n.Tag != nil && n.Tag.Op() == ir.OTYPESW {
					return // no special type-switch case yet.
				}
				var x constant.Value // value we're switching on
				if n.Tag != nil {
					if ir.ConstType(n.Tag) == constant.Unknown {
						return
					}
					x = n.Tag.Val()
				} else {
					x = constant.MakeBool(true) // switch { ... }  =>  switch true { ... }
				}
				var def *ir.CaseClause
				for _, cas := range n.Cases {
					if len(cas.List) == 0 { // default case
						def = cas
						continue
					}
					for _, c := range cas.List {
						if ir.ConstType(c) == constant.Unknown {
							return // can't statically tell if it matches or not - give up.
						}
						if constant.Compare(x, token.EQL, c.Val()) {
							for _, n := range cas.Body {
								if n.Op() == ir.OFALL {
									return // fallthrough makes it complicated - abort.
								}
							}
							// This switch entry is the one that always triggers.
							for _, cas2 := range n.Cases {
								for _, c2 := range cas2.List {
									if cas2 != cas || c2 != c {
										ir.Visit(c2, markHiddenClosureDead)
									}
								}
								if cas2 != cas {
									ir.VisitList(cas2.Body, markHiddenClosureDead)
								}
							}

							cas.List[0] = c
							cas.List = cas.List[:1]
							n.Cases[0] = cas
							n.Cases = n.Cases[:1]
							return
						}
					}
				}
				if def != nil {
					for _, n := range def.Body {
						if n.Op() == ir.OFALL {
							return // fallthrough makes it complicated - abort.
						}
					}
					for _, cas := range n.Cases {
						if cas != def {
							ir.VisitList(cas.List, markHiddenClosureDead)
							ir.VisitList(cas.Body, markHiddenClosureDead)
						}
					}
					n.Cases[0] = def
					n.Cases = n.Cases[:1]
					return
				}

				// TODO: handle case bodies ending with panic/return as we do in the IF case above.

				// entire switch is a nop - no case ever triggers
				for _, cas := range n.Cases {
					ir.VisitList(cas.List, markHiddenClosureDead)
					ir.VisitList(cas.Body, markHiddenClosureDead)
				}
				n.Cases = n.Cases[:0]
			}()
		}

		if len(n.Init()) != 0 {
			stmts(n.(ir.InitNode).PtrInit())
		}
		switch n.Op() {
		case ir.OBLOCK:
			n := n.(*ir.BlockStmt)
			stmts(&n.List)
		case ir.OFOR:
			n := n.(*ir.ForStmt)
			stmts(&n.Body)
		case ir.OIF:
			n := n.(*ir.IfStmt)
			stmts(&n.Body)
			stmts(&n.Else)
		case ir.ORANGE:
			n := n.(*ir.RangeStmt)
			stmts(&n.Body)
		case ir.OSELECT:
			n := n.(*ir.SelectStmt)
			for _, cas := range n.Cases {
				stmts(&cas.Body)
			}
		case ir.OSWITCH:
			n := n.(*ir.SwitchStmt)
			for _, cas := range n.Cases {
				stmts(&cas.Body)
			}
		}

		if cut {
			ir.VisitList((*nn)[i+1:len(*nn)], markHiddenClosureDead)
			*nn = (*nn)[:i+1]
			break
		}
	}
}

func expr(n ir.Node) ir.Node {
	// Perform dead-code elimination on short-circuited boolean
	// expressions involving constants with the intent of
	// producing a constant 'if' condition.
	switch n.Op() {
	case ir.OANDAND:
		n := n.(*ir.LogicalExpr)
		n.X = expr(n.X)
		n.Y = expr(n.Y)
		if ir.IsConst(n.X, constant.Bool) {
			if ir.BoolVal(n.X) {
				return n.Y // true && x => x
			} else {
				return n.X // false && x => false
			}
		}
	case ir.OOROR:
		n := n.(*ir.LogicalExpr)
		n.X = expr(n.X)
		n.Y = expr(n.Y)
		if ir.IsConst(n.X, constant.Bool) {
			if ir.BoolVal(n.X) {
				return n.X // true || x => true
			} else {
				return n.Y // false || x => x
			}
		}
	}
	return n
}

func markHiddenClosureDead(n ir.Node) {
	if n.Op() != ir.OCLOSURE {
		return
	}
	clo := n.(*ir.ClosureExpr)
	if clo.Func.IsHiddenClosure() {
		clo.Func.SetIsDeadcodeClosure(true)
	}
	ir.VisitList(clo.Func.Body, markHiddenClosureDead)
}
