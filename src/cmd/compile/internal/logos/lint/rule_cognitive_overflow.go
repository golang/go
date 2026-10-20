// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lint

import (
	"strconv"
	"strings"

	"cmd/compile/internal/syntax"
)

// cognitiveOverflow is complexity::cognitive_overflow (SPEC Sec 1.2): a
// function that exceeds one of four limits of the logos.toml
// [lints.thresholds] table: max_nesting_depth, max_function_params,
// max_function_lines, or max_cognitive_complexity. A function declaration, a
// function literal, and a loGo/s lambda are each a function, measured on
// their own: a nested function body adds nothing to the function around it,
// though its lines belong to that function's line count. A function with
// more than one exceeded limit gets one finding that names them all.
//
// The nesting structures are if, for, switch, and select, with the loGo/s
// const if, const switch, and switch expression. The nesting depth of a
// function is the deepest level of nested structures, counting the deepest
// structure itself, so a function with a for inside an if inside a for has
// depth 3. The cognitive score adds 1 plus the number of structures around
// the point for each of: a structure, a case with a condition (not a default
// clause), a goto, and a labeled break or continue; a case sits at the level
// of its switch, as gofmt writes it. An else if continues its if at the same
// level and adds the same as an if there. Each change between && and || in
// an unparenthesized chain adds 1, so a && b && c adds nothing and
// a && b || c adds 1. The line count runs from the func line through the
// closing brace, blank and comment lines included. The parameter count is
// the number of declared parameters, not the receiver.
var cognitiveOverflow = Rule{
	ID:    "complexity::cognitive_overflow",
	Group: Complexity,
	Go:    true,
	Check: checkCognitiveOverflow,
}

// checkCognitiveOverflow is the Check of cognitiveOverflow.
func checkCognitiveOverflow(pass *Pass) {
	limits := pass.Policy.Thresholds
	for _, f := range pass.Files {
		syntax.Inspect(f, func(n syntax.Node) bool {
			fn, ok := cogFunctionOf(n)
			if !ok {
				return true
			}
			if over := fn.measure().exceeded(limits); len(over) > 0 {
				pass.Report(Diagnostic{
					Pos:     fn.pos,
					End:     fn.end,
					Message: fn.name + " is too complex: " + strings.Join(over, ", "),
				})
			}
			return true
		})
	}
}

// cogFunction is one function body the rule measures. pos and end bound a
// finding, and first and last bound the lines of the function.
type cogFunction struct {
	name        string
	pos, end    syntax.Pos
	params      int
	body        syntax.Node
	first, last uint
}

// cogFunctionOf returns the function n declares, if n is one with a body.
func cogFunctionOf(n syntax.Node) (cogFunction, bool) {
	switch n := n.(type) {
	case *syntax.FuncDecl:
		if n.Body == nil {
			return cogFunction{}, false
		}
		return cogFunction{
			name: "function " + n.Name.Value, pos: n.Name.Pos(), end: cogAfterBrace(n.Body), params: len(n.Type.ParamList),
			body: n.Body, first: syntax.StartPos(n).Line(), last: n.Body.Rbrace.Line(),
		}, true
	case *syntax.FuncLit:
		return cogFunction{
			name: "function literal", pos: n.Pos(), end: cogAfterBrace(n.Body), params: len(n.Type.ParamList),
			body: n.Body, first: syntax.StartPos(n).Line(), last: n.Body.Rbrace.Line(),
		}, true
	case *syntax.LambdaExpr:
		return cogFunction{
			name: "function literal", pos: n.Pos(), end: syntax.EndPos(n), params: len(n.Params),
			body: n.Body, first: syntax.StartPos(n).Line(), last: syntax.EndPos(n).Line(),
		}, true
	}
	return cogFunction{}, false
}

// cogAfterBrace returns the position just after the closing brace of b, which
// syntax.EndPos reports at the brace itself.
func cogAfterBrace(b *syntax.BlockStmt) syntax.Pos {
	return syntax.MakePos(b.Rbrace.Base(), b.Rbrace.Line(), b.Rbrace.Col()+1)
}

// cogMetrics are the four measures of one function.
type cogMetrics struct {
	depth, params, lines, score int
}

// measure computes the metrics of fn.
func (fn cogFunction) measure() cogMetrics {
	c := &cogWalk{chained: make(map[syntax.Node]bool), inChain: make(map[*syntax.Operation]bool)}
	syntax.Inspect(fn.body, c.visit)
	return cogMetrics{depth: c.depth, params: fn.params, lines: int(fn.last-fn.first) + 1, score: c.score}
}

// exceeded describes each measure above its limit, in the order of the SPEC
// sentence.
func (m cogMetrics) exceeded(limits Thresholds) []string {
	var out []string
	for _, c := range []struct {
		name       string
		got, limit int
	}{
		{"nesting depth", m.depth, limits.MaxNestingDepth},
		{"parameter count", m.params, limits.MaxFunctionParams},
		{"line count", m.lines, limits.MaxFunctionLines},
		{"cognitive score", m.score, limits.MaxCognitiveComplexity},
	} {
		if c.got > c.limit {
			out = append(out, c.name+" "+strconv.Itoa(c.got)+" exceeds "+strconv.Itoa(c.limit))
		}
	}
	return out
}

// cogWalk is the walk of one function body. frames holds one entry per
// open node and says whether that node opened a nesting level; level counts
// the open levels and depth is the deepest level reached. chained holds the
// else-if statements, and inChain the operations that belong to a boolean
// chain already counted from its root.
type cogWalk struct {
	frames  []bool
	level   int
	depth   int
	score   int
	chained map[syntax.Node]bool
	inChain map[*syntax.Operation]bool
}

// visit is the syntax.Inspect callback. It leaves nested functions to their
// own measurement.
func (c *cogWalk) visit(n syntax.Node) bool {
	switch n.(type) {
	case nil:
		c.leave()
		return false
	case *syntax.FuncLit, *syntax.LambdaExpr:
		return false
	}
	c.enter(c.weigh(n))
	return true
}

// enter opens a frame for a node, and a nesting level when opens is set.
func (c *cogWalk) enter(opens bool) {
	c.frames = append(c.frames, opens)
	if opens {
		c.level++
		c.depth = max(c.depth, c.level)
	}
}

// leave closes the innermost frame.
func (c *cogWalk) leave() {
	last := len(c.frames) - 1
	if c.frames[last] {
		c.level--
	}
	c.frames = c.frames[:last]
}

// weigh adds the score of n and reports whether n opens a nesting level.
func (c *cogWalk) weigh(n syntax.Node) bool {
	switch n := n.(type) {
	case *syntax.IfStmt:
		return c.branch(n, n.Else)
	case *syntax.ConstIfStmt:
		return c.branch(n, n.Else)
	case *syntax.ForStmt, *syntax.SwitchStmt, *syntax.SelectStmt, *syntax.SwitchExpr:
		return c.structure(1)
	case *syntax.ConstSwitchStmt:
		return c.structure(1 + len(n.Cases))
	}
	c.weighLeaf(n)
	return false
}

// structure adds points scores for a nesting structure and its cases, each
// worth 1 plus the levels around the structure, and reports true.
func (c *cogWalk) structure(points int) bool {
	c.score += points * (1 + c.level)
	return true
}

// branch scores an if or const if and marks the if that follows its else.
// An else if adds the score of an if at the level of the first if and opens
// no level of its own.
func (c *cogWalk) branch(n, els syntax.Node) bool {
	if next := cogElseIf(els); next != nil {
		c.chained[next] = true
	}
	if c.chained[n] {
		c.score += c.level
		return false
	}
	return c.structure(1)
}

// cogElseIf returns s when it is an if or const if, which an else holds for an
// else if, and nil otherwise.
func cogElseIf(s syntax.Node) syntax.Node {
	switch s.(type) {
	case *syntax.IfStmt, *syntax.ConstIfStmt:
		return s
	}
	return nil
}

// weighLeaf scores a node that opens no level: a case with a condition, a
// goto or labeled jump, or the root of a boolean chain.
func (c *cogWalk) weighLeaf(n syntax.Node) {
	switch {
	case cogConditionedCase(n):
		c.score += c.level
	case cogJumps(n):
		c.score += 1 + c.level
	default:
		if op, ok := n.(*syntax.Operation); ok && !c.inChain[op] {
			c.score += cogChanges(c.chainOps(op))
		}
	}
}

// cogConditionedCase reports whether n is a case clause that has a condition,
// as a default clause does not.
func cogConditionedCase(n syntax.Node) bool {
	switch n := n.(type) {
	case *syntax.CaseClause:
		return n.Cases != nil
	case *syntax.CommClause:
		return n.Comm != nil
	case *syntax.ArmClause:
		return n.Cases != nil
	}
	return false
}

// cogJumps reports whether n is a goto, or a break or continue with a label.
func cogJumps(n syntax.Node) bool {
	b, ok := n.(*syntax.BranchStmt)
	return ok && (b.Tok == syntax.Goto || b.Label != nil && (b.Tok == syntax.Break || b.Tok == syntax.Continue))
}

// chainOps returns the operators of the unparenthesized && and || chain that
// e roots, in source order, and marks each of its operations as counted.
func (c *cogWalk) chainOps(e syntax.Expr) []syntax.Operator {
	op, ok := e.(*syntax.Operation)
	if !ok || op.Y == nil || (op.Op != syntax.AndAnd && op.Op != syntax.OrOr) {
		return nil
	}
	c.inChain[op] = true
	left := c.chainOps(op.X)
	return append(append(left, op.Op), c.chainOps(op.Y)...)
}

// cogChanges counts the places where an operator differs from the one before.
func cogChanges(ops []syntax.Operator) int {
	n := 0
	for i := 1; i < len(ops); i++ {
		if ops[i] != ops[i-1] {
			n++
		}
	}
	return n
}
