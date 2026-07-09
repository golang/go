// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package specexpr manipulates symbolic constraints on vector shapes.
//
// # Shapes
//
// A vector shape consists of a base type B (e.g., int, float), element width N
// (8, 16, 32, or 64), and a vector width W (128, 256, 512, or scalable). It
// also has a lane count L, which is the vector width / element width. These are
// written like "Int32x4" or "Int32w128" or, for a scalable vector, "Int32s"
// Masks are represented as vectors with a base type of "Mask", e.g.,
// "Mask32x4".
//
// Scalar shapes consist only of a base type and an element width, e.g.,
// "uint32".
//
// # Expressions
//
// Constraints are written as boolean expressions over shapes and a few basic
// types.
//
// The primary expressions are:
//
//   - Variable ([a-zA-Z]+), such as x or xL
//
//   - Integers ([0-9]+)
//
//   - Shapes, written in the form given above, but where each component can be
//     written as a bracketed expression, such as "Int32x{z/2}".
//
// These can be combined with operators * or /, or comparison operators =, >, <,
// >=, <=. Arithmetic operators bind more tightly than comparison operators.
//
// # Example
//
// Consider a DotProductPairs function that takes two vectors x and y that have
// the same shape and produces a vector z that has the same base type as x and
// y, but has half as many elements, each of double the width. This can be
// expressed as:
//
//	y=x
//	z={xB}{xN*2}x{xL/2}
//
// # Width rounding
//
// The minimum vector width is 128 bits. Sometimes, operations would naturally
// produce a width smaller than this, so hardware simply pads the vector out to
// 128 bits. Shapes implement this behavior. For example, consider a "convert to
// float32 operation" with constraints
//
//	z=Float32x{xL}
//
// If x is Float64x2, then z would naturally be Float32x2, but since this is
// only 64 bits, the shape is "rounded" up to Float32x4.
//
// # Limitations
//
// The solver is intentionally simple. See [Solver] for a description of its
// limitations. If you run up against its limitations, you're probably being too
// clever.
package specexpr

import (
	"cmp"
	"container/heap"
	"fmt"
	"io"
	"iter"
	"log"
	"maps"
	"slices"
	"strings"
)

// A Solver solves a set of constraints.
//
// This is a simple monotonic solver. It looks for a single order in which it
// can resolve all constraints, assertions of the form "var=expr" are treated as
// candidates for resolving the value of "var", and anything else is treated as
// a boolean check. Variables that appear only on the right hand side of
// assignments are "independent" and it will enumerate all possible values of
// these variables. It never tries to invert any formulas and refuses to solve a
// system with cycles. This is intentional to keep this solver fast: if your
// formulas are cyclic, you're doing something too complicated.
type Solver struct {
	vars    map[Variable][]any
	asserts []Expr
	tracer  *tracer
}

// SetTrace enables emitting a solver trace to w.
func (s *Solver) SetTrace(w io.Writer) {
	if w == nil {
		s.tracer = nil
	} else {
		s.tracer = &tracer{w: w}
	}
}

// Declare declares a variable and its domain.
//
// Any "int" values in domain will be converted to [Int].
func (s *Solver) Declare(v Variable, domain []any) {
	if s.vars == nil {
		s.vars = make(map[Variable][]any)
	}
	if _, ok := s.vars[v]; ok {
		panic(v + " redeclared")
	}
	for i, d := range domain {
		if d, ok := d.(int); ok {
			domain[i] = Int(d)
		}
	}
	s.vars[v] = domain
}

// Assign is a convenience for asserting that v=val.
func (s *Solver) Assign(v Variable, val Expr) Variable {
	s.Assert(&BinExpr{Op: OpEqual, X: v, Y: val})
	return v
}

// Assert asserts a boolean condition must be true.
func (s *Solver) Assert(cond Expr) {
	s.asserts = append(s.asserts, cond)
}

func (s *Solver) Fprint(w io.Writer) {
	for _, v := range slices.Sorted(maps.Keys(s.vars)) {
		fmt.Fprintf(w, "%s in %v\n", v, s.vars[v])
	}
	for _, expr := range s.asserts {
		fmt.Fprintf(w, "%s\n", expr)
	}
}

// Bindings is a set of variable values.
type Bindings struct {
	varNames map[Variable]int // Shared between all solutions
	vals     []any
}

// Get returns the value of v if resolved, or nil.
func (b *Bindings) Get(v Variable) any {
	vid, ok := b.varNames[v]
	if !ok || vid >= len(b.vals) {
		return nil
	}
	return b.vals[vid]
}

// All yields all variable bindings.
func (b *Bindings) All() iter.Seq2[Variable, any] {
	return func(yield func(Variable, any) bool) {
		for _, varName := range slices.Sorted(maps.Keys(b.varNames)) {
			vid := b.varNames[varName]
			if vid < len(b.vals) && !yield(varName, b.vals[vid]) {
				return
			}
		}
	}
}

func (b *Bindings) String() string {
	var buf strings.Builder
	buf.WriteByte('{')
	for v, val := range b.All() {
		if buf.Len() > 1 {
			buf.WriteByte(' ')
		}
		fmt.Fprintf(&buf, "%s=%v", v, val)
	}
	buf.WriteByte('}')
	return buf.String()
}

// Solve yields all satisfying assignments of the variables in s.
func (s *Solver) Solve() iter.Seq2[*Bindings, error] {
	steps, err := s.topoSort()
	if err != nil {
		return func(yield func(*Bindings, error) bool) {
			yield(nil, err)
		}
	}

	// Assign variable indexes
	varIDs := make(map[Variable]int)
	for _, step := range steps {
		if step.kind == solverStepCheck {
			continue
		}
		if _, ok := varIDs[step.bind]; ok {
			panic(fmt.Sprintf("variable %s resolved multiple times by solver sequence", step.bind))
		}
		varIDs[step.bind] = len(varIDs)
	}

	// If there are no solutions, then we report any evaluation errors. As soon
	// as we yield anything, we set this to nil to indicate that.
	errors := make(map[string]bool)
	addErr := func(err error) {
		if errors != nil {
			errors[err.Error()] = true
		}
	}

	// Walk solver steps
	b := Bindings{
		varNames: varIDs,
		vals:     make([]any, 0, len(varIDs)),
	}
	pop := func() {
		b.vals = b.vals[:len(b.vals)-1]
	}
	var visit func(steps []*solverStep, yield func(*Bindings, error) bool) bool
	visit = func(steps []*solverStep, yield func(*Bindings, error) bool) bool {
		if len(steps) == 0 {
			errors = nil // Discard any errors
			// Snapshot Bindings.
			s.tracer.sat()
			return yield(&Bindings{varNames: b.varNames, vals: slices.Clone(b.vals)}, nil)
		}
		step := steps[0]
		steps = steps[1:]
		switch step.kind {
		case solverStepAssign:
			val, err := step.expr.(*BinExpr).Y.eval(&b)
			if err == nil {
				if domain, ok := s.vars[step.bind]; ok && !slices.Contains(domain, val) {
					err = fmt.Errorf("cannot assign %s=%v: not in domain", step.bind, val)
				}
			}
			s.tracer.assign(step.expr, step.bind, val, err)
			if err != nil {
				addErr(err)
				return true
			}
			b.vals = append(b.vals, val)
			defer pop()
			return visit(steps, yield)

		case solverStepCheck:
			val, err := step.expr.eval(&b)
			s.tracer.check(step.expr, val, err)
			if err != nil {
				addErr(err)
				return true
			}
			vBool, ok := val.(bool)
			if !ok {
				panic(fmt.Errorf("%s has type %T, expected bool", step.expr, val))
			}
			if vBool {
				return visit(steps, yield)
			}
			return true

		case solverStepIndep:
			i := len(b.vals)
			b.vals = append(b.vals, nil)
			defer pop()
			for _, val := range s.vars[step.bind] {
				b.vals[i] = val
				s.tracer.enter(step.bind, val)
				if !visit(steps, yield) {
					return false
				}
				s.tracer.exit()
			}
			return true
		}
		panic("bad step kind")
	}
	return func(yield func(*Bindings, error) bool) {
		if visit(steps, yield) {
			if len(errors) > 0 {
				err := fmt.Errorf("%s", strings.Join(slices.Sorted(maps.Keys(errors)), "\n"))
				yield(nil, err)
			}
		}
	}
}

type solverStep struct {
	kind solverStepKind
	id   int
	expr Expr
	bind Variable // Variable to bind for solverStepAssign or solverStepIndep
	hid  int      // heap index
}

type solverStepKind int

const (
	// solverStepIndep is a solverStep that simultaneously binds
	// [solverStep.bind] to every possible value in bind's domain.
	solverStepIndep solverStepKind = iota
	// solverStepAssign is a solverStep where expr is an [OpEqual] [BinExpr]
	// where the LHS is a Variable. It evaluates the RHS and assigns it to the
	// variable. Variable must not have been bound by an earlier step (any
	// subsequent OpEqual expressions for this variable should instead be a
	// solverStepCheck).
	solverStepAssign
	// solverStepCheck is a solverStep that checks that expr is true and
	// otherwise terminates the current solver branch.
	solverStepCheck
)

func (s *solverStep) Compare(t *solverStep) int {
	// Put assertions before independent variables because they may cut off paths
	// before we have to enumerate values.
	if s.kind != t.kind {
		return cmp.Compare(s.kind, t.kind)
	}
	switch s.kind {
	case solverStepCheck, solverStepAssign:
		return cmp.Compare(s.id, t.id)
	case solverStepIndep:
		return cmp.Compare(s.bind, t.bind)
	}
	panic("bad solverStep kind")
}

type solverHeap []*solverStep

func (sh solverHeap) Len() int { return len(sh) }

func (sh solverHeap) Less(i, j int) bool {
	return sh[i].Compare(sh[j]) < 0
}

func (sh solverHeap) Swap(i, j int) {
	sh[i], sh[j] = sh[j], sh[i]
	sh[i].hid, sh[j].hid = i, j
}

func (sh *solverHeap) Push(x any) {
	item := x.(*solverStep)
	item.hid = len(*sh)
	*sh = append(*sh, item)
}

func (sh *solverHeap) Pop() any {
	old := *sh
	n := len(old)
	item := old[n-1]
	old[n-1] = nil
	item.hid = -1
	*sh = old[0 : n-1]
	return item
}

func (s *Solver) topoSort() (order []*solverStep, err error) {
	// This is a topo-sort with a few tricks: an assertion can be evaluated once
	// all of its input variables are available, BUT a variable value can be
	// resolved by potentially more than one assertion. Hence, we have a mix of
	// "AND" and "OR" dependencies. For example, if we have:
	//
	//   x=Basic{xB, xN}
	//   x=y
	//
	// Then x depends on "Basic{xB, xN}" OR "y" because we can assign x's value
	// as soon as we resolve either of these. But resolving the first of these
	// two expressions depends on xB AND xN.
	//
	// To handle this mix of "AND" and "OR" dependencies, we use a
	// wavefront-style topo sort where we track the number of unresolved input
	// variables to each assertion and whenever we resolve one of these inputs
	// for the first time, we decrement that count. Once it reaches zero, that
	// assertion is let out of the gate.
	//
	// The second trick is that in the set of possible next steps, we bias
	// toward taking steps that are more likely to cut off a path and less
	// likely to cause more fan-out.

	var queue solverHeap
	defs := make(map[Variable][]*solverStep)
	uses := make(map[Variable]map[*solverStep]bool)
	remaining := make(map[*solverStep]int)

	isVarAssign := func(e Expr) (Variable, Expr, bool) {
		switch e := e.(type) {
		case *BinExpr:
			if e.Op == OpEqual {
				switch x := e.X.(type) {
				case Variable:
					return x, e.Y, true
				}
			}
		}
		return "", nil, false
	}

	for i, assert := range s.asserts {
		var rhs Expr
		// Wrap the assertion in a step, assigning an ID.
		var step *solverStep
		if def, val, ok := isVarAssign(assert); ok {
			if def == val {
				// x=x assertion. It's always true and will muck up the sorting,
				// so throw it out.
				continue
			}
			// Variable assignment
			step = &solverStep{kind: solverStepAssign, id: i, expr: assert, hid: -1, bind: def}
			// Record variable definition.
			defs[def] = append(defs[def], step)
			rhs = val
		} else {
			// Boolean check
			step = &solverStep{kind: solverStepCheck, id: i, expr: assert, hid: -1}
			rhs = assert
		}

		// Record variables this step depends on.
		deps := 0
		for use := range exprVars(rhs) {
			if uses[use] == nil {
				uses[use] = make(map[*solverStep]bool)
			}
			if !uses[use][step] {
				uses[use][step] = true
				deps++
			}
		}
		if deps == 0 {
			// Already solvable, enqueue it.
			step.hid = len(queue)
			queue = append(queue, step)
		} else {
			remaining[step] = deps
		}
	}

	// Find the independent variables and also seed the frontier with them.
	for v := range uses {
		if defs[v] == nil {
			if _, ok := s.vars[v]; !ok {
				return nil, fmt.Errorf("no domain for independent variable %q", v)
			}
			queue = append(queue, &solverStep{kind: solverStepIndep, bind: v, hid: -1})
		}
	}
	// Any variables that are declared but not referenced are also independent.
	for v := range s.vars {
		if uses[v] == nil && defs[v] == nil {
			queue = append(queue, &solverStep{kind: solverStepIndep, bind: v, hid: -1})
		}
	}

	// Drive the frontier. queue is the frontier of variables whose dependencies
	// are all resolved, maintained in a heuristic order.
	heap.Init(&queue)
	for len(queue) > 0 {
		step := queue[0]
		heap.Pop(&queue)

		order = append(order, step)

		if step.kind == solverStepCheck {
			continue
		}

		// This step resolved variable v.
		v := step.bind

		// Demote any other assignments of this variable to checks.
		for _, def := range defs[v] {
			if def != step {
				def.kind = solverStepCheck
				// Adjust heap
				if def.hid != -1 {
					heap.Fix(&queue, def.hid)
				}
			}
		}
		delete(defs, v)

		// Check steps that depend on v.
		for use := range uses[v] {
			remaining[use]--
			if remaining[use] == 0 {
				// All variables used by this step are now resolved. Add it to the
				// queue.
				heap.Push(&queue, use)
				delete(remaining, use)
			}
		}
	}

	if len(remaining) > 0 {
		return nil, reportCycle(defs, remaining)
	}
	return order, nil
}

func reportCycle(defs map[Variable][]*solverStep, remaining map[*solverStep]int) error {
	// There was a cycle. There could be more than one cycle, but report one.
	// Start with the "minimum" remaining step for stability.
	var step *solverStep
	for rem := range remaining {
		if rem.kind == solverStepAssign && (step == nil || rem.Compare(step) < 0) {
			step = rem
		}
	}
	// Walk forward through the graph, filtered to unresolved nodes.
	var cycle []Expr
	have := make(map[*solverStep]int)
	var visit func(step *solverStep) error
	visit = func(step *solverStep) error {
		if step.kind != solverStepAssign || remaining[step] == 0 {
			return nil
		}
		if i, ok := have[step]; ok {
			// Found the cycle
			return fmt.Errorf("cyclic requirements: %v", cycle[i:])
		}
		have[step] = len(cycle)
		cycle = append(cycle, step.expr)
		for v := range exprVars(step.expr.(*BinExpr).Y) {
			for _, def := range defs[v] {
				if err := visit(def); err != nil {
					return err
				}
			}
		}
		delete(have, step)
		cycle = cycle[:len(cycle)-1]
		return nil
	}
	err := visit(step)
	if err == nil {
		log.Printf("remaining:")
		for rem, count := range remaining {
			log.Printf("  %v (%d)", rem, count)
		}
		log.Fatal("unresolved assertions, but failed to find a cycle")
	}
	return err
}
