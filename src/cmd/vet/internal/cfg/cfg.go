// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This package constructs a simple control-flow graph (CFG) of the
// statements and expressions within a single function.
//
// Use cfg.New to construct the CFG for a function body.
//
// The blocks of the CFG contain all the function's non-control
// statements.  The CFG does not contain control statements such as If,
// Switch, Select, and Branch, but does contain their subexpressions.
// For example, this source code:
//
//	if x := f(); x != nil {
//		T()
//	} else {
//		F()
//	}
//
// produces this CFG:
//
//    1:  x := f()
//        x != nil
//        succs: 2, 3
//    2:  T()
//        succs: 4
//    3:  F()
//        succs: 4
//    4:
//
// The CFG does contain Return statements; even implicit returns are
// materialized (at the position of the function's closing brace).
//
// The CFG does not record conditions associated with conditional branch
// edges, nor the short-circuit semantics of the && and || operators,
// nor abnormal control flow caused by panic.  If you need this
// information, use golang.org/x/tools/go/ssa instead.
//
package cfg

// Although the vet tool has type information, it is often extremely
// fragmentary, so for simplicity this package does not depend on
// go/types.  Consequently control-flow conditions are ignored even
// when constant, and "mayReturn" information must be provided by the
// client.
import (
	"bytes"
	"fmt"
	"go/ast"
	"go/format"
	"go/token"
)

// A CFG represents the control-flow graph of a single function.
//
// The entry point is Blocks[0]; there may be multiple return blocks.
type CFG struct {
	Blocks []*Block // block[0] is entry; order otherwise undefined
}

// A Block represents a basic block: a list of statements and
// expressions that are always evaluated sequentially.
//
// A block may have 0-2 successors: zero for a return block or a block
// that calls a function such as panic that never returns; one for a
// normal (jump) block; and two for a conditional (if) block.
type Block struct {
	Nodes []ast.Node // statements, expressions, and ValueSpecs
	Succs []*Block   // successor nodes in the graph

	comment     string    // for debugging
	index       int32     // index within CFG.Blocks
	unreachable bool      // is block of stmts following return/panic/for{}
	succs2      [2]*Block // underlying array for Succs
}

// New returns a new control-flow graph for the specified function body,
// which must be non-nil.
//
// The CFG builder calls mayReturn to determine whether a given function
// call may return.  For example, calls to panic, os.Exit, and log.Fatal
// do not return, so the builder can remove infeasible graph edges
// following such calls.  The builder calls mayReturn only for a
// CallExpr beneath an ExprStmt.
func New(body *ast.BlockStmt, mayReturn func(*ast.CallExpr) bool) *CFG {
	b := builder{
		mayReturn: mayReturn,
		cfg:       new(CFG),
	}
	b.current = b.newBlock("entry")
	b.stmt(body)

	// Does control fall off the end of the function's body?
	// Make implicit return explicit.
	if b.current != nil && !b.current.unreachable {
		b.add(&ast.ReturnStmt{
			Return: body.End() - 1,
		})
	}

	return b.cfg
}

func (b *Block) String() string {
	return fmt.Sprintf("block %d (%s)", b.index, b.comment)
}

// Return returns the return statement at the end of this block if present, nil otherwise.
func (b *Block) Return() (ret *ast.ReturnStmt) {
	if len(b.Nodes) > 0 {
		ret, _ = b.Nodes[len(b.Nodes)-1].(*ast.ReturnStmt)
	}
	return
}

// Format formats the control-flow graph for ease of debugging.
func (g *CFG) Format(fset *token.FileSet) string {
	var buf bytes.Buffer
	for _, b := range g.Blocks {
		fmt.Fprintf(&buf, ".%d: # %s\n", b.index, b.comment)
		for _, n := range b.Nodes {
			fmt.Fprintf(&buf, "\t%s\n", formatNode(fset, n))
		}
		if len(b.Succs) > 0 {
			fmt.Fprintf(&buf, "\tsuccs:")
			for _, succ := range b.Succs {
				fmt.Fprintf(&buf, " %d", succ.index)
			}
			buf.WriteByte('\n')
		}
		buf.WriteByte('\n')
	}
	return buf.String()
}

func formatNode(fset *token.FileSet, n ast.Node) string {
	var buf bytes.Buffer
	format.Node(&buf, fset, n)
	// Indent secondary lines by a tab.
	return string(bytes.Replace(buf.Bytes(), []byte("\n"), []byte("\n\t"), -1))
}
