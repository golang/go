// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package eval

import (
	"fmt";
	"go/ast";
	"go/scanner";
	"go/token";
)


type positioned interface {
	Pos() token.Position;
}


// A compiler captures information used throughout an entire
// compilation.  Currently it includes only the error handler.
//
// TODO(austin) This might actually represent package level, in which
// case it should be package compiler.
type compiler struct {
	errors scanner.ErrorHandler;
	numErrors int;
	silentErrors int;
}

func (a *compiler) diagAt(pos positioned, format string, args ...) {
	a.errors.Error(pos.Pos(), fmt.Sprintf(format, args));
	a.numErrors++;
}

func (a *compiler) numError() int {
	return a.numErrors + a.silentErrors;
}


// TODO(austin) These can all go in stmt.go now
type label struct {
	name string;
	desc string;
	// The PC goto statements should jump to, or nil if this label
	// cannot be goto'd (such as an anonymous for loop label).
	gotoPC *uint;
	// The PC break statements should jump to, or nil if a break
	// statement is invalid.
	breakPC *uint;
	// The PC continue statements should jump to, or nil if a
	// continue statement is invalid.
	continuePC *uint;
	// The position where this label was resolved.  If it has not
	// been resolved yet, an invalid position.
	resolved token.Position;
	// The position where this label was first jumped to.
	used token.Position;
}

// A funcCompiler captures information used throughout the compilation
// of a single function body.
type funcCompiler struct {
	*compiler;
	fnType *FuncType;
	// Whether the out variables are named.  This affects what
	// kinds of return statements are legal.
	outVarsNamed bool;
	*codeBuf;
	flow *flowBuf;
	labels map[string] *label;
}

// A blockCompiler captures information used throughout the compilation
// of a single block within a function.
type blockCompiler struct {
	*funcCompiler;
	block *block;
	// The label of this block, used for finding break and
	// continue labels.
	label *label;
	// The blockCompiler for the block enclosing this one, or nil
	// for a function-level block.
	parent *blockCompiler;
}
