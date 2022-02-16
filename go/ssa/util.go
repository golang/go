// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

// This file defines a number of miscellaneous utility functions.

import (
	"fmt"
	"go/ast"
	"go/token"
	"go/types"
	"io"
	"os"
	"sync"

	"golang.org/x/tools/go/ast/astutil"
	"golang.org/x/tools/go/types/typeutil"
)

//// Sanity checking utilities

// assert panics with the mesage msg if p is false.
// Avoid combining with expensive string formatting.
func assert(p bool, msg string) {
	if !p {
		panic(msg)
	}
}

//// AST utilities

func unparen(e ast.Expr) ast.Expr { return astutil.Unparen(e) }

// isBlankIdent returns true iff e is an Ident with name "_".
// They have no associated types.Object, and thus no type.
//
func isBlankIdent(e ast.Expr) bool {
	id, ok := e.(*ast.Ident)
	return ok && id.Name == "_"
}

//// Type utilities.  Some of these belong in go/types.

// isPointer returns true for types whose underlying type is a pointer.
func isPointer(typ types.Type) bool {
	_, ok := typ.Underlying().(*types.Pointer)
	return ok
}

func isInterface(T types.Type) bool { return types.IsInterface(T) }

// deref returns a pointer's element type; otherwise it returns typ.
func deref(typ types.Type) types.Type {
	if p, ok := typ.Underlying().(*types.Pointer); ok {
		return p.Elem()
	}
	return typ
}

// recvType returns the receiver type of method obj.
func recvType(obj *types.Func) types.Type {
	return obj.Type().(*types.Signature).Recv().Type()
}

// logStack prints the formatted "start" message to stderr and
// returns a closure that prints the corresponding "end" message.
// Call using 'defer logStack(...)()' to show builder stack on panic.
// Don't forget trailing parens!
//
func logStack(format string, args ...interface{}) func() {
	msg := fmt.Sprintf(format, args...)
	io.WriteString(os.Stderr, msg)
	io.WriteString(os.Stderr, "\n")
	return func() {
		io.WriteString(os.Stderr, msg)
		io.WriteString(os.Stderr, " end\n")
	}
}

// newVar creates a 'var' for use in a types.Tuple.
func newVar(name string, typ types.Type) *types.Var {
	return types.NewParam(token.NoPos, nil, name, typ)
}

// anonVar creates an anonymous 'var' for use in a types.Tuple.
func anonVar(typ types.Type) *types.Var {
	return newVar("", typ)
}

var lenResults = types.NewTuple(anonVar(tInt))

// makeLen returns the len builtin specialized to type func(T)int.
func makeLen(T types.Type) *Builtin {
	lenParams := types.NewTuple(anonVar(T))
	return &Builtin{
		name: "len",
		sig:  types.NewSignature(nil, lenParams, lenResults, false),
	}
}

// Mapping of a type T to a canonical instance C s.t. types.Indentical(T, C).
// Thread-safe.
type canonizer struct {
	mu    sync.Mutex
	canon typeutil.Map // map from type to a canonical instance
}

// Tuple returns a canonical representative of a Tuple of types.
// Representative of the empty Tuple is nil.
func (c *canonizer) Tuple(ts []types.Type) *types.Tuple {
	if len(ts) == 0 {
		return nil
	}
	vars := make([]*types.Var, len(ts))
	for i, t := range ts {
		vars[i] = anonVar(t)
	}
	tuple := types.NewTuple(vars...)
	return c.Type(tuple).(*types.Tuple)
}

// Type returns a canonical representative of type T.
func (c *canonizer) Type(T types.Type) types.Type {
	c.mu.Lock()
	defer c.mu.Unlock()

	if r := c.canon.At(T); r != nil {
		return r.(types.Type)
	}
	c.canon.Set(T, T)
	return T
}
