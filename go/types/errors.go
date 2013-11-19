// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements various error reporters.

package types

import (
	"fmt"
	"go/ast"
	"go/token"
	"strings"
)

func assert(p bool) {
	if !p {
		panic("assertion failed")
	}
}

func unreachable() {
	panic("unreachable")
}

func (check *checker) sprintf(format string, args ...interface{}) string {
	for i, arg := range args {
		switch a := arg.(type) {
		case nil:
			args[i] = "<nil>"
		case operand:
			panic("internal error: should always pass *operand")
		case token.Pos:
			args[i] = check.fset.Position(a).String()
		case ast.Expr:
			args[i] = ExprString(a)
		}
	}
	return fmt.Sprintf(format, args...)
}

func (check *checker) trace(pos token.Pos, format string, args ...interface{}) {
	fmt.Printf("%s:\t%s%s\n",
		check.fset.Position(pos),
		strings.Repeat(".  ", check.indent),
		check.sprintf(format, args...),
	)
}

// dump is only needed for debugging
func (check *checker) dump(format string, args ...interface{}) {
	fmt.Println(check.sprintf(format, args...))
}

func (check *checker) err(pos token.Pos, msg string) {
	err := Error{check.fset, pos, msg}
	if check.firstErr == nil {
		check.firstErr = err
	}
	f := check.conf.Error
	if f == nil {
		panic(bailout{}) // report only first error
	}
	f(err)
}

func (check *checker) errorf(pos token.Pos, format string, args ...interface{}) {
	check.err(pos, check.sprintf(format, args...))
}

func (check *checker) invalidAST(pos token.Pos, format string, args ...interface{}) {
	check.errorf(pos, "invalid AST: "+format, args...)
}

func (check *checker) invalidArg(pos token.Pos, format string, args ...interface{}) {
	check.errorf(pos, "invalid argument: "+format, args...)
}

func (check *checker) invalidOp(pos token.Pos, format string, args ...interface{}) {
	check.errorf(pos, "invalid operation: "+format, args...)
}
