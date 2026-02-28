// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

import (
	"context"
	"database/sql"
)

type I interface {
	m(int, int, *int, int)
}

type T struct{}

func (_ *T) m(a, b, c, d int) {}

var _ I = new /* ERROR "have m(int, int, int, int)\n\t\twant m(int, int, *int, int)" */ (T)

// (slightly modified) test case from issue

type Result struct {
	Value string
}

type Executor interface {
	Execute(context.Context, sql.Stmt, int, []sql.NamedArg, int) (Result, error)
}

type myExecutor struct{}

func (_ *myExecutor) Execute(ctx context.Context, stmt sql.Stmt, maxrows int, args []sql.NamedArg, urgency int) (*Result, error) {
	return &Result{}, nil
}

var ex Executor = new /* ERROR "have Execute(context.Context, sql.Stmt, int, []sql.NamedArg, int) (*Result, error)\n\t\twant Execute(context.Context, sql.Stmt, int, []sql.NamedArg, int) (Result, error)" */ (myExecutor)
