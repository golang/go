// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file holds the go generate command to run yacc on the grammar in expr.y.
// To build expr:
//	% go generate
//	% go build

//go:generate -command yacc go tool yacc
//go:generate yacc -o expr.go -p "expr" expr.y

// Expr is a simple expression evaluator that serves as a working example of
// how to use Go's yacc implementation.
package main
