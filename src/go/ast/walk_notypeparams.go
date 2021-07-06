// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !typeparams
// +build !typeparams

package ast

import "fmt"

func walkFuncTypeParams(v Visitor, n *FuncType) {}
func walkTypeSpecParams(v Visitor, n *TypeSpec) {}

func walkOtherNodes(v Visitor, n Node) {
	panic(fmt.Sprintf("ast.Walk: unexpected node type %T", n))
}
