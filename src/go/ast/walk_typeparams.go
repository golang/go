// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build typeparams
// +build typeparams

package ast

import (
	"fmt"
)

func walkFuncTypeParams(v Visitor, n *FuncType) {
	if n.TParams != nil {
		Walk(v, n.TParams)
	}
}

func walkTypeSpecParams(v Visitor, n *TypeSpec) {
	if n.TParams != nil {
		Walk(v, n.TParams)
	}
}

func walkOtherNodes(v Visitor, n Node) {
	if e, ok := n.(*ListExpr); ok {
		if e != nil {
			for _, elem := range e.ElemList {
				Walk(v, elem)
			}
		}
	} else {
		panic(fmt.Sprintf("ast.Walk: unexpected node type %T", n))
	}
}
