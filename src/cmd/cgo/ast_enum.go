// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !compiler_bootstrap

package main

import "go/ast"

// walkEnum handles AST nodes unavailable in the bootstrap toolchain.
func (f *File) walkEnum(x any, visit func(*File, any, astContext)) bool {
	switch node := x.(type) {
	case *ast.EnumVariant:
		if node.Fields != nil {
			f.walk(node.Fields, ctxField, visit)
		}
		return true
	case *ast.EnumDecl:
		if node.TypeParams != nil {
			f.walk(node.TypeParams, ctxParam, visit)
		}
		for _, variant := range node.Variants {
			f.walk(variant, ctxDecl, visit)
		}
		return true
	}
	return false
}
