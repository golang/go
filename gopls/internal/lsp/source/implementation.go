// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package source

import (
	"errors"
	"go/ast"
	"go/token"
	"go/types"
)

// TODO(adonovan): move these declarations elsewhere.

// concreteImplementsIntf returns true if a is an interface type implemented by
// concrete type b, or vice versa.
func concreteImplementsIntf(a, b types.Type) bool {
	aIsIntf, bIsIntf := types.IsInterface(a), types.IsInterface(b)

	// Make sure exactly one is an interface type.
	if aIsIntf == bIsIntf {
		return false
	}

	// Rearrange if needed so "a" is the concrete type.
	if aIsIntf {
		a, b = b, a
	}

	// TODO(adonovan): this should really use GenericAssignableTo
	// to report (e.g.) "ArrayList[T] implements List[T]", but
	// GenericAssignableTo doesn't work correctly on pointers to
	// generic named types. Thus the legacy implementation and the
	// "local" part of implementation2 fail to report generics.
	// The global algorithm based on subsets does the right thing.
	return types.AssignableTo(a, b)
}

var (
	// TODO(adonovan): why do various RPC handlers related to
	// IncomingCalls return (nil, nil) on the protocol in response
	// to this error? That seems like a violation of the protocol.
	// Is it perhaps a workaround for VSCode behavior?
	errNoObjectFound = errors.New("no object found")
)

// pathEnclosingObjNode returns the AST path to the object-defining
// node associated with pos. "Object-defining" means either an
// *ast.Ident mapped directly to a types.Object or an ast.Node mapped
// implicitly to a types.Object.
func pathEnclosingObjNode(f *ast.File, pos token.Pos) []ast.Node {
	var (
		path  []ast.Node
		found bool
	)

	ast.Inspect(f, func(n ast.Node) bool {
		if found {
			return false
		}

		if n == nil {
			path = path[:len(path)-1]
			return false
		}

		path = append(path, n)

		switch n := n.(type) {
		case *ast.Ident:
			// Include the position directly after identifier. This handles
			// the common case where the cursor is right after the
			// identifier the user is currently typing. Previously we
			// handled this by calling astutil.PathEnclosingInterval twice,
			// once for "pos" and once for "pos-1".
			found = n.Pos() <= pos && pos <= n.End()
		case *ast.ImportSpec:
			if n.Path.Pos() <= pos && pos < n.Path.End() {
				found = true
				// If import spec has a name, add name to path even though
				// position isn't in the name.
				if n.Name != nil {
					path = append(path, n.Name)
				}
			}
		case *ast.StarExpr:
			// Follow star expressions to the inner identifier.
			if pos == n.Star {
				pos = n.X.Pos()
			}
		}

		return !found
	})

	if len(path) == 0 {
		return nil
	}

	// Reverse path so leaf is first element.
	for i := 0; i < len(path)/2; i++ {
		path[i], path[len(path)-1-i] = path[len(path)-1-i], path[i]
	}

	return path
}
