// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package astutil

import "go/ast"

// Unparen returns e with any enclosing parentheses stripped.
// Deprecated: use [ast.Unparen].
func Unparen(e ast.Expr) ast.Expr { return ast.Unparen(e) }
