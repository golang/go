// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package issue13742

import (
	"go/ast"
	. "go/ast"
)

// Both F0 and G0 should appear as functions.
func F0(Node)  {}
func G0() Node { return nil }

// Both F1 and G1 should appear as functions.
func F1(ast.Node)  {}
func G1() ast.Node { return nil }
