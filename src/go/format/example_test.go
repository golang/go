// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package format_test

import (
	"bytes"
	"fmt"
	"go/format"
	"go/parser"
	"go/token"
	"log"
)

func ExampleNode() {
	const expr = "(6+2*3)/4"

	// parser.ParseExpr parses the argument and returns the
	// corresponding ast.Node.
	node, err := parser.ParseExpr(expr)
	if err != nil {
		log.Fatal(err)
	}

	// Create a FileSet for node. Since the node does not come
	// from a real source file, fset will be empty.
	fset := token.NewFileSet()

	var buf bytes.Buffer
	err = format.Node(&buf, fset, node)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println(buf.String())

	// Output: (6 + 2*3) / 4
}
