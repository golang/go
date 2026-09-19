// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

import (
	"go/ast"
	"go/parser"
	"go/token"
	"log"
)

const src = `package p

import "fmt"

type T struct{ x int }

func (t *T) M(n int) (r int) {
	defer func() { r++ }()
	go fmt.Println(n)
	if n > 0 {
		r = n
	} else {
		r = -n
	}
	for i := 0; i < n; i++ {
		r += t.x
	}
	switch v := any(n).(type) {
	case int:
		r = v
	}
	return
}
`

// Walk parses src and walks the syntax tree. ast.Walk converts each
// node to ast.Node, which looks up the itab in the itab table, and
// then compares that itab with the itab of the host in a type switch.
func Walk() {
	f, err := parser.ParseFile(token.NewFileSet(), "src.go", src, parser.SkipObjectResolution)
	if err != nil {
		log.Fatal(err)
	}
	ast.Inspect(f, func(ast.Node) bool { return true })
}
