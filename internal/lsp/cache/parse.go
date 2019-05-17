// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cache

import (
	"context"
	"fmt"
	"go/ast"
	"go/parser"
	"go/scanner"
	"go/token"
	"os"
	"path/filepath"
	"strings"
	"sync"

	"golang.org/x/tools/internal/span"
)

func parseFile(fset *token.FileSet, filename string, src []byte) (*ast.File, error) {
	return parser.ParseFile(fset, filename, src, parser.AllErrors|parser.ParseComments)
}

// We use a counting semaphore to limit
// the number of parallel I/O calls per process.
var ioLimit = make(chan bool, 20)

// parseFiles reads and parses the Go source files and returns the ASTs
// of the ones that could be at least partially parsed, along with a
// list of I/O and parse errors encountered.
//
// Because files are scanned in parallel, the token.Pos
// positions of the resulting ast.Files are not ordered.
//
func (imp *importer) parseFiles(filenames []string) ([]*ast.File, []error) {
	var wg sync.WaitGroup
	n := len(filenames)
	parsed := make([]*ast.File, n)
	errors := make([]error, n)
	for i, filename := range filenames {
		if imp.ctx.Err() != nil {
			parsed[i] = nil
			errors[i] = imp.ctx.Err()
			continue
		}

		// First, check if we have already cached an AST for this file.
		f, err := imp.view.findFile(span.FileURI(filename))
		if err != nil || f == nil {
			parsed[i], errors[i] = nil, err
			continue
		}
		gof, ok := f.(*goFile)
		if !ok {
			parsed[i], errors[i] = nil, fmt.Errorf("Non go file in parse call: %v", filename)
			continue
		}

		wg.Add(1)
		go func(i int, filename string) {
			ioLimit <- true // wait

			if gof.ast != nil {
				parsed[i], errors[i] = gof.ast, nil
			} else {
				// We don't have a cached AST for this file.
				gof.read(imp.ctx)
				if gof.fc.Error != nil {
					return
				}
				src := gof.fc.Data
				if src == nil {
					parsed[i], errors[i] = nil, fmt.Errorf("No source for %v", filename)
				} else {
					// ParseFile may return both an AST and an error.
					parsed[i], errors[i] = parseFile(imp.fset, filename, src)

					// Fix any badly parsed parts of the AST.
					if file := parsed[i]; file != nil {
						tok := imp.fset.File(file.Pos())
						imp.view.fix(imp.ctx, parsed[i], tok, src)
					}
				}
			}

			<-ioLimit // signal
			wg.Done()
		}(i, filename)
	}
	wg.Wait()

	// Eliminate nils, preserving order.
	var o int
	for _, f := range parsed {
		if f != nil {
			parsed[o] = f
			o++
		}
	}
	parsed = parsed[:o]

	o = 0
	for _, err := range errors {
		if err != nil {
			errors[o] = err
			o++
		}
	}
	errors = errors[:o]

	return parsed, errors
}

// sameFile returns true if x and y have the same basename and denote
// the same file.
//
func sameFile(x, y string) bool {
	if x == y {
		// It could be the case that y doesn't exist.
		// For instance, it may be an overlay file that
		// hasn't been written to disk. To handle that case
		// let x == y through. (We added the exact absolute path
		// string to the CompiledGoFiles list, so the unwritten
		// overlay case implies x==y.)
		return true
	}
	if strings.EqualFold(filepath.Base(x), filepath.Base(y)) { // (optimisation)
		if xi, err := os.Stat(x); err == nil {
			if yi, err := os.Stat(y); err == nil {
				return os.SameFile(xi, yi)
			}
		}
	}
	return false
}

// fix inspects and potentially modifies any *ast.BadStmts or *ast.BadExprs in the AST.

// We attempt to modify the AST such that we can type-check it more effectively.
func (v *view) fix(ctx context.Context, file *ast.File, tok *token.File, src []byte) {
	var parent ast.Node
	ast.Inspect(file, func(n ast.Node) bool {
		if n == nil {
			return false
		}
		switch n := n.(type) {
		case *ast.BadStmt:
			if err := v.parseDeferOrGoStmt(n, parent, tok, src); err != nil {
				v.Session().Logger().Debugf(ctx, "unable to parse defer or go from *ast.BadStmt: %v", err)
			}
			return false
		default:
			parent = n
			return true
		}
	})
}

// parseDeferOrGoStmt tries to parse an *ast.BadStmt into a defer or a go statement.
//
// go/parser packages a statement of the form "defer x." as an *ast.BadStmt because
// it does not include a call expression. This means that go/types skips type-checking
// this statement entirely, and we can't use the type information when completing.
// Here, we try to generate a fake *ast.DeferStmt or *ast.GoStmt to put into the AST,
// instead of the *ast.BadStmt.
func (v *view) parseDeferOrGoStmt(bad *ast.BadStmt, parent ast.Node, tok *token.File, src []byte) error {
	// Check if we have a bad statement containing either a "go" or "defer".
	s := &scanner.Scanner{}
	s.Init(tok, src, nil, 0)

	var pos token.Pos
	var tkn token.Token
	var lit string
	for {
		if tkn == token.EOF {
			return fmt.Errorf("reached the end of the file")
		}
		if pos >= bad.From {
			break
		}
		pos, tkn, lit = s.Scan()
	}
	var stmt ast.Stmt
	switch lit {
	case "defer":
		stmt = &ast.DeferStmt{
			Defer: pos,
		}
	case "go":
		stmt = &ast.GoStmt{
			Go: pos,
		}
	default:
		return fmt.Errorf("no defer or go statement found")
	}

	// The expression after the "defer" or "go" starts at this position.
	from, _, _ := s.Scan()
	var to, curr token.Pos
FindTo:
	for {
		curr, tkn, lit = s.Scan()
		// TODO(rstambler): This still needs more handling to work correctly.
		// We encounter a specific issue with code that looks like this:
		//
		//      defer x.<>
		//      y := 1
		//
		// In this scenario, we parse it as "defer x.y", which then fails to
		// type-check, and we don't get completions as expected.
		switch tkn {
		case token.COMMENT, token.EOF, token.SEMICOLON, token.DEFINE:
			break FindTo
		}
		// to is the end of expression that should become the Fun part of the call.
		to = curr
	}
	if !from.IsValid() || tok.Offset(from) >= len(src) {
		return fmt.Errorf("invalid from position")
	}
	if !to.IsValid() || tok.Offset(to)+1 >= len(src) {
		return fmt.Errorf("invalid to position")
	}
	exprstr := string(src[tok.Offset(from) : tok.Offset(to)+1])
	expr, err := parser.ParseExpr(exprstr)
	if expr == nil {
		return fmt.Errorf("no expr in %s: %v", exprstr, err)
	}
	// parser.ParseExpr returns undefined positions.
	// Adjust them for the current file.
	v.offsetPositions(expr, from-1)

	// Package the expression into a fake *ast.CallExpr and re-insert into the function.
	call := &ast.CallExpr{
		Fun:    expr,
		Lparen: to,
		Rparen: to,
	}
	switch stmt := stmt.(type) {
	case *ast.DeferStmt:
		stmt.Call = call
	case *ast.GoStmt:
		stmt.Call = call
	}
	switch parent := parent.(type) {
	case *ast.BlockStmt:
		for i, s := range parent.List {
			if s == bad {
				parent.List[i] = stmt
				break
			}
		}
	}
	return nil
}

// offsetPositions applies an offset to the positions in an ast.Node.
// TODO(rstambler): Add more cases here as they become necessary.
func (v *view) offsetPositions(expr ast.Expr, offset token.Pos) {
	ast.Inspect(expr, func(n ast.Node) bool {
		switch n := n.(type) {
		case *ast.Ident:
			n.NamePos += offset
			return false
		default:
			return true
		}
	})
}
