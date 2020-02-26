// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cache

import (
	"bytes"
	"context"
	"go/ast"
	"go/parser"
	"go/scanner"
	"go/token"
	"reflect"

	"golang.org/x/tools/internal/event"
	"golang.org/x/tools/internal/lsp/debug/tag"
	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/memoize"
	"golang.org/x/tools/internal/span"
	errors "golang.org/x/xerrors"
)

// parseKey uniquely identifies a parsed Go file.
type parseKey struct {
	file source.FileIdentity
	mode source.ParseMode
}

// astCacheKey is similar to parseKey, but is a distinct type because
// it is used to key a different value within the same map.
type astCacheKey parseKey

type parseGoHandle struct {
	handle         *memoize.Handle
	file           source.FileHandle
	mode           source.ParseMode
	astCacheHandle *memoize.Handle
}

type parseGoData struct {
	memoize.NoCopy

	ast    *ast.File
	mapper *protocol.ColumnMapper

	// Source code used to build the AST. It may be different from the
	// actual content of the file if we have fixed the AST, in which case,
	// fixed will be true.
	src   []byte
	fixed bool

	parseError error // errors associated with parsing the file
	err        error // any other errors
}

func (c *Cache) ParseGoHandle(ctx context.Context, fh source.FileHandle, mode source.ParseMode) source.ParseGoHandle {
	return c.parseGoHandle(ctx, fh, mode)
}

func (c *Cache) parseGoHandle(ctx context.Context, fh source.FileHandle, mode source.ParseMode) *parseGoHandle {
	key := parseKey{
		file: fh.Identity(),
		mode: mode,
	}
	fset := c.fset
	h := c.store.Bind(key, func(ctx context.Context) interface{} {
		return parseGo(ctx, fset, fh, mode)
	})

	return &parseGoHandle{
		handle: h,
		file:   fh,
		mode:   mode,
		astCacheHandle: c.store.Bind(astCacheKey(key), func(ctx context.Context) interface{} {
			return buildASTCache(ctx, h)
		}),
	}
}

func (pgh *parseGoHandle) String() string {
	return pgh.File().URI().Filename()
}

func (pgh *parseGoHandle) File() source.FileHandle {
	return pgh.file
}

func (pgh *parseGoHandle) Mode() source.ParseMode {
	return pgh.mode
}

func (pgh *parseGoHandle) Parse(ctx context.Context) (*ast.File, []byte, *protocol.ColumnMapper, error, error) {
	data, err := pgh.parse(ctx)
	if err != nil {
		return nil, nil, nil, nil, err
	}
	return data.ast, data.src, data.mapper, data.parseError, data.err
}

func (pgh *parseGoHandle) parse(ctx context.Context) (*parseGoData, error) {
	v, err := pgh.handle.Get(ctx)
	if err != nil {
		return nil, err
	}
	data, ok := v.(*parseGoData)
	if !ok {
		return nil, errors.Errorf("no parsed file for %s", pgh.File().URI())
	}
	return data, nil
}

func (pgh *parseGoHandle) Cached() (*ast.File, []byte, *protocol.ColumnMapper, error, error) {
	v := pgh.handle.Cached()
	if v == nil {
		return nil, nil, nil, nil, errors.Errorf("no cached AST for %s", pgh.file.URI())
	}
	data := v.(*parseGoData)
	return data.ast, data.src, data.mapper, data.parseError, data.err
}

func (pgh *parseGoHandle) PosToDecl(ctx context.Context) (map[token.Pos]ast.Decl, error) {
	v, err := pgh.astCacheHandle.Get(ctx)
	if err != nil || v == nil {
		return nil, err
	}

	data := v.(*astCacheData)
	if data.err != nil {
		return nil, data.err
	}

	return data.posToDecl, nil
}

func (pgh *parseGoHandle) PosToField(ctx context.Context) (map[token.Pos]*ast.Field, error) {
	v, err := pgh.astCacheHandle.Get(ctx)
	if err != nil || v == nil {
		return nil, err
	}

	data := v.(*astCacheData)
	if data.err != nil {
		return nil, data.err
	}

	return data.posToField, nil
}

type astCacheData struct {
	memoize.NoCopy

	err error

	posToDecl  map[token.Pos]ast.Decl
	posToField map[token.Pos]*ast.Field
}

// buildASTCache builds caches to aid in quickly going from the typed
// world to the syntactic world.
func buildASTCache(ctx context.Context, parseHandle *memoize.Handle) *astCacheData {
	var (
		// path contains all ancestors, including n.
		path []ast.Node
		// decls contains all ancestors that are decls.
		decls []ast.Decl
	)

	v, err := parseHandle.Get(ctx)
	if err != nil || v == nil || v.(*parseGoData).ast == nil {
		return &astCacheData{err: err}
	}

	data := &astCacheData{
		posToDecl:  make(map[token.Pos]ast.Decl),
		posToField: make(map[token.Pos]*ast.Field),
	}

	ast.Inspect(v.(*parseGoData).ast, func(n ast.Node) bool {
		if n == nil {
			lastP := path[len(path)-1]
			path = path[:len(path)-1]
			if len(decls) > 0 && decls[len(decls)-1] == lastP {
				decls = decls[:len(decls)-1]
			}
			return false
		}

		path = append(path, n)

		switch n := n.(type) {
		case *ast.Field:
			addField := func(f ast.Node) {
				if f.Pos().IsValid() {
					data.posToField[f.Pos()] = n
					if len(decls) > 0 {
						data.posToDecl[f.Pos()] = decls[len(decls)-1]
					}
				}
			}

			// Add mapping for *ast.Field itself. This handles embedded
			// fields which have no associated *ast.Ident name.
			addField(n)

			// Add mapping for each field name since you can have
			// multiple names for the same type expression.
			for _, name := range n.Names {
				addField(name)
			}

			// Also map "X" in "...X" to the containing *ast.Field. This
			// makes it easy to format variadic signature params
			// properly.
			if elips, ok := n.Type.(*ast.Ellipsis); ok && elips.Elt != nil {
				addField(elips.Elt)
			}
		case *ast.FuncDecl:
			decls = append(decls, n)

			if n.Name != nil && n.Name.Pos().IsValid() {
				data.posToDecl[n.Name.Pos()] = n
			}
		case *ast.GenDecl:
			decls = append(decls, n)

			for _, spec := range n.Specs {
				switch spec := spec.(type) {
				case *ast.TypeSpec:
					if spec.Name != nil && spec.Name.Pos().IsValid() {
						data.posToDecl[spec.Name.Pos()] = n
					}
				case *ast.ValueSpec:
					for _, id := range spec.Names {
						if id != nil && id.Pos().IsValid() {
							data.posToDecl[id.Pos()] = n
						}
					}
				}
			}
		}

		return true
	})

	return data
}

func hashParseKeys(pghs []*parseGoHandle) string {
	b := bytes.NewBuffer(nil)
	for _, pgh := range pghs {
		b.WriteString(pgh.file.Identity().String())
		b.WriteByte(byte(pgh.Mode()))
	}
	return hashContents(b.Bytes())
}

func parseGo(ctx context.Context, fset *token.FileSet, fh source.FileHandle, mode source.ParseMode) *parseGoData {
	ctx, done := event.Start(ctx, "cache.parseGo", tag.File.Of(fh.URI().Filename()))
	defer done()

	if fh.Kind() != source.Go {
		return &parseGoData{err: errors.Errorf("cannot parse non-Go file %s", fh.URI())}
	}
	buf, err := fh.Read()
	if err != nil {
		return &parseGoData{err: err}
	}

	parserMode := parser.AllErrors | parser.ParseComments
	if mode == source.ParseHeader {
		parserMode = parser.ImportsOnly | parser.ParseComments
	}
	file, parseError := parser.ParseFile(fset, fh.URI().Filename(), buf, parserMode)
	var tok *token.File
	var fixed bool
	if file != nil {
		tok = fset.File(file.Pos())
		if tok == nil {
			return &parseGoData{err: errors.Errorf("successfully parsed but no token.File for %s (%v)", fh.URI(), parseError)}
		}

		// Fix any badly parsed parts of the AST.
		fixed = fixAST(ctx, file, tok, buf)

		// Fix certain syntax errors that render the file unparseable.
		newSrc := fixSrc(file, tok, buf)
		if newSrc != nil {
			newFile, _ := parser.ParseFile(fset, fh.URI().Filename(), newSrc, parserMode)
			if newFile != nil {
				// Maintain the original parseError so we don't try formatting the doctored file.
				file = newFile
				buf = newSrc
				tok = fset.File(file.Pos())

				fixed = fixAST(ctx, file, tok, buf)
			}
		}

		if mode == source.ParseExported {
			trimAST(file)
		}
	}
	if file == nil {
		// If the file is nil only due to parse errors,
		// the parse errors are the actual errors.
		err := parseError
		if err == nil {
			err = errors.Errorf("no AST for %s", fh.URI())
		}
		return &parseGoData{parseError: parseError, err: err}
	}
	m := &protocol.ColumnMapper{
		URI:       fh.URI(),
		Converter: span.NewTokenConverter(fset, tok),
		Content:   buf,
	}
	return &parseGoData{
		src:        buf,
		ast:        file,
		mapper:     m,
		parseError: parseError,
		fixed:      fixed,
	}
}

// trimAST clears any part of the AST not relevant to type checking
// expressions at pos.
func trimAST(file *ast.File) {
	ast.Inspect(file, func(n ast.Node) bool {
		if n == nil {
			return false
		}
		switch n := n.(type) {
		case *ast.FuncDecl:
			n.Body = nil
		case *ast.BlockStmt:
			n.List = nil
		case *ast.CaseClause:
			n.Body = nil
		case *ast.CommClause:
			n.Body = nil
		case *ast.CompositeLit:
			// Leave elts in place for [...]T
			// array literals, because they can
			// affect the expression's type.
			if !isEllipsisArray(n.Type) {
				n.Elts = nil
			}
		}
		return true
	})
}

func isEllipsisArray(n ast.Expr) bool {
	at, ok := n.(*ast.ArrayType)
	if !ok {
		return false
	}
	_, ok = at.Len.(*ast.Ellipsis)
	return ok
}

// fixAST inspects the AST and potentially modifies any *ast.BadStmts so that it can be
// type-checked more effectively.
func fixAST(ctx context.Context, n ast.Node, tok *token.File, src []byte) (fixed bool) {
	var err error
	walkASTWithParent(n, func(n, parent ast.Node) bool {
		switch n := n.(type) {
		case *ast.BadStmt:
			if fixed = fixDeferOrGoStmt(n, parent, tok, src); fixed {
				// Recursively fix in our fixed node.
				_ = fixAST(ctx, parent, tok, src)
			} else {
				err = errors.Errorf("unable to parse defer or go from *ast.BadStmt: %v", err)
			}
			return false
		case *ast.BadExpr:
			if fixed = fixArrayType(n, parent, tok, src); fixed {
				// Recursively fix in our fixed node.
				_ = fixAST(ctx, parent, tok, src)
				return false
			}

			// Fix cases where parser interprets if/for/switch "init"
			// statement as "cond" expression, e.g.:
			//
			//   // "i := foo" is init statement, not condition.
			//   for i := foo
			//
			fixInitStmt(n, parent, tok, src)

			return false
		case *ast.SelectorExpr:
			// Fix cases where a keyword prefix results in a phantom "_" selector, e.g.:
			//
			//   foo.var<> // want to complete to "foo.variance"
			//
			fixPhantomSelector(n, tok, src)
			return true

		case *ast.BlockStmt:
			switch parent.(type) {
			case *ast.SwitchStmt, *ast.TypeSwitchStmt, *ast.SelectStmt:
				// Adjust closing curly brace of empty switch/select
				// statements so we can complete inside them.
				fixEmptySwitch(n, tok, src)
			}

			return true
		default:
			return true
		}
	})
	return fixed
}

// walkASTWithParent walks the AST rooted at n. The semantics are
// similar to ast.Inspect except it does not call f(nil).
func walkASTWithParent(n ast.Node, f func(n ast.Node, parent ast.Node) bool) {
	var ancestors []ast.Node
	ast.Inspect(n, func(n ast.Node) (recurse bool) {
		defer func() {
			if recurse {
				ancestors = append(ancestors, n)
			}
		}()

		if n == nil {
			ancestors = ancestors[:len(ancestors)-1]
			return false
		}

		var parent ast.Node
		if len(ancestors) > 0 {
			parent = ancestors[len(ancestors)-1]
		}

		return f(n, parent)
	})
}

// fixSrc attempts to modify the file's source code to fix certain
// syntax errors that leave the rest of the file unparsed.
func fixSrc(f *ast.File, tok *token.File, src []byte) (newSrc []byte) {
	walkASTWithParent(f, func(n, parent ast.Node) bool {
		if newSrc != nil {
			return false
		}

		switch n := n.(type) {
		case *ast.BlockStmt:
			newSrc = fixMissingCurlies(f, n, parent, tok, src)
		case *ast.SelectorExpr:
			newSrc = fixDanglingSelector(n, tok, src)
		}

		return newSrc == nil
	})

	return newSrc
}

// fixMissingCurlies adds in curly braces for block statements that
// are missing curly braces. For example:
//
//   if foo
//
// becomes
//
//   if foo {}
func fixMissingCurlies(f *ast.File, b *ast.BlockStmt, parent ast.Node, tok *token.File, src []byte) []byte {
	// If the "{" is already in the source code, there isn't anything to
	// fix since we aren't missing curlies.
	if b.Lbrace.IsValid() {
		braceOffset := tok.Offset(b.Lbrace)
		if braceOffset < len(src) && src[braceOffset] == '{' {
			return nil
		}
	}

	parentLine := tok.Line(parent.Pos())

	if parentLine >= tok.LineCount() {
		// If we are the last line in the file, no need to fix anything.
		return nil
	}

	// Insert curlies at the end of parent's starting line. The parent
	// is the statement that contains the block, e.g. *ast.IfStmt. The
	// block's Pos()/End() can't be relied upon because they are based
	// on the (missing) curly braces. We assume the statement is a
	// single line for now and try sticking the curly braces at the end.
	insertPos := tok.LineStart(parentLine+1) - 1

	// Scootch position backwards until it's not in a comment. For example:
	//
	// if foo<> // some amazing comment |
	// someOtherCode()
	//
	// insertPos will be located at "|", so we back it out of the comment.
	didSomething := true
	for didSomething {
		didSomething = false
		for _, c := range f.Comments {
			if c.Pos() < insertPos && insertPos <= c.End() {
				insertPos = c.Pos()
				didSomething = true
			}
		}
	}

	// Bail out if line doesn't end in an ident or ".". This is to avoid
	// cases like below where we end up making things worse by adding
	// curlies:
	//
	//   if foo &&
	//     bar<>
	switch precedingToken(insertPos, tok, src) {
	case token.IDENT, token.PERIOD:
		// ok
	default:
		return nil
	}

	var buf bytes.Buffer
	buf.Grow(len(src) + 3)
	buf.Write(src[:tok.Offset(insertPos)])

	// Detect if we need to insert a semicolon to fix "for" loop situations like:
	//
	//   for i := foo(); foo<>
	//
	// Just adding curlies is not sufficient to make things parse well.
	if fs, ok := parent.(*ast.ForStmt); ok {
		if _, ok := fs.Cond.(*ast.BadExpr); !ok {
			if xs, ok := fs.Post.(*ast.ExprStmt); ok {
				if _, ok := xs.X.(*ast.BadExpr); ok {
					buf.WriteByte(';')
				}
			}
		}
	}

	// Insert "{}" at insertPos.
	buf.WriteByte('{')
	buf.WriteByte('}')
	buf.Write(src[tok.Offset(insertPos):])
	return buf.Bytes()
}

// fixEmptySwitch moves empty switch/select statements' closing curly
// brace down one line. This allows us to properly detect incomplete
// "case" and "default" keywords as inside the switch statement. For
// example:
//
//   switch {
//   def<>
//   }
//
// gets parsed like:
//
//   switch {
//   }
//
// Later we manually pull out the "def" token, but we need to detect
// that our "<>" position is inside the switch block. To do that we
// move the curly brace so it looks like:
//
//   switch {
//
//   }
//
func fixEmptySwitch(body *ast.BlockStmt, tok *token.File, src []byte) {
	// We only care about empty switch statements.
	if len(body.List) > 0 || !body.Rbrace.IsValid() {
		return
	}

	// If the right brace is actually in the source code at the
	// specified position, don't mess with it.
	braceOffset := tok.Offset(body.Rbrace)
	if braceOffset < len(src) && src[braceOffset] == '}' {
		return
	}

	braceLine := tok.Line(body.Rbrace)
	if braceLine >= tok.LineCount() {
		// If we are the last line in the file, no need to fix anything.
		return
	}

	// Move the right brace down one line.
	body.Rbrace = tok.LineStart(braceLine + 1)
}

// fixDanglingSelector inserts real "_" selector expressions in place
// of phantom "_" selectors. For example:
//
// func _() {
//   x.<>
// }
// var x struct { i int }
//
// To fix completion at "<>", we insert a real "_" after the "." so the
// following declaration of "x" can be parsed and type checked
// normally.
func fixDanglingSelector(s *ast.SelectorExpr, tok *token.File, src []byte) []byte {
	if !isPhantomUnderscore(s.Sel, tok, src) {
		return nil
	}

	if !s.X.End().IsValid() {
		return nil
	}

	// Insert directly after the selector's ".".
	insertOffset := tok.Offset(s.X.End()) + 1
	if src[insertOffset-1] != '.' {
		return nil
	}

	var buf bytes.Buffer
	buf.Grow(len(src) + 1)
	buf.Write(src[:insertOffset])
	buf.WriteByte('_')
	buf.Write(src[insertOffset:])
	return buf.Bytes()
}

// fixPhantomSelector tries to fix selector expressions with phantom
// "_" selectors. In particular, we check if the selector is a
// keyword, and if so we swap in an *ast.Ident with the keyword text. For example:
//
// foo.var
//
// yields a "_" selector instead of "var" since "var" is a keyword.
func fixPhantomSelector(sel *ast.SelectorExpr, tok *token.File, src []byte) {
	if !isPhantomUnderscore(sel.Sel, tok, src) {
		return
	}

	// Only consider selectors directly abutting the selector ".". This
	// avoids false positives in cases like:
	//
	//   foo. // don't think "var" is our selector
	//   var bar = 123
	//
	if sel.Sel.Pos() != sel.X.End()+1 {
		return
	}

	maybeKeyword := readKeyword(sel.Sel.Pos(), tok, src)
	if maybeKeyword == "" {
		return
	}

	replaceNode(sel, sel.Sel, &ast.Ident{
		Name:    maybeKeyword,
		NamePos: sel.Sel.Pos(),
	})
}

// isPhantomUnderscore reports whether the given ident is a phantom
// underscore. The parser sometimes inserts phantom underscores when
// it encounters otherwise unparseable situations.
func isPhantomUnderscore(id *ast.Ident, tok *token.File, src []byte) bool {
	if id == nil || id.Name != "_" {
		return false
	}

	// Phantom underscore means the underscore is not actually in the
	// program text.
	offset := tok.Offset(id.Pos())
	return len(src) <= offset || src[offset] != '_'
}

// fixInitStmt fixes cases where the parser misinterprets an
// if/for/switch "init" statement as the "cond" conditional. In cases
// like "if i := 0" the user hasn't typed the semicolon yet so the
// parser is looking for the conditional expression. However, "i := 0"
// are not valid expressions, so we get a BadExpr.
func fixInitStmt(bad *ast.BadExpr, parent ast.Node, tok *token.File, src []byte) {
	if !bad.Pos().IsValid() || !bad.End().IsValid() {
		return
	}

	// Try to extract a statement from the BadExpr.
	stmtBytes := src[tok.Offset(bad.Pos()) : tok.Offset(bad.End()-1)+1]
	stmt, err := parseStmt(bad.Pos(), stmtBytes)
	if err != nil {
		return
	}

	// If the parent statement doesn't already have an "init" statement,
	// move the extracted statement into the "init" field and insert a
	// dummy expression into the required "cond" field.
	switch p := parent.(type) {
	case *ast.IfStmt:
		if p.Init != nil {
			return
		}
		p.Init = stmt
		p.Cond = &ast.Ident{
			Name:    "_",
			NamePos: stmt.End(),
		}
	case *ast.ForStmt:
		if p.Init != nil {
			return
		}
		p.Init = stmt
		p.Cond = &ast.Ident{
			Name:    "_",
			NamePos: stmt.End(),
		}
	case *ast.SwitchStmt:
		if p.Init != nil {
			return
		}
		p.Init = stmt
		p.Tag = nil
	}
}

// readKeyword reads the keyword starting at pos, if any.
func readKeyword(pos token.Pos, tok *token.File, src []byte) string {
	var kwBytes []byte
	for i := tok.Offset(pos); i < len(src); i++ {
		// Use a simplified identifier check since keywords are always lowercase ASCII.
		if src[i] < 'a' || src[i] > 'z' {
			break
		}
		kwBytes = append(kwBytes, src[i])

		// Stop search at arbitrarily chosen too-long-for-a-keyword length.
		if len(kwBytes) > 15 {
			return ""
		}
	}

	if kw := string(kwBytes); token.Lookup(kw).IsKeyword() {
		return kw
	}

	return ""
}

// fixArrayType tries to parse an *ast.BadExpr into an *ast.ArrayType.
// go/parser often turns lone array types like "[]int" into BadExprs
// if it isn't expecting a type.
func fixArrayType(bad *ast.BadExpr, parent ast.Node, tok *token.File, src []byte) bool {
	// Our expected input is a bad expression that looks like "[]someExpr".

	from := bad.Pos()
	to := bad.End()

	if !from.IsValid() || !to.IsValid() {
		return false
	}

	exprBytes := make([]byte, 0, int(to-from)+3)
	// Avoid doing tok.Offset(to) since that panics if badExpr ends at EOF.
	exprBytes = append(exprBytes, src[tok.Offset(from):tok.Offset(to-1)+1]...)
	exprBytes = bytes.TrimSpace(exprBytes)

	// If our expression ends in "]" (e.g. "[]"), add a phantom selector
	// so we can complete directly after the "[]".
	if len(exprBytes) > 0 && exprBytes[len(exprBytes)-1] == ']' {
		exprBytes = append(exprBytes, '_')
	}

	// Add "{}" to turn our ArrayType into a CompositeLit. This is to
	// handle the case of "[...]int" where we must make it a composite
	// literal to be parseable.
	exprBytes = append(exprBytes, '{', '}')

	expr, err := parseExpr(from, exprBytes)
	if err != nil {
		return false
	}

	cl, _ := expr.(*ast.CompositeLit)
	if cl == nil {
		return false
	}

	at, _ := cl.Type.(*ast.ArrayType)
	if at == nil {
		return false
	}

	return replaceNode(parent, bad, at)
}

// precedingToken scans src to find the token preceding pos.
func precedingToken(pos token.Pos, tok *token.File, src []byte) token.Token {
	s := &scanner.Scanner{}
	s.Init(tok, src, nil, 0)

	var lastTok token.Token
	for {
		p, t, _ := s.Scan()
		if t == token.EOF || p >= pos {
			break
		}

		lastTok = t
	}
	return lastTok
}

// fixDeferOrGoStmt tries to parse an *ast.BadStmt into a defer or a go statement.
//
// go/parser packages a statement of the form "defer x." as an *ast.BadStmt because
// it does not include a call expression. This means that go/types skips type-checking
// this statement entirely, and we can't use the type information when completing.
// Here, we try to generate a fake *ast.DeferStmt or *ast.GoStmt to put into the AST,
// instead of the *ast.BadStmt.
func fixDeferOrGoStmt(bad *ast.BadStmt, parent ast.Node, tok *token.File, src []byte) bool {
	// Check if we have a bad statement containing either a "go" or "defer".
	s := &scanner.Scanner{}
	s.Init(tok, src, nil, 0)

	var (
		pos token.Pos
		tkn token.Token
	)
	for {
		if tkn == token.EOF {
			return false
		}
		if pos >= bad.From {
			break
		}
		pos, tkn, _ = s.Scan()
	}

	var stmt ast.Stmt
	switch tkn {
	case token.DEFER:
		stmt = &ast.DeferStmt{
			Defer: pos,
		}
	case token.GO:
		stmt = &ast.GoStmt{
			Go: pos,
		}
	default:
		return false
	}

	var (
		from, to, last   token.Pos
		lastToken        token.Token
		braceDepth       int
		phantomSelectors []token.Pos
	)
FindTo:
	for {
		to, tkn, _ = s.Scan()

		if from == token.NoPos {
			from = to
		}

		switch tkn {
		case token.EOF:
			break FindTo
		case token.SEMICOLON:
			// If we aren't in nested braces, end of statement means
			// end of expression.
			if braceDepth == 0 {
				break FindTo
			}
		case token.LBRACE:
			braceDepth++
		}

		// This handles the common dangling selector case. For example in
		//
		// defer fmt.
		// y := 1
		//
		// we notice the dangling period and end our expression.
		//
		// If the previous token was a "." and we are looking at a "}",
		// the period is likely a dangling selector and needs a phantom
		// "_". Likewise if the current token is on a different line than
		// the period, the period is likely a dangling selector.
		if lastToken == token.PERIOD && (tkn == token.RBRACE || tok.Line(to) > tok.Line(last)) {
			// Insert phantom "_" selector after the dangling ".".
			phantomSelectors = append(phantomSelectors, last+1)
			// If we aren't in a block then end the expression after the ".".
			if braceDepth == 0 {
				to = last + 1
				break
			}
		}

		lastToken = tkn
		last = to

		switch tkn {
		case token.RBRACE:
			braceDepth--
			if braceDepth <= 0 {
				if braceDepth == 0 {
					// +1 to include the "}" itself.
					to += 1
				}
				break FindTo
			}
		}
	}

	if !from.IsValid() || tok.Offset(from) >= len(src) {
		return false
	}

	if !to.IsValid() || tok.Offset(to) >= len(src) {
		return false
	}

	// Insert any phantom selectors needed to prevent dangling "." from messing
	// up the AST.
	exprBytes := make([]byte, 0, int(to-from)+len(phantomSelectors))
	for i, b := range src[tok.Offset(from):tok.Offset(to)] {
		if len(phantomSelectors) > 0 && from+token.Pos(i) == phantomSelectors[0] {
			exprBytes = append(exprBytes, '_')
			phantomSelectors = phantomSelectors[1:]
		}
		exprBytes = append(exprBytes, b)
	}

	if len(phantomSelectors) > 0 {
		exprBytes = append(exprBytes, '_')
	}

	expr, err := parseExpr(from, exprBytes)
	if err != nil {
		return false
	}

	// Package the expression into a fake *ast.CallExpr and re-insert
	// into the function.
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

	return replaceNode(parent, bad, stmt)
}

// parseStmt parses the statement in src and updates its position to
// start at pos.
func parseStmt(pos token.Pos, src []byte) (ast.Stmt, error) {
	// Wrap our expression to make it a valid Go file we can pass to ParseFile.
	fileSrc := bytes.Join([][]byte{
		[]byte("package fake;func _(){"),
		src,
		[]byte("}"),
	}, nil)

	// Use ParseFile instead of ParseExpr because ParseFile has
	// best-effort behavior, whereas ParseExpr fails hard on any error.
	fakeFile, err := parser.ParseFile(token.NewFileSet(), "", fileSrc, 0)
	if fakeFile == nil {
		return nil, errors.Errorf("error reading fake file source: %v", err)
	}

	// Extract our expression node from inside the fake file.
	if len(fakeFile.Decls) == 0 {
		return nil, errors.Errorf("error parsing fake file: %v", err)
	}

	fakeDecl, _ := fakeFile.Decls[0].(*ast.FuncDecl)
	if fakeDecl == nil || len(fakeDecl.Body.List) == 0 {
		return nil, errors.Errorf("no statement in %s: %v", src, err)
	}

	stmt := fakeDecl.Body.List[0]

	// parser.ParseFile returns undefined positions.
	// Adjust them for the current file.
	offsetPositions(stmt, pos-1-(stmt.Pos()-1))

	return stmt, nil
}

// parseExpr parses the expression in src and updates its position to
// start at pos.
func parseExpr(pos token.Pos, src []byte) (ast.Expr, error) {
	stmt, err := parseStmt(pos, src)
	if err != nil {
		return nil, err
	}

	exprStmt, ok := stmt.(*ast.ExprStmt)
	if !ok {
		return nil, errors.Errorf("no expr in %s: %v", src, err)
	}

	return exprStmt.X, nil
}

var tokenPosType = reflect.TypeOf(token.NoPos)

// offsetPositions applies an offset to the positions in an ast.Node.
func offsetPositions(n ast.Node, offset token.Pos) {
	ast.Inspect(n, func(n ast.Node) bool {
		if n == nil {
			return false
		}

		v := reflect.ValueOf(n).Elem()

		switch v.Kind() {
		case reflect.Struct:
			for i := 0; i < v.NumField(); i++ {
				f := v.Field(i)
				if f.Type() != tokenPosType {
					continue
				}

				if !f.CanSet() {
					continue
				}

				f.SetInt(f.Int() + int64(offset))
			}
		}

		return true
	})
}

// replaceNode updates parent's child oldChild to be newChild. It
// returns whether it replaced successfully.
func replaceNode(parent, oldChild, newChild ast.Node) bool {
	if parent == nil || oldChild == nil || newChild == nil {
		return false
	}

	parentVal := reflect.ValueOf(parent).Elem()
	if parentVal.Kind() != reflect.Struct {
		return false
	}

	newChildVal := reflect.ValueOf(newChild)

	tryReplace := func(v reflect.Value) bool {
		if !v.CanSet() || !v.CanInterface() {
			return false
		}

		// If the existing value is oldChild, we found our child. Make
		// sure our newChild is assignable and then make the swap.
		if v.Interface() == oldChild && newChildVal.Type().AssignableTo(v.Type()) {
			v.Set(newChildVal)
			return true
		}

		return false
	}

	// Loop over parent's struct fields.
	for i := 0; i < parentVal.NumField(); i++ {
		f := parentVal.Field(i)

		switch f.Kind() {
		// Check interface and pointer fields.
		case reflect.Interface, reflect.Ptr:
			if tryReplace(f) {
				return true
			}

		// Search through any slice fields.
		case reflect.Slice:
			for i := 0; i < f.Len(); i++ {
				if tryReplace(f.Index(i)) {
					return true
				}
			}
		}
	}

	return false
}
