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

	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/lsp/telemetry"
	"golang.org/x/tools/internal/memoize"
	"golang.org/x/tools/internal/span"
	"golang.org/x/tools/internal/telemetry/trace"
	errors "golang.org/x/xerrors"
)

// Limits the number of parallel parser calls per process.
var parseLimit = make(chan struct{}, 20)

// parseKey uniquely identifies a parsed Go file.
type parseKey struct {
	file source.FileIdentity
	mode source.ParseMode
}

type parseGoHandle struct {
	handle *memoize.Handle
	file   source.FileHandle
	mode   source.ParseMode
}

type parseGoData struct {
	memoize.NoCopy

	src        []byte
	ast        *ast.File
	parseError error // errors associated with parsing the file
	mapper     *protocol.ColumnMapper
	err        error
}

func (c *Cache) ParseGoHandle(fh source.FileHandle, mode source.ParseMode) source.ParseGoHandle {
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
	}
}

func (pgh *parseGoHandle) String() string {
	return pgh.File().Identity().URI.Filename()
}

func (pgh *parseGoHandle) File() source.FileHandle {
	return pgh.file
}

func (pgh *parseGoHandle) Mode() source.ParseMode {
	return pgh.mode
}

func (pgh *parseGoHandle) Parse(ctx context.Context) (*ast.File, []byte, *protocol.ColumnMapper, error, error) {
	v := pgh.handle.Get(ctx)
	if v == nil {
		return nil, nil, nil, nil, errors.Errorf("no parsed file for %s", pgh.File().Identity().URI)
	}
	data := v.(*parseGoData)
	return data.ast, data.src, data.mapper, data.parseError, data.err
}

func (pgh *parseGoHandle) Cached() (*ast.File, []byte, *protocol.ColumnMapper, error, error) {
	v := pgh.handle.Cached()
	if v == nil {
		return nil, nil, nil, nil, errors.Errorf("no cached AST for %s", pgh.file.Identity().URI)
	}
	data := v.(*parseGoData)
	return data.ast, data.src, data.mapper, data.parseError, data.err
}

func hashParseKey(ph source.ParseGoHandle) string {
	b := bytes.NewBuffer(nil)
	b.WriteString(ph.File().Identity().String())
	b.WriteString(string(ph.Mode()))
	return hashContents(b.Bytes())
}

func hashParseKeys(phs []source.ParseGoHandle) string {
	b := bytes.NewBuffer(nil)
	for _, ph := range phs {
		b.WriteString(hashParseKey(ph))
	}
	return hashContents(b.Bytes())
}

func parseGo(ctx context.Context, fset *token.FileSet, fh source.FileHandle, mode source.ParseMode) *parseGoData {
	ctx, done := trace.StartSpan(ctx, "cache.parseGo", telemetry.File.Of(fh.Identity().URI.Filename()))
	defer done()

	if fh.Identity().Kind != source.Go {
		return &parseGoData{err: errors.Errorf("cannot parse non-Go file %s", fh.Identity().URI)}
	}
	buf, _, err := fh.Read(ctx)
	if err != nil {
		return &parseGoData{err: err}
	}
	parseLimit <- struct{}{}
	defer func() { <-parseLimit }()
	parserMode := parser.AllErrors | parser.ParseComments
	if mode == source.ParseHeader {
		parserMode = parser.ImportsOnly | parser.ParseComments
	}
	file, parseError := parser.ParseFile(fset, fh.Identity().URI.Filename(), buf, parserMode)
	var tok *token.File
	if file != nil {
		tok = fset.File(file.Pos())
		if tok == nil {
			return &parseGoData{err: errors.Errorf("successfully parsed but no token.File for %s (%v)", fh.Identity().URI, parseError)}
		}

		// Fix any badly parsed parts of the AST.
		_ = fixAST(ctx, file, tok, buf)

		// Fix certain syntax errors that render the file unparseable.
		newSrc := fixSrc(file, tok, buf)
		if newSrc != nil {
			newFile, _ := parser.ParseFile(fset, fh.Identity().URI.Filename(), newSrc, parserMode)
			if newFile != nil {
				// Maintain the original parseError so we don't try formatting the doctored file.
				file = newFile
				buf = newSrc
				tok = fset.File(file.Pos())

				_ = fixAST(ctx, file, tok, buf)
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
			err = errors.Errorf("no AST for %s", fh.Identity().URI)
		}
		return &parseGoData{parseError: parseError, err: err}
	}
	m := &protocol.ColumnMapper{
		URI:       fh.Identity().URI,
		Converter: span.NewTokenConverter(fset, tok),
		Content:   buf,
	}

	return &parseGoData{
		src:        buf,
		ast:        file,
		mapper:     m,
		parseError: parseError,
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
func fixAST(ctx context.Context, n ast.Node, tok *token.File, src []byte) error {
	var err error
	walkASTWithParent(n, func(n, parent ast.Node) bool {
		switch n := n.(type) {
		case *ast.BadStmt:
			err = fixDeferOrGoStmt(n, parent, tok, src) // don't shadow err
			if err == nil {
				// Recursively fix in our fixed node.
				err = fixAST(ctx, parent, tok, src)
			} else {
				err = errors.Errorf("unable to parse defer or go from *ast.BadStmt: %v", err)
			}
			return false
		case *ast.BadExpr:
			// Don't propagate this error since *ast.BadExpr is very common
			// and it is only sometimes due to array types. Errors from here
			// are expected and not actionable in general.
			if fixArrayType(n, parent, tok, src) == nil {
				// Recursively fix in our fixed node.
				err = fixAST(ctx, parent, tok, src)
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
		default:
			return true
		}
	})

	return err
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
			newSrc = fixDanglingSelector(f, n, parent, tok, src)
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
	// fix since we aren't mising curlies.
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
func fixDanglingSelector(f *ast.File, s *ast.SelectorExpr, parent ast.Node, tok *token.File, src []byte) []byte {
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
		p.Cond = &ast.Ident{Name: "_"}
	case *ast.ForStmt:
		if p.Init != nil {
			return
		}
		p.Init = stmt
		p.Cond = &ast.Ident{Name: "_"}
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
func fixArrayType(bad *ast.BadExpr, parent ast.Node, tok *token.File, src []byte) error {
	// Our expected input is a bad expression that looks like "[]someExpr".

	from := bad.Pos()
	to := bad.End()

	if !from.IsValid() || !to.IsValid() {
		return errors.Errorf("invalid BadExpr from/to: %d/%d", from, to)
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
		return err
	}

	cl, _ := expr.(*ast.CompositeLit)
	if cl == nil {
		return errors.Errorf("expr not compLit (%T)", expr)
	}

	at, _ := cl.Type.(*ast.ArrayType)
	if at == nil {
		return errors.Errorf("compLit type not array (%T)", cl.Type)
	}

	if !replaceNode(parent, bad, at) {
		return errors.Errorf("couldn't replace array type")
	}

	return nil
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
func fixDeferOrGoStmt(bad *ast.BadStmt, parent ast.Node, tok *token.File, src []byte) error {
	// Check if we have a bad statement containing either a "go" or "defer".
	s := &scanner.Scanner{}
	s.Init(tok, src, nil, 0)

	var (
		pos token.Pos
		tkn token.Token
	)
	for {
		if tkn == token.EOF {
			return errors.Errorf("reached the end of the file")
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
		return errors.Errorf("no defer or go statement found")
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
		return errors.Errorf("invalid from position")
	}

	if !to.IsValid() || tok.Offset(to) >= len(src) {
		return errors.Errorf("invalid to position %d", to)
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
		return err
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

	if !replaceNode(parent, bad, stmt) {
		return errors.Errorf("couldn't replace CallExpr")
	}

	return nil
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
