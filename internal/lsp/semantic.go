// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lsp

import (
	"bytes"
	"context"
	"fmt"
	"go/ast"
	"go/token"
	"go/types"
	"log"
	"runtime"
	"sort"
	"strings"
	"time"

	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/lsp/source"
	errors "golang.org/x/xerrors"
)

func (s *Server) semanticTokensFull(ctx context.Context, p *protocol.SemanticTokensParams) (*protocol.SemanticTokens, error) {
	ret, err := s.computeSemanticTokens(ctx, p.TextDocument, nil)
	return ret, err
}

func (s *Server) semanticTokensFullDelta(ctx context.Context, p *protocol.SemanticTokensDeltaParams) (interface{}, error) {
	return nil, errors.Errorf("implement SemanticTokensFullDelta")
}

func (s *Server) semanticTokensRange(ctx context.Context, p *protocol.SemanticTokensRangeParams) (*protocol.SemanticTokens, error) {
	ret, err := s.computeSemanticTokens(ctx, p.TextDocument, &p.Range)
	return ret, err
}

func (s *Server) semanticTokensRefresh(ctx context.Context) error {
	// in the code, but not in the protocol spec
	return errors.Errorf("implement SemanticTokensRefresh")
}

func (s *Server) computeSemanticTokens(ctx context.Context, td protocol.TextDocumentIdentifier, rng *protocol.Range) (*protocol.SemanticTokens, error) {
	ans := protocol.SemanticTokens{
		Data: []float64{},
	}
	snapshot, _, ok, release, err := s.beginFileRequest(ctx, td.URI, source.Go)
	defer release()
	if !ok {
		return nil, err
	}
	vv := snapshot.View()
	if !vv.Options().SemanticTokens {
		// return an error, so if the option changes
		// the client won't remember the wrong answer
		return nil, errors.Errorf("semantictokens are disabled")
	}
	pkg, err := snapshot.PackageForFile(ctx, td.URI.SpanURI(), source.TypecheckFull, source.WidestPackage)
	if err != nil {
		return nil, err
	}
	info := pkg.GetTypesInfo()
	pgf, err := pkg.File(td.URI.SpanURI())
	if err != nil {
		return nil, err
	}
	if pgf.ParseErr != nil {
		return nil, pgf.ParseErr
	}
	e := &encoded{
		pgf:  pgf,
		rng:  rng,
		ti:   info,
		fset: snapshot.FileSet(),
	}
	if err := e.init(); err != nil {
		return nil, err
	}
	e.semantics()
	ans.Data, err = e.Data()
	if err != nil {
		// this is an internal error, likely caused by a typo
		// for a token or modifier
		return nil, err
	}
	// for small cache, some day. for now, the client ignores this
	ans.ResultID = fmt.Sprintf("%v", time.Now())
	return &ans, nil
}

func (e *encoded) semantics() {
	f := e.pgf.File
	e.token(f.Package, len("package"), tokKeyword, nil)
	e.token(f.Name.NamePos, len(f.Name.Name), tokNamespace, nil)
	inspect := func(n ast.Node) bool {
		return e.inspector(n)
	}
	for _, d := range f.Decls {
		// only look at the decls that overlap the range
		start, end := d.Pos(), d.End()
		if end <= e.start || start >= e.end {
			continue
		}
		ast.Inspect(d, inspect)
	}
}

type tokenType string

const (
	tokNamespace tokenType = "namespace"
	tokType      tokenType = "type"
	tokInterface tokenType = "interface"
	tokParameter tokenType = "parameter"
	tokVariable  tokenType = "variable"
	tokMember    tokenType = "member"
	tokFunction  tokenType = "function"
	tokKeyword   tokenType = "keyword"
	tokComment   tokenType = "comment"
	tokString    tokenType = "string"
	tokNumber    tokenType = "number"
	tokOperator  tokenType = "operator"
)

var lastPosition token.Position

func (e *encoded) token(start token.Pos, leng int, typ tokenType, mods []string) {
	if start == 0 {
		// Temporary, pending comprehensive tests
		log.Printf("SAW token.NoPos")
		for i := 0; i < 6; i++ {
			_, f, l, ok := runtime.Caller(i)
			if !ok {
				break
			}
			log.Printf("%d: %s:%d", i, f, l)
		}
	}
	if start >= e.end || start+token.Pos(leng) <= e.start {
		return
	}
	// want a line and column from start (in LSP coordinates)
	rng := source.NewMappedRange(e.fset, e.pgf.Mapper, start, start+token.Pos(leng))
	lspRange, err := rng.Range()
	if err != nil {
		log.Printf("failed to convert to range %v", err)
	}
	// token is all on one line
	length := lspRange.End.Character - lspRange.Start.Character
	e.add(lspRange.Start.Line, lspRange.Start.Character, length, typ, mods)
}

func (e *encoded) add(line, start float64, len float64, tok tokenType, mod []string) {
	x := semItem{line, start, len, tok, mod}
	e.items = append(e.items, x)
}

// semItem represents a token found walking the parse tree
type semItem struct {
	line, start float64
	len         float64
	typeStr     tokenType
	mods        []string
}

type encoded struct {
	// the generated data
	items []semItem

	pgf  *source.ParsedGoFile
	rng  *protocol.Range
	ti   *types.Info
	fset *token.FileSet
	// allowed starting and ending token.Pos, set by init
	// used to avoid looking at declarations not in range
	start, end token.Pos
	// path from the root of the parse tree, used for debugging
	stack []ast.Node
}

// convert the stack to a string, for debugging
func (e *encoded) strStack() string {
	msg := []string{"["}
	for _, s := range e.stack {
		msg = append(msg, fmt.Sprintf("%T", s)[5:])
	}
	if len(e.stack) > 0 {
		loc := e.stack[len(e.stack)-1].Pos()
		add := e.pgf.Tok.PositionFor(loc, false)
		msg = append(msg, fmt.Sprintf("(%d:%d)", add.Line, add.Column))
	}
	msg = append(msg, "]")
	return strings.Join(msg, " ")
}

func (e *encoded) inspector(n ast.Node) bool {
	pop := func() {
		e.stack = e.stack[:len(e.stack)-1]
	}
	if n == nil {
		pop()
		return true
	}
	e.stack = append(e.stack, n)
	switch x := n.(type) {
	case *ast.ArrayType:
	case *ast.AssignStmt:
		e.token(x.TokPos, len(x.Tok.String()), tokOperator, nil)
	case *ast.BasicLit:
		// if it extends across a line, skip it
		// better would be to mark each line as string TODO(pjw)
		if strings.Contains(x.Value, "\n") {
			break
		}
		ln := len(x.Value)
		what := tokNumber
		if x.Kind == token.STRING {
			what = tokString
			if _, ok := e.stack[len(e.stack)-2].(*ast.Field); ok {
				// struct tags (this is probably pointless, as the
				// TextMate grammar will treat all the other comments the same)
				what = tokComment
			}
		}
		e.token(x.Pos(), ln, what, nil)
	case *ast.BinaryExpr:
		e.token(x.OpPos, len(x.Op.String()), tokOperator, nil)
	case *ast.BlockStmt:
	case *ast.BranchStmt:
		e.token(x.TokPos, len(x.Tok.String()), tokKeyword, nil)
		// There's no semantic encoding for labels
	case *ast.CallExpr:
		if x.Ellipsis != token.NoPos {
			e.token(x.Ellipsis, len("..."), tokOperator, nil)
		}
	case *ast.CaseClause:
		iam := "case"
		if x.List == nil {
			iam = "default"
		}
		e.token(x.Case, len(iam), tokKeyword, nil)
	case *ast.ChanType:
		// chan | chan <- | <- chan
		if x.Arrow == token.NoPos || x.Arrow != x.Begin {
			e.token(x.Begin, len("chan"), tokKeyword, nil)
			break
		}
		pos := e.findKeyword("chan", x.Begin+2, x.Value.Pos())
		e.token(pos, len("chan"), tokKeyword, nil)
	case *ast.CommClause:
		iam := len("case")
		if x.Comm == nil {
			iam = len("default")
		}
		e.token(x.Case, iam, tokKeyword, nil)
	case *ast.CompositeLit:
	case *ast.DeclStmt:
	case *ast.DeferStmt:
		e.token(x.Defer, len("defer"), tokKeyword, nil)
	case *ast.Ellipsis:
		e.token(x.Ellipsis, len("..."), tokOperator, nil)
	case *ast.EmptyStmt:
	case *ast.ExprStmt:
	case *ast.Field:
	case *ast.FieldList:
	case *ast.ForStmt:
		e.token(x.For, len("for"), tokKeyword, nil)
	case *ast.FuncDecl:
	case *ast.FuncLit:
	case *ast.FuncType:
		if x.Func != token.NoPos {
			e.token(x.Func, len("func"), tokKeyword, nil)
		}
	case *ast.GenDecl:
		e.token(x.TokPos, len(x.Tok.String()), tokKeyword, nil)
	case *ast.GoStmt:
		e.token(x.Go, len("go"), tokKeyword, nil)
	case *ast.Ident:
		e.ident(x)
	case *ast.IfStmt:
		e.token(x.If, len("if"), tokKeyword, nil)
		if x.Else != nil {
			// x.Body.End() or x.Body.End()+1, not that it matters
			pos := e.findKeyword("else", x.Body.End(), x.Else.Pos())
			e.token(pos, len("else"), tokKeyword, nil)
		}
	case *ast.ImportSpec:
		e.importSpec(x)
		pop()
		return false
	case *ast.IncDecStmt:
		e.token(x.TokPos, len(x.Tok.String()), tokOperator, nil)
	case *ast.IndexExpr:
	case *ast.InterfaceType:
		e.token(x.Interface, len("interface"), tokKeyword, nil)
	case *ast.KeyValueExpr:
	case *ast.LabeledStmt:
	case *ast.MapType:
		e.token(x.Map, len("map"), tokKeyword, nil)
	case *ast.ParenExpr:
	case *ast.RangeStmt:
		e.token(x.For, len("for"), tokKeyword, nil)
		// x.TokPos == token.NoPos should mean a syntax error
		pos := e.findKeyword("range", x.TokPos, x.X.Pos())
		e.token(pos, len("range"), tokKeyword, nil)
	case *ast.ReturnStmt:
		e.token(x.Return, len("return"), tokKeyword, nil)
	case *ast.SelectStmt:
		e.token(x.Select, len("select"), tokKeyword, nil)
	case *ast.SelectorExpr:
	case *ast.SendStmt:
		e.token(x.Arrow, len("<-"), tokOperator, nil)
	case *ast.SliceExpr:
	case *ast.StarExpr:
		e.token(x.Star, len("*"), tokOperator, nil)
	case *ast.StructType:
		e.token(x.Struct, len("struct"), tokKeyword, nil)
	case *ast.SwitchStmt:
		e.token(x.Switch, len("switch"), tokKeyword, nil)
	case *ast.TypeAssertExpr:
		if x.Type == nil {
			pos := e.findKeyword("type", x.Lparen, x.Rparen)
			e.token(pos, len("type"), tokKeyword, nil)
		}
	case *ast.TypeSpec:
	case *ast.TypeSwitchStmt:
		e.token(x.Switch, len("switch"), tokKeyword, nil)
	case *ast.UnaryExpr:
		e.token(x.OpPos, len(x.Op.String()), tokOperator, nil)
	case *ast.ValueSpec:
	// things we won't see
	case *ast.BadDecl, *ast.BadExpr, *ast.BadStmt,
		*ast.File, *ast.Package:
		log.Printf("implement %T %s", x, e.pgf.Tok.PositionFor(x.Pos(), false))
	// things we knowingly ignore
	case *ast.Comment, *ast.CommentGroup:
		pop()
		return false
	default: // just to be super safe.
		panic(fmt.Sprintf("failed to implement %T", x))
	}
	return true
}
func (e *encoded) ident(x *ast.Ident) {
	def := e.ti.Defs[x]
	if def != nil {
		what, mods := e.definitionFor(x)
		if what != "" {
			e.token(x.Pos(), len(x.String()), what, mods)
		}
		return
	}
	use := e.ti.Uses[x]
	switch y := use.(type) {
	case nil:
		e.token(x.NamePos, len(x.Name), tokVariable, []string{"definition"})
	case *types.Builtin:
		e.token(x.NamePos, len(x.Name), tokFunction, []string{"defaultLibrary"})
	case *types.Const:
		mods := []string{"readonly"}
		tt := y.Type()
		if ttx, ok := tt.(*types.Basic); ok {
			switch bi := ttx.Info(); {
			case bi&types.IsNumeric != 0:
				me := tokVariable
				if x.String() == "iota" {
					me = tokKeyword
					mods = []string{}
				}
				e.token(x.Pos(), len(x.String()), me, mods)
			case bi&types.IsString != 0:
				e.token(x.Pos(), len(x.String()), tokString, mods)
			case bi&types.IsBoolean != 0:
				e.token(x.Pos(), len(x.Name), tokKeyword, nil)
			case bi == 0:
				// nothing to say
			default:
				// replace with panic after extensive testing
				log.Printf("unexpected %x at %s", bi, e.pgf.Tok.PositionFor(x.Pos(), false))
			}
			break
		}
		if ttx, ok := tt.(*types.Named); ok {
			if x.String() == "iota" {
				log.Printf("ttx:%T", ttx)
			}
			if _, ok := ttx.Underlying().(*types.Basic); ok {
				e.token(x.Pos(), len("nil"), tokVariable, mods)
				break
			}
			// can this happen?
			log.Printf("unexpectd %q/%T", x.String(), tt)
			e.token(x.Pos(), len(x.String()), tokVariable, nil)
			break
		}
		// can this happen? Don't think so
		log.Printf("%s %T %#v", x.String(), tt, tt)
	case *types.Func:
		e.token(x.Pos(), len(x.Name), tokFunction, nil)
	case *types.Label:
		// nothing to map it to
	case *types.Nil:
		// nil is a predeclared identifier
		e.token(x.Pos(), 3, tokKeyword, []string{"readonly"})
	case *types.PkgName:
		e.token(x.Pos(), len(x.Name), tokNamespace, nil)
	case *types.TypeName:
		e.token(x.Pos(), len(x.String()), tokType, nil)
	case *types.Var:
		e.token(x.Pos(), len(x.Name), tokVariable, nil)
	default:
		// replace with panic after extensive testing
		if use == nil {
			log.Printf("HOW did we get here? %#v/%#v %#v %#v", x, x.Obj, e.ti.Defs[x], e.ti.Uses[x])
		}
		if use.Type() != nil {
			log.Printf("%s %T/%T,%#v", x.String(), use, use.Type(), use)
		} else {
			log.Printf("%s %T", x.String(), use)
		}
	}
}

func (e *encoded) definitionFor(x *ast.Ident) (tokenType, []string) {
	mods := []string{"definition"}
	for i := len(e.stack) - 1; i >= 0; i-- {
		s := e.stack[i]
		switch y := s.(type) {
		case *ast.AssignStmt, *ast.RangeStmt:
			if x.Name == "_" {
				return "", nil // not really a variable
			}
			return "variable", mods
		case *ast.GenDecl:
			if y.Tok == token.CONST {
				mods = append(mods, "readonly")
			}
			return tokVariable, mods
		case *ast.FuncDecl:
			// If x is immediately under a FuncDecl, it is a function or method
			if i == len(e.stack)-2 {
				if y.Recv != nil {
					return tokMember, mods
				}
				return tokFunction, mods
			}
			// if x < ... < FieldList < FuncDecl, this is the receiver, a variable
			if _, ok := e.stack[i+1].(*ast.FieldList); ok {
				return tokVariable, nil
			}
			// if x < ... < FieldList < FuncType < FuncDecl, this is a param
			return tokParameter, mods
		case *ast.InterfaceType:
			return tokMember, mods
		case *ast.TypeSpec:
			return tokType, mods
		}
	}
	// panic after extensive testing
	log.Printf("failed to find the decl for %s", e.pgf.Tok.PositionFor(x.Pos(), false))
	return "", []string{""}
}

// findKeyword finds a keyword rather than guessing its location
func (e *encoded) findKeyword(keyword string, start, end token.Pos) token.Pos {
	offset := int(start) - e.pgf.Tok.Base()
	last := int(end) - e.pgf.Tok.Base()
	buf := e.pgf.Src
	idx := bytes.Index(buf[offset:last], []byte(keyword))
	if idx != -1 {
		return start + token.Pos(idx)
	}
	// can't happen
	return token.NoPos
}

func (e *encoded) init() error {
	e.start = token.Pos(e.pgf.Tok.Base())
	e.end = e.start + token.Pos(e.pgf.Tok.Size())
	if e.rng == nil {
		return nil
	}
	span, err := e.pgf.Mapper.RangeSpan(*e.rng)
	if err != nil {
		return errors.Errorf("range span error for %s", e.pgf.File.Name)
	}
	e.end = e.start + token.Pos(span.End().Offset())
	e.start += token.Pos(span.Start().Offset())
	return nil
}

func (e *encoded) Data() ([]float64, error) {
	// binary operators, at least, will be out of order
	sort.Slice(e.items, func(i, j int) bool {
		if e.items[i].line != e.items[j].line {
			return e.items[i].line < e.items[j].line
		}
		return e.items[i].start < e.items[j].start
	})
	// each semantic token needs five values
	// (see Integer Encoding for Tokens in the LSP spec)
	x := make([]float64, 5*len(e.items))
	for i := 0; i < len(e.items); i++ {
		j := 5 * i
		if i == 0 {
			x[0] = e.items[0].line
		} else {
			x[j] = e.items[i].line - e.items[i-1].line
		}
		x[j+1] = e.items[i].start
		if i > 0 && e.items[i].line == e.items[i-1].line {
			x[j+1] = e.items[i].start - e.items[i-1].start
		}
		x[j+2] = e.items[i].len
		x[j+3] = float64(SemanticMemo.TypeMap[e.items[i].typeStr])
		mask := 0
		for _, s := range e.items[i].mods {
			mask |= SemanticMemo.ModMap[s]
		}
		x[j+4] = float64(mask)
	}
	return x, nil
}

func (e *encoded) importSpec(d *ast.ImportSpec) {
	// a local package name or the last component of the Path
	if d.Name != nil {
		nm := d.Name.String()
		// import . x => x is not a namespace
		// import _ x => x is a namespace
		if nm != "_" && nm != "." {
			e.token(d.Name.Pos(), len(nm), tokNamespace, nil)
			return
		}
		if nm == "." {
			return
		}
		// and fall through for _
	}
	nm := d.Path.Value[1 : len(d.Path.Value)-1] // trailing "
	v := strings.LastIndex(nm, "/")
	if v != -1 {
		nm = nm[v+1:]
	}
	start := d.Path.End() - token.Pos(1+len(nm))
	e.token(start, len(nm), tokNamespace, nil)
}

// SemMemo supports semantic token translations between numbers and strings
type SemMemo struct {
	tokTypes, tokMods []string
	// these exported fields are used in the 'gopls semtok' command
	TypeMap map[tokenType]int
	ModMap  map[string]int
}

var SemanticMemo *SemMemo

// Type returns a string equivalent of the type, for gopls semtok
func (m *SemMemo) Type(n int) string {
	if n >= 0 && n < len(m.tokTypes) {
		return m.tokTypes[n]
	}
	return fmt.Sprintf("?%d[%d,%d]?", n, len(m.tokTypes), len(m.tokMods))
}

// Mods returns the []string equivalent of the mods, for gopls semtok.
func (m *SemMemo) Mods(n int) []string {
	mods := []string{}
	for i := 0; i < len(m.tokMods); i++ {
		if (n & (1 << uint(i))) != 0 {
			mods = append(mods, m.tokMods[i])
		}
	}
	return mods
}

// save what the client sent
func rememberToks(toks []string, mods []string) ([]string, []string) {
	SemanticMemo = &SemMemo{
		tokTypes: toks,
		tokMods:  mods,
		TypeMap:  make(map[tokenType]int),
		ModMap:   make(map[string]int),
	}
	for i, t := range toks {
		SemanticMemo.TypeMap[tokenType(t)] = i
	}
	for i, m := range mods {
		SemanticMemo.ModMap[m] = 1 << uint(i)
	}
	// we could have pruned or rearranged them.
	// But then change the list in cmd.go too
	return SemanticMemo.tokTypes, SemanticMemo.tokMods
}
