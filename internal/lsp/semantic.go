// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lsp

import (
	"context"
	"fmt"
	"go/ast"
	"go/types"
	"log"
	"sort"
	"strings"
	"time"

	"golang.org/x/tools/internal/event"
	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/lsp/source"
	errors "golang.org/x/xerrors"
)

func (s *Server) semanticTokensFull(ctx context.Context, p *protocol.SemanticTokensParams) (*protocol.SemanticTokens, error) {
	now := time.Now()
	ret, err := s.computeSemanticTokens(ctx, p.TextDocument, nil)
	if ret != nil && err == nil {
		event.Log(ctx, fmt.Sprintf("Full(%v): %d items for %s in %s",
			s.session.Options().SemanticTokens, len(ret.Data)/5, p.TextDocument.URI.SpanURI().Filename(), time.Since(now)))
	} else {
		event.Error(ctx, fmt.Sprintf("Full failed for %s in %s", p.TextDocument.URI.SpanURI().Filename(), time.Since(now)), err)
	}
	return ret, err
}

func (s *Server) semanticTokensFullDelta(ctx context.Context, p *protocol.SemanticTokensDeltaParams) (interface{}, error) {
	return nil, errors.Errorf("implement SemanticTokensFullDelta")
}

func (s *Server) semanticTokensRange(ctx context.Context, p *protocol.SemanticTokensRangeParams) (*protocol.SemanticTokens, error) {
	now := time.Now()
	ret, err := s.computeSemanticTokens(ctx, p.TextDocument, &p.Range)
	if ret != nil && err == nil {
		event.Log(ctx, fmt.Sprintf("Range(%v): %d items for %s %s in %s",
			s.session.Options().SemanticTokens, len(ret.Data)/5, p.TextDocument.URI.SpanURI().Filename(),
			p.Range, time.Since(now)))
	} else {
		event.Error(ctx, "semanticTokensRange failed", err)
	}
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
	vv, err := s.session.ViewOf(td.URI.SpanURI())
	if err != nil {
		return nil, err
	}
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
	e := &encoded{
		pgf: pgf,
		rng: rng,
		ti:  info,
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
	inspect := func(n ast.Node) bool {
		return e.inspector(n)
	}
	for _, d := range f.Decls {
		// only look at the decls that overlap the range
		start, end := d.Pos(), d.End()
		if int(end) <= e.start || int(start) >= e.end {
			continue
		}
		ast.Inspect(d, inspect)
	}
}

// semItem represents a token found walking the parse tree
type semItem struct {
	line, start, len int
	typeStr          string
	mods             []string
}

type encoded struct {
	// the generated data
	items []semItem

	pgf *source.ParsedGoFile
	rng *protocol.Range
	ti  *types.Info
	// allowed starting and ending token.Pos, set by init
	// used to avoid looking at declarations not in range
	start, end int
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
	// this will be filled in, in the next CL
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
	case *ast.BasicLit:
	case *ast.BinaryExpr:
	case *ast.BlockStmt:
	case *ast.BranchStmt:
	case *ast.CallExpr:
	case *ast.CaseClause:
	case *ast.ChanType:
	case *ast.CommClause:
	case *ast.CompositeLit:
	case *ast.DeclStmt:
	case *ast.DeferStmt:
	case *ast.Ellipsis:
	case *ast.EmptyStmt:
	case *ast.ExprStmt:
	case *ast.Field:
	case *ast.FieldList:
	case *ast.ForStmt:
	case *ast.FuncDecl:
	case *ast.FuncLit:
	case *ast.FuncType:
	case *ast.GenDecl:
	case *ast.GoStmt:
	case *ast.Ident:
	case *ast.IfStmt:
	case *ast.ImportSpec:
		pop()
		return false
	case *ast.IncDecStmt:
	case *ast.IndexExpr:
	case *ast.InterfaceType:
	case *ast.KeyValueExpr:
	case *ast.LabeledStmt:
	case *ast.MapType:
	case *ast.ParenExpr:
	case *ast.RangeStmt:
	case *ast.ReturnStmt:
	case *ast.SelectStmt:
	case *ast.SelectorExpr:
	case *ast.SendStmt:
	case *ast.SliceExpr:
	case *ast.StarExpr:
	case *ast.StructType:
	case *ast.SwitchStmt:
	case *ast.TypeAssertExpr:
	case *ast.TypeSpec:
	case *ast.TypeSwitchStmt:
	case *ast.UnaryExpr:
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

func (e *encoded) init() error {
	e.start = e.pgf.Tok.Base()
	e.end = e.start + e.pgf.Tok.Size()
	if e.rng == nil {
		return nil
	}
	span, err := e.pgf.Mapper.RangeSpan(*e.rng)
	if err != nil {
		return errors.Errorf("range span error for %s", e.pgf.File.Name)
	}
	e.end = e.start + span.End().Offset()
	e.start += span.Start().Offset()
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
	x := make([]int, 5*len(e.items))
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
		x[j+3] = SemanticMemo.typeMap[e.items[i].typeStr]
		for _, s := range e.items[i].mods {
			x[j+4] |= SemanticMemo.modMap[s]
		}
	}
	// As the client never sends these, we could fix the generated
	// type to want []int rather than []float64.
	ans := make([]float64, len(x))
	for i, y := range x {
		ans[i] = float64(y)
	}
	return ans, nil
}

// SemMemo supports semantic token translations between numbers and strings
type SemMemo struct {
	tokTypes, tokMods []string
	typeMap           map[string]int
	modMap            map[string]int
}

var SemanticMemo *SemMemo

// save what the client sent
func rememberToks(toks []string, mods []string) ([]string, []string) {
	SemanticMemo = &SemMemo{
		tokTypes: toks,
		tokMods:  mods,
		typeMap:  make(map[string]int),
		modMap:   make(map[string]int),
	}
	for i, t := range toks {
		SemanticMemo.typeMap[t] = i
	}
	for i, m := range mods {
		SemanticMemo.modMap[m] = 1 << uint(i)
	}
	// we could have pruned or rearranged them.
	// But then change the list in cmd.go too
	return SemanticMemo.tokTypes, SemanticMemo.tokMods
}
