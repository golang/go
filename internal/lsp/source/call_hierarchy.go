// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package source

import (
	"context"
	"fmt"
	"go/ast"
	"go/token"
	"go/types"
	"path/filepath"

	"golang.org/x/tools/go/ast/astutil"
	"golang.org/x/tools/internal/event"
	"golang.org/x/tools/internal/lsp/debug/tag"
	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/span"
	errors "golang.org/x/xerrors"
)

// PrepareCallHierarchy returns an array of CallHierarchyItem for a file and the position within the file.
func PrepareCallHierarchy(ctx context.Context, snapshot Snapshot, fh FileHandle, pos protocol.Position) ([]protocol.CallHierarchyItem, error) {
	ctx, done := event.Start(ctx, "source.PrepareCallHierarchy")
	defer done()

	identifier, err := Identifier(ctx, snapshot, fh, pos)
	if err != nil {
		if errors.Is(err, ErrNoIdentFound) || errors.Is(err, errNoObjectFound) {
			return nil, nil
		}
		return nil, err
	}
	// The identifier can be nil if it is an import spec.
	if identifier == nil {
		return nil, nil
	}

	if _, ok := identifier.Declaration.obj.Type().Underlying().(*types.Signature); !ok {
		return nil, nil
	}

	if len(identifier.Declaration.MappedRange) == 0 {
		return nil, nil
	}
	declMappedRange := identifier.Declaration.MappedRange[0]
	rng, err := declMappedRange.Range()
	if err != nil {
		return nil, err
	}

	callHierarchyItem := protocol.CallHierarchyItem{
		Name:           identifier.Name,
		Kind:           protocol.Function,
		Tags:           []protocol.SymbolTag{},
		Detail:         fmt.Sprintf("%s • %s", identifier.Declaration.obj.Pkg().Path(), filepath.Base(declMappedRange.URI().Filename())),
		URI:            protocol.DocumentURI(declMappedRange.URI()),
		Range:          rng,
		SelectionRange: rng,
	}
	return []protocol.CallHierarchyItem{callHierarchyItem}, nil
}

// IncomingCalls returns an array of CallHierarchyIncomingCall for a file and the position within the file.
func IncomingCalls(ctx context.Context, snapshot Snapshot, fh FileHandle, pos protocol.Position) ([]protocol.CallHierarchyIncomingCall, error) {
	ctx, done := event.Start(ctx, "source.IncomingCalls")
	defer done()

	refs, err := References(ctx, snapshot, fh, pos, false)
	if err != nil {
		if errors.Is(err, ErrNoIdentFound) || errors.Is(err, errNoObjectFound) {
			return nil, nil
		}
		return nil, err
	}

	return toProtocolIncomingCalls(ctx, snapshot, refs)
}

// toProtocolIncomingCalls returns an array of protocol.CallHierarchyIncomingCall for ReferenceInfo's.
// References inside same enclosure are assigned to the same enclosing function.
func toProtocolIncomingCalls(ctx context.Context, snapshot Snapshot, refs []*ReferenceInfo) ([]protocol.CallHierarchyIncomingCall, error) {
	// an enclosing node could have multiple calls to a reference, we only show the enclosure
	// once in the result but highlight all calls using FromRanges (ranges at which the calls occur)
	var incomingCalls = map[protocol.Location]*protocol.CallHierarchyIncomingCall{}
	for _, ref := range refs {
		refRange, err := ref.Range()
		if err != nil {
			return nil, err
		}

		callItem, err := enclosingNodeCallItem(snapshot, ref.pkg, ref.URI(), ref.ident.NamePos)
		if err != nil {
			event.Error(ctx, "error getting enclosing node", err, tag.Method.Of(ref.Name))
			continue
		}
		loc := protocol.Location{
			URI:   callItem.URI,
			Range: callItem.Range,
		}

		if incomingCall, ok := incomingCalls[loc]; ok {
			incomingCall.FromRanges = append(incomingCall.FromRanges, refRange)
			continue
		}
		incomingCalls[loc] = &protocol.CallHierarchyIncomingCall{
			From:       callItem,
			FromRanges: []protocol.Range{refRange},
		}
	}

	incomingCallItems := make([]protocol.CallHierarchyIncomingCall, 0, len(incomingCalls))
	for _, callItem := range incomingCalls {
		incomingCallItems = append(incomingCallItems, *callItem)
	}
	return incomingCallItems, nil
}

// enclosingNodeCallItem creates a CallHierarchyItem representing the function call at pos
func enclosingNodeCallItem(snapshot Snapshot, pkg Package, uri span.URI, pos token.Pos) (protocol.CallHierarchyItem, error) {
	pgf, err := pkg.File(uri)
	if err != nil {
		return protocol.CallHierarchyItem{}, err
	}

	var funcDecl *ast.FuncDecl
	var funcLit *ast.FuncLit // innermost function literal
	var litCount int
	// Find the enclosing function, if any, and the number of func literals in between.
	path, _ := astutil.PathEnclosingInterval(pgf.File, pos, pos)
outer:
	for _, node := range path {
		switch n := node.(type) {
		case *ast.FuncDecl:
			funcDecl = n
			break outer
		case *ast.FuncLit:
			litCount++
			if litCount > 1 {
				continue
			}
			funcLit = n
		}
	}

	nameIdent := path[len(path)-1].(*ast.File).Name
	kind := protocol.Package
	if funcDecl != nil {
		nameIdent = funcDecl.Name
		kind = protocol.Function
	}

	nameStart, nameEnd := nameIdent.NamePos, nameIdent.NamePos+token.Pos(len(nameIdent.Name))
	if funcLit != nil {
		nameStart, nameEnd = funcLit.Type.Func, funcLit.Type.Params.Pos()
		kind = protocol.Function
	}
	rng, err := NewMappedRange(snapshot.FileSet(), pgf.Mapper, nameStart, nameEnd).Range()
	if err != nil {
		return protocol.CallHierarchyItem{}, err
	}

	name := nameIdent.Name
	for i := 0; i < litCount; i++ {
		name += ".func()"
	}

	return protocol.CallHierarchyItem{
		Name:           name,
		Kind:           kind,
		Tags:           []protocol.SymbolTag{},
		Detail:         fmt.Sprintf("%s • %s", pkg.PkgPath(), filepath.Base(uri.Filename())),
		URI:            protocol.DocumentURI(uri),
		Range:          rng,
		SelectionRange: rng,
	}, nil
}

// OutgoingCalls returns an array of CallHierarchyOutgoingCall for a file and the position within the file.
func OutgoingCalls(ctx context.Context, snapshot Snapshot, fh FileHandle, pos protocol.Position) ([]protocol.CallHierarchyOutgoingCall, error) {
	ctx, done := event.Start(ctx, "source.OutgoingCalls")
	defer done()

	identifier, err := Identifier(ctx, snapshot, fh, pos)
	if err != nil {
		if errors.Is(err, ErrNoIdentFound) || errors.Is(err, errNoObjectFound) {
			return nil, nil
		}
		return nil, err
	}

	if _, ok := identifier.Declaration.obj.Type().Underlying().(*types.Signature); !ok {
		return nil, nil
	}
	if identifier.Declaration.node == nil {
		return nil, nil
	}
	if len(identifier.Declaration.MappedRange) == 0 {
		return nil, nil
	}
	declMappedRange := identifier.Declaration.MappedRange[0]
	callExprs, err := collectCallExpressions(snapshot.FileSet(), declMappedRange.m, identifier.Declaration.node)
	if err != nil {
		return nil, err
	}

	return toProtocolOutgoingCalls(ctx, snapshot, fh, callExprs)
}

// collectCallExpressions collects call expression ranges inside a function.
func collectCallExpressions(fset *token.FileSet, mapper *protocol.ColumnMapper, node ast.Node) ([]protocol.Range, error) {
	type callPos struct {
		start, end token.Pos
	}
	callPositions := []callPos{}

	ast.Inspect(node, func(n ast.Node) bool {
		if call, ok := n.(*ast.CallExpr); ok {
			var start, end token.Pos
			switch n := call.Fun.(type) {
			case *ast.SelectorExpr:
				start, end = n.Sel.NamePos, call.Lparen
			case *ast.Ident:
				start, end = n.NamePos, call.Lparen
			case *ast.FuncLit:
				// while we don't add the function literal as an 'outgoing' call
				// we still want to traverse into it
				return true
			default:
				// ignore any other kind of call expressions
				// for ex: direct function literal calls since that's not an 'outgoing' call
				return false
			}
			callPositions = append(callPositions, callPos{start: start, end: end})
		}
		return true
	})

	callRanges := []protocol.Range{}
	for _, call := range callPositions {
		callRange, err := NewMappedRange(fset, mapper, call.start, call.end).Range()
		if err != nil {
			return nil, err
		}
		callRanges = append(callRanges, callRange)
	}
	return callRanges, nil
}

// toProtocolOutgoingCalls returns an array of protocol.CallHierarchyOutgoingCall for ast call expressions.
// Calls to the same function are assigned to the same declaration.
func toProtocolOutgoingCalls(ctx context.Context, snapshot Snapshot, fh FileHandle, callRanges []protocol.Range) ([]protocol.CallHierarchyOutgoingCall, error) {
	// Multiple calls could be made to the same function, defined by "same declaration
	// AST node & same idenfitier name" to provide a unique identifier key even when
	// the func is declared in a struct or interface.
	type key struct {
		decl ast.Node
		name string
	}
	outgoingCalls := map[key]*protocol.CallHierarchyOutgoingCall{}
	for _, callRange := range callRanges {
		identifier, err := Identifier(ctx, snapshot, fh, callRange.Start)
		if err != nil {
			if errors.Is(err, ErrNoIdentFound) || errors.Is(err, errNoObjectFound) {
				continue
			}
			return nil, err
		}

		// ignore calls to builtin functions
		if identifier.Declaration.obj.Pkg() == nil {
			continue
		}

		if outgoingCall, ok := outgoingCalls[key{identifier.Declaration.node, identifier.Name}]; ok {
			outgoingCall.FromRanges = append(outgoingCall.FromRanges, callRange)
			continue
		}

		if len(identifier.Declaration.MappedRange) == 0 {
			continue
		}
		declMappedRange := identifier.Declaration.MappedRange[0]
		rng, err := declMappedRange.Range()
		if err != nil {
			return nil, err
		}

		outgoingCalls[key{identifier.Declaration.node, identifier.Name}] = &protocol.CallHierarchyOutgoingCall{
			To: protocol.CallHierarchyItem{
				Name:           identifier.Name,
				Kind:           protocol.Function,
				Tags:           []protocol.SymbolTag{},
				Detail:         fmt.Sprintf("%s • %s", identifier.Declaration.obj.Pkg().Path(), filepath.Base(declMappedRange.URI().Filename())),
				URI:            protocol.DocumentURI(declMappedRange.URI()),
				Range:          rng,
				SelectionRange: rng,
			},
			FromRanges: []protocol.Range{callRange},
		}
	}

	outgoingCallItems := make([]protocol.CallHierarchyOutgoingCall, 0, len(outgoingCalls))
	for _, callItem := range outgoingCalls {
		outgoingCallItems = append(outgoingCallItems, *callItem)
	}
	return outgoingCallItems, nil
}
