// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package source

import (
	"context"
	"fmt"
	"go/ast"
	"go/token"
	"path/filepath"

	"golang.org/x/tools/go/ast/astutil"
	"golang.org/x/tools/internal/event"
	"golang.org/x/tools/internal/lsp/debug/tag"
	"golang.org/x/tools/internal/lsp/protocol"
	errors "golang.org/x/xerrors"
)

// PrepareCallHierarchy returns an array of CallHierarchyItem for a file and the position within the file
func PrepareCallHierarchy(ctx context.Context, snapshot Snapshot, fh FileHandle, pos protocol.Position) ([]protocol.CallHierarchyItem, error) {
	ctx, done := event.Start(ctx, "source.prepareCallHierarchy")
	defer done()

	identifier, err := Identifier(ctx, snapshot, fh, pos)
	if err != nil {
		if errors.Is(err, ErrNoIdentFound) {
			event.Log(ctx, err.Error(), tag.Position.Of(pos))
		} else {
			event.Error(ctx, "error getting identifier", err, tag.Position.Of(pos))
		}
		return nil, nil
	}

	// if identifier's declaration is not of type function declaration
	_, ok := identifier.Declaration.node.(*ast.FuncDecl)
	if !ok {
		event.Log(ctx, "invalid identifier declaration, expected funtion declaration", tag.Position.Of(pos))
		return nil, nil
	}
	rng, err := identifier.Range()
	if err != nil {
		return nil, err
	}
	callHierarchyItem := protocol.CallHierarchyItem{
		Name:           identifier.Name,
		Kind:           protocol.Function,
		Tags:           []protocol.SymbolTag{},
		Detail:         fmt.Sprintf("%s %s", identifier.pkg.PkgPath(), filepath.Base(fh.URI().Filename())),
		URI:            protocol.DocumentURI(fh.URI()),
		Range:          rng,
		SelectionRange: rng,
	}
	return []protocol.CallHierarchyItem{callHierarchyItem}, nil
}

// IncomingCalls returns an array of CallHierarchyIncomingCall for a file and the position within the file
func IncomingCalls(ctx context.Context, snapshot Snapshot, fh FileHandle, pos protocol.Position) ([]protocol.CallHierarchyIncomingCall, error) {
	ctx, done := event.Start(ctx, "source.incomingCalls")
	defer done()

	qualifiedObjs, err := qualifiedObjsAtProtocolPos(ctx, snapshot, fh, pos)
	if err != nil {
		if errors.Is(err, errBuiltin) || errors.Is(err, ErrNoIdentFound) {
			event.Log(ctx, err.Error(), tag.Position.Of(pos))
		} else {
			event.Error(ctx, "error getting identifier", err, tag.Position.Of(pos))
		}
		return nil, nil
	}

	refs, err := references(ctx, snapshot, qualifiedObjs, false)
	if err != nil {
		return nil, err
	}

	return toProtocolIncomingCalls(ctx, snapshot, refs)
}

// OutgoingCalls returns an array of CallHierarchyOutgoingCall for a file and the position within the file
func OutgoingCalls(ctx context.Context, snapshot Snapshot, fh FileHandle, pos protocol.Position) ([]protocol.CallHierarchyOutgoingCall, error) {
	ctx, done := event.Start(ctx, "source.outgoingCalls")
	defer done()

	// TODO: Remove this once the context is used.
	_ = ctx // avoid staticcheck SA4006 warning

	return []protocol.CallHierarchyOutgoingCall{}, nil
}

// toProtocolIncomingCalls returns an array of protocol.CallHierarchyIncomingCall for ReferenceInfo's.
// References inside same enclosure are assigned to the same enclosing function.
func toProtocolIncomingCalls(ctx context.Context, snapshot Snapshot, refs []*ReferenceInfo) ([]protocol.CallHierarchyIncomingCall, error) {
	// an enclosing node could have multiple calls to a reference, we only show the enclosure
	// once in the result but highlight all calls using FromRanges (ranges at which the calls occur)
	var incomingCalls = map[protocol.Range]*protocol.CallHierarchyIncomingCall{}
	for _, ref := range refs {
		refRange, err := ref.Range()
		if err != nil {
			return nil, err
		}

		enclosingName, enclosingRange, err := enclosingNodeInfo(snapshot, ref)
		if err != nil {
			event.Error(ctx, "error getting enclosing node", err, tag.Method.Of(ref.Name))
			continue
		}

		if incomingCall, ok := incomingCalls[enclosingRange]; ok {
			incomingCall.FromRanges = append(incomingCall.FromRanges, refRange)
			continue
		}

		incomingCalls[enclosingRange] = &protocol.CallHierarchyIncomingCall{
			From: protocol.CallHierarchyItem{
				Name:           enclosingName,
				Kind:           protocol.Function,
				Tags:           []protocol.SymbolTag{},
				Detail:         fmt.Sprintf("%s • %s", ref.pkg.PkgPath(), filepath.Base(ref.URI().Filename())),
				URI:            protocol.DocumentURI(ref.URI()),
				Range:          enclosingRange,
				SelectionRange: enclosingRange,
			},
			FromRanges: []protocol.Range{refRange},
		}
	}

	incomingCallItems := make([]protocol.CallHierarchyIncomingCall, 0, len(incomingCalls))
	for _, callItem := range incomingCalls {
		incomingCallItems = append(incomingCallItems, *callItem)
	}
	return incomingCallItems, nil
}

// enclosingNodeInfo returns name and position for package/function declaration/function literal
// containing given call reference
func enclosingNodeInfo(snapshot Snapshot, ref *ReferenceInfo) (string, protocol.Range, error) {
	pgf, err := ref.pkg.File(ref.URI())
	if err != nil {
		return "", protocol.Range{}, err
	}

	var funcDecl *ast.FuncDecl
	var funcLit *ast.FuncLit // innermost function literal
	var litCount int
	// Find the enclosing function, if any, and the number of func literals in between.
	path, _ := astutil.PathEnclosingInterval(pgf.File, ref.ident.NamePos, ref.ident.NamePos)
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
	if funcDecl != nil {
		nameIdent = funcDecl.Name
	}

	nameStart, nameEnd := nameIdent.NamePos, nameIdent.NamePos+token.Pos(len(nameIdent.Name))
	if funcLit != nil {
		nameStart, nameEnd = funcLit.Type.Func, funcLit.Type.Params.Pos()
	}
	rng, err := posToProtocolRange(snapshot, ref.pkg, nameStart, nameEnd)
	if err != nil {
		return "", protocol.Range{}, err
	}

	name := nameIdent.Name
	for i := 0; i < litCount; i++ {
		name += ".func()"
	}

	return name, rng, nil
}

// posToProtocolRange returns protocol.Range for start and end token.Pos
func posToProtocolRange(snapshot Snapshot, pkg Package, start, end token.Pos) (protocol.Range, error) {
	mappedRange, err := posToMappedRange(snapshot, pkg, start, end)
	if err != nil {
		return protocol.Range{}, err
	}
	protocolRange, err := mappedRange.Range()
	if err != nil {
		return protocol.Range{}, err
	}
	return protocolRange, nil
}
