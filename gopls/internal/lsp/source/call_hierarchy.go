// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package source

import (
	"context"
	"errors"
	"fmt"
	"go/ast"
	"go/token"
	"go/types"
	"path/filepath"

	"golang.org/x/tools/go/ast/astutil"
	"golang.org/x/tools/gopls/internal/bug"
	"golang.org/x/tools/gopls/internal/lsp/protocol"
	"golang.org/x/tools/gopls/internal/lsp/safetoken"
	"golang.org/x/tools/gopls/internal/span"
	"golang.org/x/tools/internal/event"
	"golang.org/x/tools/internal/event/tag"
)

// PrepareCallHierarchy returns an array of CallHierarchyItem for a file and the position within the file.
func PrepareCallHierarchy(ctx context.Context, snapshot Snapshot, fh FileHandle, pp protocol.Position) ([]protocol.CallHierarchyItem, error) {
	ctx, done := event.Start(ctx, "source.PrepareCallHierarchy")
	defer done()

	pkg, pgf, err := NarrowestPackageForFile(ctx, snapshot, fh.URI())
	if err != nil {
		return nil, err
	}
	pos, err := pgf.PositionPos(pp)
	if err != nil {
		return nil, err
	}

	_, obj, _ := referencedObject(pkg, pgf, pos)
	if obj == nil {
		return nil, nil
	}

	if _, ok := obj.Type().Underlying().(*types.Signature); !ok {
		return nil, nil
	}

	declLoc, err := mapPosition(ctx, pkg.FileSet(), snapshot, obj.Pos(), adjustedObjEnd(obj))
	if err != nil {
		return nil, err
	}
	rng := declLoc.Range

	callHierarchyItem := protocol.CallHierarchyItem{
		Name:           obj.Name(),
		Kind:           protocol.Function,
		Tags:           []protocol.SymbolTag{},
		Detail:         fmt.Sprintf("%s • %s", obj.Pkg().Path(), filepath.Base(declLoc.URI.SpanURI().Filename())),
		URI:            declLoc.URI,
		Range:          rng,
		SelectionRange: rng,
	}
	return []protocol.CallHierarchyItem{callHierarchyItem}, nil
}

// IncomingCalls returns an array of CallHierarchyIncomingCall for a file and the position within the file.
func IncomingCalls(ctx context.Context, snapshot Snapshot, fh FileHandle, pos protocol.Position) ([]protocol.CallHierarchyIncomingCall, error) {
	ctx, done := event.Start(ctx, "source.IncomingCalls")
	defer done()

	refs, err := references(ctx, snapshot, fh, pos, false)
	if err != nil {
		if errors.Is(err, ErrNoIdentFound) || errors.Is(err, errNoObjectFound) {
			return nil, nil
		}
		return nil, err
	}

	// Group references by their enclosing function declaration.
	incomingCalls := make(map[protocol.Location]*protocol.CallHierarchyIncomingCall)
	for _, ref := range refs {
		callItem, err := enclosingNodeCallItem(ctx, snapshot, ref.pkgPath, ref.location)
		if err != nil {
			event.Error(ctx, "error getting enclosing node", err, tag.Method.Of(string(ref.pkgPath)))
			continue
		}
		loc := protocol.Location{
			URI:   callItem.URI,
			Range: callItem.Range,
		}
		call, ok := incomingCalls[loc]
		if !ok {
			call = &protocol.CallHierarchyIncomingCall{From: callItem}
			incomingCalls[loc] = call
		}
		call.FromRanges = append(call.FromRanges, ref.location.Range)
	}

	// Flatten the map of pointers into a slice of values.
	incomingCallItems := make([]protocol.CallHierarchyIncomingCall, 0, len(incomingCalls))
	for _, callItem := range incomingCalls {
		incomingCallItems = append(incomingCallItems, *callItem)
	}
	return incomingCallItems, nil
}

// enclosingNodeCallItem creates a CallHierarchyItem representing the function call at loc.
func enclosingNodeCallItem(ctx context.Context, snapshot Snapshot, pkgPath PackagePath, loc protocol.Location) (protocol.CallHierarchyItem, error) {
	// Parse the file containing the reference.
	fh, err := snapshot.ReadFile(ctx, loc.URI.SpanURI())
	if err != nil {
		return protocol.CallHierarchyItem{}, err
	}
	// TODO(adonovan): opt: before parsing, trim the bodies of functions
	// that don't contain the reference, using either a scanner-based
	// implementation such as https://go.dev/play/p/KUrObH1YkX8
	// (~31% speedup), or a byte-oriented implementation (2x speedup).
	pgf, err := snapshot.ParseGo(ctx, fh, ParseFull)
	if err != nil {
		return protocol.CallHierarchyItem{}, err
	}
	start, end, err := pgf.RangePos(loc.Range)
	if err != nil {
		return protocol.CallHierarchyItem{}, err
	}

	// Find the enclosing function, if any, and the number of func literals in between.
	var funcDecl *ast.FuncDecl
	var funcLit *ast.FuncLit // innermost function literal
	var litCount int
	path, _ := astutil.PathEnclosingInterval(pgf.File, start, end)
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

	nameStart, nameEnd := nameIdent.Pos(), nameIdent.End()
	if funcLit != nil {
		nameStart, nameEnd = funcLit.Type.Func, funcLit.Type.Params.Pos()
		kind = protocol.Function
	}
	rng, err := pgf.PosRange(nameStart, nameEnd)
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
		Detail:         fmt.Sprintf("%s • %s", pkgPath, filepath.Base(fh.URI().Filename())),
		URI:            loc.URI,
		Range:          rng,
		SelectionRange: rng,
	}, nil
}

// OutgoingCalls returns an array of CallHierarchyOutgoingCall for a file and the position within the file.
func OutgoingCalls(ctx context.Context, snapshot Snapshot, fh FileHandle, pp protocol.Position) ([]protocol.CallHierarchyOutgoingCall, error) {
	ctx, done := event.Start(ctx, "source.OutgoingCalls")
	defer done()

	pkg, pgf, err := NarrowestPackageForFile(ctx, snapshot, fh.URI())
	if err != nil {
		return nil, err
	}
	pos, err := pgf.PositionPos(pp)
	if err != nil {
		return nil, err
	}

	_, obj, _ := referencedObject(pkg, pgf, pos)
	if obj == nil {
		return nil, nil
	}

	if _, ok := obj.Type().Underlying().(*types.Signature); !ok {
		return nil, nil
	}

	// Skip builtins.
	if obj.Pkg() == nil {
		return nil, nil
	}

	if !obj.Pos().IsValid() {
		return nil, bug.Errorf("internal error: object %s.%s missing position", obj.Pkg().Path(), obj.Name())
	}

	declFile := pkg.FileSet().File(obj.Pos())
	if declFile == nil {
		return nil, bug.Errorf("file not found for %d", obj.Pos())
	}

	uri := span.URIFromPath(declFile.Name())
	offset, err := safetoken.Offset(declFile, obj.Pos())
	if err != nil {
		return nil, err
	}

	// Use TypecheckFull as we want to inspect the body of the function declaration.
	declPkg, declPGF, err := NarrowestPackageForFile(ctx, snapshot, uri)
	if err != nil {
		return nil, err
	}

	declPos, err := safetoken.Pos(declPGF.Tok, offset)
	if err != nil {
		return nil, err
	}

	declNode, _, _ := findDeclInfo([]*ast.File{declPGF.File}, declPos)
	if declNode == nil {
		// TODO(rfindley): why don't we return an error here, or even bug.Errorf?
		return nil, nil
		// return nil, bug.Errorf("failed to find declaration for object %s.%s", obj.Pkg().Path(), obj.Name())
	}

	type callRange struct {
		start, end token.Pos
	}
	callRanges := []callRange{}
	ast.Inspect(declNode, func(n ast.Node) bool {
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
			callRanges = append(callRanges, callRange{start: start, end: end})
		}
		return true
	})

	outgoingCalls := map[token.Pos]*protocol.CallHierarchyOutgoingCall{}
	for _, callRange := range callRanges {
		_, obj, _ := referencedObject(declPkg, declPGF, callRange.start)
		if obj == nil {
			continue
		}

		// ignore calls to builtin functions
		if obj.Pkg() == nil {
			continue
		}

		outgoingCall, ok := outgoingCalls[obj.Pos()]
		if !ok {
			loc, err := mapPosition(ctx, declPkg.FileSet(), snapshot, obj.Pos(), obj.Pos()+token.Pos(len(obj.Name())))
			if err != nil {
				return nil, err
			}
			outgoingCall = &protocol.CallHierarchyOutgoingCall{
				To: protocol.CallHierarchyItem{
					Name:           obj.Name(),
					Kind:           protocol.Function,
					Tags:           []protocol.SymbolTag{},
					Detail:         fmt.Sprintf("%s • %s", obj.Pkg().Path(), filepath.Base(loc.URI.SpanURI().Filename())),
					URI:            loc.URI,
					Range:          loc.Range,
					SelectionRange: loc.Range,
				},
			}
			outgoingCalls[obj.Pos()] = outgoingCall
		}

		rng, err := declPGF.PosRange(callRange.start, callRange.end)
		if err != nil {
			return nil, err
		}
		outgoingCall.FromRanges = append(outgoingCall.FromRanges, rng)
	}

	outgoingCallItems := make([]protocol.CallHierarchyOutgoingCall, 0, len(outgoingCalls))
	for _, callItem := range outgoingCalls {
		outgoingCallItems = append(outgoingCallItems, *callItem)
	}
	return outgoingCallItems, nil
}
