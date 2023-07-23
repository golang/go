// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lsp

import (
	"bytes"
	"context"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	"golang.org/x/tools/gopls/internal/bug"
	"golang.org/x/tools/gopls/internal/lsp/cache"
	"golang.org/x/tools/gopls/internal/lsp/command"
	"golang.org/x/tools/gopls/internal/lsp/debug"
	"golang.org/x/tools/gopls/internal/lsp/protocol"
	"golang.org/x/tools/gopls/internal/lsp/source"
	"golang.org/x/tools/gopls/internal/lsp/tests"
	"golang.org/x/tools/gopls/internal/lsp/tests/compare"
	"golang.org/x/tools/gopls/internal/span"
	"golang.org/x/tools/internal/testenv"
)

func TestMain(m *testing.M) {
	bug.PanicOnBugs = true
	testenv.ExitIfSmallMachine()

	os.Exit(m.Run())
}

// TestLSP runs the marker tests in files beneath testdata/ using
// implementations of each of the marker operations (e.g. @codelens) that
// make LSP RPCs (e.g. textDocument/codeLens) to a gopls server.
func TestLSP(t *testing.T) {
	tests.RunTests(t, "testdata", true, testLSP)
}

func testLSP(t *testing.T, datum *tests.Data) {
	ctx := tests.Context(t)

	// Setting a debug instance suppresses logging to stderr, but ensures that we
	// still e.g. convert events into runtime/trace/instrumentation.
	//
	// Previously, we called event.SetExporter(nil), which turns off all
	// instrumentation.
	ctx = debug.WithInstance(ctx, "", "off")

	session := cache.NewSession(ctx, cache.New(nil), nil)
	options := source.DefaultOptions().Clone()
	tests.DefaultOptions(options)
	session.SetOptions(options)
	options.SetEnvSlice(datum.Config.Env)
	view, snapshot, release, err := session.NewView(ctx, datum.Config.Dir, span.URIFromPath(datum.Config.Dir), options)
	if err != nil {
		t.Fatal(err)
	}

	defer session.RemoveView(view)

	// Enable type error analyses for tests.
	// TODO(golang/go#38212): Delete this once they are enabled by default.
	tests.EnableAllAnalyzers(options)
	session.SetViewOptions(ctx, view, options)

	// Enable all inlay hints for tests.
	tests.EnableAllInlayHints(options)

	// Only run the -modfile specific tests in module mode with Go 1.14 or above.
	datum.ModfileFlagAvailable = len(snapshot.ModFiles()) > 0 && testenv.Go1Point() >= 14
	release()

	// Open all files for performance reasons, because gopls only
	// keeps active packages (those with open files) in memory.
	//
	// In practice clients will only send document-oriented requests for open
	// files.
	var modifications []source.FileModification
	for _, module := range datum.Exported.Modules {
		for name := range module.Files {
			filename := datum.Exported.File(module.Name, name)
			if filepath.Ext(filename) != ".go" {
				continue
			}
			content, err := datum.Exported.FileContents(filename)
			if err != nil {
				t.Fatal(err)
			}
			modifications = append(modifications, source.FileModification{
				URI:        span.URIFromPath(filename),
				Action:     source.Open,
				Version:    -1,
				Text:       content,
				LanguageID: "go",
			})
		}
	}
	for filename, content := range datum.Config.Overlay {
		if filepath.Ext(filename) != ".go" {
			continue
		}
		modifications = append(modifications, source.FileModification{
			URI:        span.URIFromPath(filename),
			Action:     source.Open,
			Version:    -1,
			Text:       content,
			LanguageID: "go",
		})
	}
	if err := session.ModifyFiles(ctx, modifications); err != nil {
		t.Fatal(err)
	}
	r := &runner{
		data:     datum,
		ctx:      ctx,
		editRecv: make(chan map[span.URI][]byte, 1),
	}

	r.server = NewServer(session, testClient{runner: r})
	tests.Run(t, r, datum)
}

// runner implements tests.Tests by making LSP RPCs to a gopls server.
type runner struct {
	server      *Server
	data        *tests.Data
	diagnostics map[span.URI][]*source.Diagnostic
	ctx         context.Context
	editRecv    chan map[span.URI][]byte
}

// testClient stubs any client functions that may be called by LSP functions.
type testClient struct {
	protocol.Client
	runner *runner
}

func (c testClient) Close() error {
	return nil
}

// Trivially implement PublishDiagnostics so that we can call
// server.publishReports below to de-dup sent diagnostics.
func (c testClient) PublishDiagnostics(context.Context, *protocol.PublishDiagnosticsParams) error {
	return nil
}

func (c testClient) ShowMessage(context.Context, *protocol.ShowMessageParams) error {
	return nil
}

func (c testClient) ApplyEdit(ctx context.Context, params *protocol.ApplyWorkspaceEditParams) (*protocol.ApplyWorkspaceEditResult, error) {
	res, err := applyTextDocumentEdits(c.runner, params.Edit.DocumentChanges)
	if err != nil {
		return nil, err
	}
	c.runner.editRecv <- res
	return &protocol.ApplyWorkspaceEditResult{Applied: true}, nil
}

func (r *runner) CallHierarchy(t *testing.T, spn span.Span, expectedCalls *tests.CallHierarchyResult) {
	mapper, err := r.data.Mapper(spn.URI())
	if err != nil {
		t.Fatal(err)
	}
	loc, err := mapper.SpanLocation(spn)
	if err != nil {
		t.Fatalf("failed for %v: %v", spn, err)
	}

	params := &protocol.CallHierarchyPrepareParams{
		TextDocumentPositionParams: protocol.LocationTextDocumentPositionParams(loc),
	}

	items, err := r.server.PrepareCallHierarchy(r.ctx, params)
	if err != nil {
		t.Fatal(err)
	}
	if len(items) == 0 {
		t.Fatalf("expected call hierarchy item to be returned for identifier at %v\n", loc.Range)
	}

	callLocation := protocol.Location{
		URI:   items[0].URI,
		Range: items[0].Range,
	}
	if callLocation != loc {
		t.Fatalf("expected server.PrepareCallHierarchy to return identifier at %v but got %v\n", loc, callLocation)
	}

	incomingCalls, err := r.server.IncomingCalls(r.ctx, &protocol.CallHierarchyIncomingCallsParams{Item: items[0]})
	if err != nil {
		t.Error(err)
	}
	var incomingCallItems []protocol.CallHierarchyItem
	for _, item := range incomingCalls {
		incomingCallItems = append(incomingCallItems, item.From)
	}
	msg := tests.DiffCallHierarchyItems(incomingCallItems, expectedCalls.IncomingCalls)
	if msg != "" {
		t.Errorf("incoming calls: %s", msg)
	}

	outgoingCalls, err := r.server.OutgoingCalls(r.ctx, &protocol.CallHierarchyOutgoingCallsParams{Item: items[0]})
	if err != nil {
		t.Error(err)
	}
	var outgoingCallItems []protocol.CallHierarchyItem
	for _, item := range outgoingCalls {
		outgoingCallItems = append(outgoingCallItems, item.To)
	}
	msg = tests.DiffCallHierarchyItems(outgoingCallItems, expectedCalls.OutgoingCalls)
	if msg != "" {
		t.Errorf("outgoing calls: %s", msg)
	}
}

func (r *runner) CodeLens(t *testing.T, uri span.URI, want []protocol.CodeLens) {
	if !strings.HasSuffix(uri.Filename(), "go.mod") {
		return
	}
	got, err := r.server.codeLens(r.ctx, &protocol.CodeLensParams{
		TextDocument: protocol.TextDocumentIdentifier{
			URI: protocol.DocumentURI(uri),
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if diff := tests.DiffCodeLens(uri, want, got); diff != "" {
		t.Errorf("%s: %s", uri, diff)
	}
}

func (r *runner) Diagnostics(t *testing.T, uri span.URI, want []*source.Diagnostic) {
	// Get the diagnostics for this view if we have not done it before.
	v := r.server.session.ViewByName(r.data.Config.Dir)
	r.collectDiagnostics(v)
	tests.CompareDiagnostics(t, uri, want, r.diagnostics[uri])
}

func (r *runner) FoldingRanges(t *testing.T, spn span.Span) {
	uri := spn.URI()
	view, err := r.server.session.ViewOf(uri)
	if err != nil {
		t.Fatal(err)
	}
	original := view.Options()
	modified := original
	defer r.server.session.SetViewOptions(r.ctx, view, original)

	for _, test := range []struct {
		lineFoldingOnly bool
		prefix          string
	}{
		{false, "foldingRange"},
		{true, "foldingRange-lineFolding"},
	} {
		modified.LineFoldingOnly = test.lineFoldingOnly
		view, err = r.server.session.SetViewOptions(r.ctx, view, modified)
		if err != nil {
			t.Error(err)
			continue
		}
		ranges, err := r.server.FoldingRange(r.ctx, &protocol.FoldingRangeParams{
			TextDocument: protocol.TextDocumentIdentifier{
				URI: protocol.URIFromSpanURI(uri),
			},
		})
		if err != nil {
			t.Error(err)
			continue
		}
		r.foldingRanges(t, test.prefix, uri, ranges)
	}
}

func (r *runner) foldingRanges(t *testing.T, prefix string, uri span.URI, ranges []protocol.FoldingRange) {
	m, err := r.data.Mapper(uri)
	if err != nil {
		t.Fatal(err)
	}
	// Fold all ranges.
	nonOverlapping := nonOverlappingRanges(ranges)
	for i, rngs := range nonOverlapping {
		got, err := foldRanges(m, string(m.Content), rngs)
		if err != nil {
			t.Error(err)
			continue
		}
		tag := fmt.Sprintf("%s-%d", prefix, i)
		want := string(r.data.Golden(t, tag, uri.Filename(), func() ([]byte, error) {
			return []byte(got), nil
		}))

		if want != got {
			t.Errorf("%s: foldingRanges failed for %s, expected:\n%v\ngot:\n%v", tag, uri.Filename(), want, got)
		}
	}

	// Filter by kind.
	kinds := []protocol.FoldingRangeKind{protocol.Imports, protocol.Comment}
	for _, kind := range kinds {
		var kindOnly []protocol.FoldingRange
		for _, fRng := range ranges {
			if fRng.Kind == string(kind) {
				kindOnly = append(kindOnly, fRng)
			}
		}

		nonOverlapping := nonOverlappingRanges(kindOnly)
		for i, rngs := range nonOverlapping {
			got, err := foldRanges(m, string(m.Content), rngs)
			if err != nil {
				t.Error(err)
				continue
			}
			tag := fmt.Sprintf("%s-%s-%d", prefix, kind, i)
			want := string(r.data.Golden(t, tag, uri.Filename(), func() ([]byte, error) {
				return []byte(got), nil
			}))

			if want != got {
				t.Errorf("%s: foldingRanges failed for %s, expected:\n%v\ngot:\n%v", tag, uri.Filename(), want, got)
			}
		}

	}
}

func nonOverlappingRanges(ranges []protocol.FoldingRange) (res [][]protocol.FoldingRange) {
	for _, fRng := range ranges {
		setNum := len(res)
		for i := 0; i < len(res); i++ {
			canInsert := true
			for _, rng := range res[i] {
				if conflict(rng, fRng) {
					canInsert = false
					break
				}
			}
			if canInsert {
				setNum = i
				break
			}
		}
		if setNum == len(res) {
			res = append(res, []protocol.FoldingRange{})
		}
		res[setNum] = append(res[setNum], fRng)
	}
	return res
}

func conflict(a, b protocol.FoldingRange) bool {
	// a start position is <= b start positions
	return (a.StartLine < b.StartLine || (a.StartLine == b.StartLine && a.StartCharacter <= b.StartCharacter)) &&
		(a.EndLine > b.StartLine || (a.EndLine == b.StartLine && a.EndCharacter > b.StartCharacter))
}

func foldRanges(m *protocol.Mapper, contents string, ranges []protocol.FoldingRange) (string, error) {
	foldedText := "<>"
	res := contents
	// Apply the edits from the end of the file forward
	// to preserve the offsets
	// TODO(adonovan): factor to use diff.ApplyEdits, which validates the input.
	for i := len(ranges) - 1; i >= 0; i-- {
		r := ranges[i]
		start, end, err := m.RangeOffsets(protocol.Range{
			Start: protocol.Position{Line: r.StartLine, Character: r.StartCharacter},
			End:   protocol.Position{Line: r.EndLine, Character: r.EndCharacter},
		})
		if err != nil {
			return "", err
		}
		res = res[:start] + foldedText + res[end:]
	}
	return res, nil
}

func (r *runner) SemanticTokens(t *testing.T, spn span.Span) {
	uri := spn.URI()
	filename := uri.Filename()
	// this is called solely for coverage in semantic.go
	_, err := r.server.semanticTokensFull(r.ctx, &protocol.SemanticTokensParams{
		TextDocument: protocol.TextDocumentIdentifier{
			URI: protocol.URIFromSpanURI(uri),
		},
	})
	if err != nil {
		t.Errorf("%v for %s", err, filename)
	}
	_, err = r.server.semanticTokensRange(r.ctx, &protocol.SemanticTokensRangeParams{
		TextDocument: protocol.TextDocumentIdentifier{
			URI: protocol.URIFromSpanURI(uri),
		},
		// any legal range. Just to exercise the call.
		Range: protocol.Range{
			Start: protocol.Position{
				Line:      0,
				Character: 0,
			},
			End: protocol.Position{
				Line:      2,
				Character: 0,
			},
		},
	})
	if err != nil {
		t.Errorf("%v for Range %s", err, filename)
	}
}

func (r *runner) SuggestedFix(t *testing.T, spn span.Span, actionKinds []tests.SuggestedFix, expectedActions int) {
	uri := spn.URI()
	view, err := r.server.session.ViewOf(uri)
	if err != nil {
		t.Fatal(err)
	}

	m, err := r.data.Mapper(uri)
	if err != nil {
		t.Fatal(err)
	}
	rng, err := m.SpanRange(spn)
	if err != nil {
		t.Fatal(err)
	}
	// Get the diagnostics for this view if we have not done it before.
	r.collectDiagnostics(view)
	var diagnostics []protocol.Diagnostic
	for _, d := range r.diagnostics[uri] {
		// Compare the start positions rather than the entire range because
		// some diagnostics have a range with the same start and end position (8:1-8:1).
		// The current marker functionality prevents us from having a range of 0 length.
		if protocol.ComparePosition(d.Range.Start, rng.Start) == 0 {
			diagnostics = append(diagnostics, toProtocolDiagnostics([]*source.Diagnostic{d})...)
			break
		}
	}
	var codeActionKinds []protocol.CodeActionKind
	for _, k := range actionKinds {
		codeActionKinds = append(codeActionKinds, protocol.CodeActionKind(k.ActionKind))
	}
	allActions, err := r.server.CodeAction(r.ctx, &protocol.CodeActionParams{
		TextDocument: protocol.TextDocumentIdentifier{
			URI: protocol.URIFromSpanURI(uri),
		},
		Range: rng,
		Context: protocol.CodeActionContext{
			Only:        codeActionKinds,
			Diagnostics: diagnostics,
		},
	})
	if err != nil {
		t.Fatalf("CodeAction %s failed: %v", spn, err)
	}
	var actions []protocol.CodeAction
	for _, action := range allActions {
		for _, fix := range actionKinds {
			if strings.Contains(action.Title, fix.Title) {
				actions = append(actions, action)
				break
			}
		}

	}
	if len(actions) != expectedActions {
		var summaries []string
		for _, a := range actions {
			summaries = append(summaries, fmt.Sprintf("%q (%s)", a.Title, a.Kind))
		}
		t.Fatalf("CodeAction(...): got %d code actions (%v), want %d", len(actions), summaries, expectedActions)
	}
	action := actions[0]
	var match bool
	for _, k := range codeActionKinds {
		if action.Kind == k {
			match = true
			break
		}
	}
	if !match {
		t.Fatalf("unexpected kind for code action %s, got %v, want one of %v", action.Title, action.Kind, codeActionKinds)
	}
	var res map[span.URI][]byte
	if cmd := action.Command; cmd != nil {
		_, err := r.server.ExecuteCommand(r.ctx, &protocol.ExecuteCommandParams{
			Command:   action.Command.Command,
			Arguments: action.Command.Arguments,
		})
		if err != nil {
			t.Fatalf("error converting command %q to edits: %v", action.Command.Command, err)
		}
		res = <-r.editRecv
	} else {
		res, err = applyTextDocumentEdits(r, action.Edit.DocumentChanges)
		if err != nil {
			t.Fatal(err)
		}
	}
	for u, got := range res {
		want := r.data.Golden(t, "suggestedfix_"+tests.SpanName(spn), u.Filename(), func() ([]byte, error) {
			return got, nil
		})
		if diff := compare.Bytes(want, got); diff != "" {
			t.Errorf("suggested fixes failed for %s:\n%s", u.Filename(), diff)
		}
	}
}

func (r *runner) MethodExtraction(t *testing.T, start span.Span, end span.Span) {
	uri := start.URI()
	m, err := r.data.Mapper(uri)
	if err != nil {
		t.Fatal(err)
	}
	spn := span.New(start.URI(), start.Start(), end.End())
	rng, err := m.SpanRange(spn)
	if err != nil {
		t.Fatal(err)
	}
	actionsRaw, err := r.server.CodeAction(r.ctx, &protocol.CodeActionParams{
		TextDocument: protocol.TextDocumentIdentifier{
			URI: protocol.URIFromSpanURI(uri),
		},
		Range: rng,
		Context: protocol.CodeActionContext{
			Only: []protocol.CodeActionKind{"refactor.extract"},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	var actions []protocol.CodeAction
	for _, action := range actionsRaw {
		if action.Command.Title == "Extract method" {
			actions = append(actions, action)
		}
	}
	// Hack: We assume that we only get one matching code action per range.
	// TODO(rstambler): Support multiple code actions per test.
	if len(actions) == 0 || len(actions) > 1 {
		t.Fatalf("unexpected number of code actions, want 1, got %v", len(actions))
	}
	_, err = r.server.ExecuteCommand(r.ctx, &protocol.ExecuteCommandParams{
		Command:   actions[0].Command.Command,
		Arguments: actions[0].Command.Arguments,
	})
	if err != nil {
		t.Fatal(err)
	}
	res := <-r.editRecv
	for u, got := range res {
		want := r.data.Golden(t, "methodextraction_"+tests.SpanName(spn), u.Filename(), func() ([]byte, error) {
			return got, nil
		})
		if diff := compare.Bytes(want, got); diff != "" {
			t.Errorf("method extraction failed for %s:\n%s", u.Filename(), diff)
		}
	}
}

// TODO(rfindley): This handler needs more work. The output is still a bit hard
// to read (range diffs do not format nicely), and it is too entangled with hover.
func (r *runner) Definition(t *testing.T, _ span.Span, d tests.Definition) {
	sm, err := r.data.Mapper(d.Src.URI())
	if err != nil {
		t.Fatal(err)
	}
	loc, err := sm.SpanLocation(d.Src)
	if err != nil {
		t.Fatalf("failed for %v: %v", d.Src, err)
	}
	tdpp := protocol.LocationTextDocumentPositionParams(loc)
	var got []protocol.Location
	var hover *protocol.Hover
	if d.IsType {
		params := &protocol.TypeDefinitionParams{
			TextDocumentPositionParams: tdpp,
		}
		got, err = r.server.TypeDefinition(r.ctx, params)
	} else {
		params := &protocol.DefinitionParams{
			TextDocumentPositionParams: tdpp,
		}
		got, err = r.server.Definition(r.ctx, params)
		if err != nil {
			t.Fatalf("failed for %v: %+v", d.Src, err)
		}
		v := &protocol.HoverParams{
			TextDocumentPositionParams: tdpp,
		}
		hover, err = r.server.Hover(r.ctx, v)
	}
	if err != nil {
		t.Fatalf("failed for %v: %v", d.Src, err)
	}
	dm, err := r.data.Mapper(d.Def.URI())
	if err != nil {
		t.Fatal(err)
	}
	def, err := dm.SpanLocation(d.Def)
	if err != nil {
		t.Fatal(err)
	}
	if !d.OnlyHover {
		want := []protocol.Location{def}
		if diff := cmp.Diff(want, got); diff != "" {
			t.Fatalf("Definition(%s) mismatch (-want +got):\n%s", d.Src, diff)
		}
	}
	didSomething := false
	if hover != nil {
		didSomething = true
		tag := fmt.Sprintf("%s-hoverdef", d.Name)
		want := string(r.data.Golden(t, tag, d.Src.URI().Filename(), func() ([]byte, error) {
			return []byte(hover.Contents.Value), nil
		}))
		got := hover.Contents.Value
		if diff := tests.DiffMarkdown(want, got); diff != "" {
			t.Errorf("%s: markdown mismatch:\n%s", d.Src, diff)
		}
	}
	if !d.OnlyHover {
		didSomething = true
		locURI := got[0].URI.SpanURI()
		lm, err := r.data.Mapper(locURI)
		if err != nil {
			t.Fatal(err)
		}
		if def, err := lm.LocationSpan(got[0]); err != nil {
			t.Fatalf("failed for %v: %v", got[0], err)
		} else if def != d.Def {
			t.Errorf("for %v got %v want %v", d.Src, def, d.Def)
		}
	}
	if !didSomething {
		t.Errorf("no tests ran for %s", d.Src.URI())
	}
}

func (r *runner) Highlight(t *testing.T, src span.Span, spans []span.Span) {
	m, err := r.data.Mapper(src.URI())
	if err != nil {
		t.Fatal(err)
	}
	loc, err := m.SpanLocation(src)
	if err != nil {
		t.Fatal(err)
	}
	params := &protocol.DocumentHighlightParams{
		TextDocumentPositionParams: protocol.LocationTextDocumentPositionParams(loc),
	}
	highlights, err := r.server.DocumentHighlight(r.ctx, params)
	if err != nil {
		t.Fatalf("DocumentHighlight(%v) failed: %v", params, err)
	}
	var got []protocol.Range
	for _, h := range highlights {
		got = append(got, h.Range)
	}

	var want []protocol.Range
	for _, s := range spans {
		rng, err := m.SpanRange(s)
		if err != nil {
			t.Fatalf("Mapper.SpanRange(%v) failed: %v", s, err)
		}
		want = append(want, rng)
	}

	sortRanges := func(s []protocol.Range) {
		sort.Slice(s, func(i, j int) bool {
			return protocol.CompareRange(s[i], s[j]) < 0
		})
	}

	sortRanges(got)
	sortRanges(want)

	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("DocumentHighlight(%v) mismatch (-want +got):\n%s", src, diff)
	}
}

func (r *runner) InlayHints(t *testing.T, spn span.Span) {
	uri := spn.URI()
	filename := uri.Filename()

	hints, err := r.server.InlayHint(r.ctx, &protocol.InlayHintParams{
		TextDocument: protocol.TextDocumentIdentifier{
			URI: protocol.URIFromSpanURI(uri),
		},
		// TODO: add Range
	})
	if err != nil {
		t.Fatal(err)
	}

	// Map inlay hints to text edits.
	edits := make([]protocol.TextEdit, len(hints))
	for i, hint := range hints {
		var paddingLeft, paddingRight string
		if hint.PaddingLeft {
			paddingLeft = " "
		}
		if hint.PaddingRight {
			paddingRight = " "
		}
		edits[i] = protocol.TextEdit{
			Range:   protocol.Range{Start: hint.Position, End: hint.Position},
			NewText: fmt.Sprintf("<%s%s%s>", paddingLeft, hint.Label[0].Value, paddingRight),
		}
	}

	m, err := r.data.Mapper(uri)
	if err != nil {
		t.Fatal(err)
	}
	got, _, err := source.ApplyProtocolEdits(m, edits)
	if err != nil {
		t.Error(err)
	}

	withinlayHints := r.data.Golden(t, "inlayHint", filename, func() ([]byte, error) {
		return got, nil
	})

	if !bytes.Equal(withinlayHints, got) {
		t.Errorf("inlay hints failed for %s, expected:\n%s\ngot:\n%s", filename, withinlayHints, got)
	}
}

func (r *runner) Rename(t *testing.T, spn span.Span, newText string) {
	tag := fmt.Sprintf("%s-rename", newText)

	uri := spn.URI()
	filename := uri.Filename()
	sm, err := r.data.Mapper(uri)
	if err != nil {
		t.Fatal(err)
	}
	loc, err := sm.SpanLocation(spn)
	if err != nil {
		t.Fatalf("failed for %v: %v", spn, err)
	}

	wedit, err := r.server.Rename(r.ctx, &protocol.RenameParams{
		TextDocument: protocol.TextDocumentIdentifier{URI: loc.URI},
		Position:     loc.Range.Start,
		NewName:      newText,
	})
	if err != nil {
		renamed := string(r.data.Golden(t, tag, filename, func() ([]byte, error) {
			return []byte(err.Error()), nil
		}))
		if err.Error() != renamed {
			t.Errorf("%s: rename failed for %s, expected:\n%v\ngot:\n%v\n", spn, newText, renamed, err)
		}
		return
	}
	res, err := applyTextDocumentEdits(r, wedit.DocumentChanges)
	if err != nil {
		t.Fatal(err)
	}
	var orderedURIs []string
	for uri := range res {
		orderedURIs = append(orderedURIs, string(uri))
	}
	sort.Strings(orderedURIs)

	// Print the name and content of each modified file,
	// concatenated, and compare against the golden.
	var buf bytes.Buffer
	for i := 0; i < len(res); i++ {
		if i != 0 {
			buf.WriteByte('\n')
		}
		uri := span.URIFromURI(orderedURIs[i])
		if len(res) > 1 {
			buf.WriteString(filepath.Base(uri.Filename()))
			buf.WriteString(":\n")
		}
		buf.Write(res[uri])
	}
	got := buf.Bytes()
	want := r.data.Golden(t, tag, filename, func() ([]byte, error) {
		return got, nil
	})
	if diff := compare.Bytes(want, got); diff != "" {
		t.Errorf("rename failed for %s:\n%s", newText, diff)
	}
}

func (r *runner) PrepareRename(t *testing.T, src span.Span, want *source.PrepareItem) {
	m, err := r.data.Mapper(src.URI())
	if err != nil {
		t.Fatal(err)
	}
	loc, err := m.SpanLocation(src)
	if err != nil {
		t.Fatalf("failed for %v: %v", src, err)
	}
	params := &protocol.PrepareRenameParams{
		TextDocumentPositionParams: protocol.LocationTextDocumentPositionParams(loc),
	}
	got, err := r.server.PrepareRename(context.Background(), params)
	if err != nil {
		t.Errorf("prepare rename failed for %v: got error: %v", src, err)
		return
	}

	// TODO(rfindley): can we consolidate on a single representation for
	// PrepareRename results, and use cmp.Diff here?

	// PrepareRename may fail with no error if there was no object found at the
	// position.
	if got == nil {
		if want.Text != "" { // expected an ident.
			t.Errorf("prepare rename failed for %v: got nil", src)
		}
		return
	}
	if got.Range.Start == got.Range.End {
		// Special case for 0-length ranges. Marks can't specify a 0-length range,
		// so just compare the start.
		if got.Range.Start != want.Range.Start {
			t.Errorf("prepare rename failed: incorrect point, got %v want %v", got.Range.Start, want.Range.Start)
		}
	} else {
		if got.Range != want.Range {
			t.Errorf("prepare rename failed: incorrect range got %v want %v", got.Range, want.Range)
		}
	}
	if got.Placeholder != want.Text {
		t.Errorf("prepare rename failed: incorrect text got %v want %v", got.Placeholder, want.Text)
	}
}

func applyTextDocumentEdits(r *runner, edits []protocol.DocumentChanges) (map[span.URI][]byte, error) {
	res := make(map[span.URI][]byte)
	for _, docEdits := range edits {
		if docEdits.TextDocumentEdit != nil {
			uri := docEdits.TextDocumentEdit.TextDocument.URI.SpanURI()
			var m *protocol.Mapper
			// If we have already edited this file, we use the edited version (rather than the
			// file in its original state) so that we preserve our initial changes.
			if content, ok := res[uri]; ok {
				m = protocol.NewMapper(uri, content)
			} else {
				var err error
				if m, err = r.data.Mapper(uri); err != nil {
					return nil, err
				}
			}
			patched, _, err := source.ApplyProtocolEdits(m, docEdits.TextDocumentEdit.Edits)
			if err != nil {
				return nil, err
			}
			res[uri] = patched
		}
	}
	return res, nil
}

func (r *runner) SignatureHelp(t *testing.T, spn span.Span, want *protocol.SignatureHelp) {
	m, err := r.data.Mapper(spn.URI())
	if err != nil {
		t.Fatal(err)
	}
	loc, err := m.SpanLocation(spn)
	if err != nil {
		t.Fatalf("failed for %v: %v", loc, err)
	}
	params := &protocol.SignatureHelpParams{
		TextDocumentPositionParams: protocol.LocationTextDocumentPositionParams(loc),
	}
	got, err := r.server.SignatureHelp(r.ctx, params)
	if err != nil {
		// Only fail if we got an error we did not expect.
		if want != nil {
			t.Fatal(err)
		}
		return
	}
	if want == nil {
		if got != nil {
			t.Errorf("expected no signature, got %v", got)
		}
		return
	}
	if got == nil {
		t.Fatalf("expected %v, got nil", want)
	}
	if diff := tests.DiffSignatures(spn, want, got); diff != "" {
		t.Error(diff)
	}
}

func (r *runner) Link(t *testing.T, uri span.URI, wantLinks []tests.Link) {
	m, err := r.data.Mapper(uri)
	if err != nil {
		t.Fatal(err)
	}
	got, err := r.server.DocumentLink(r.ctx, &protocol.DocumentLinkParams{
		TextDocument: protocol.TextDocumentIdentifier{
			URI: protocol.URIFromSpanURI(uri),
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if diff := tests.DiffLinks(m, wantLinks, got); diff != "" {
		t.Error(diff)
	}
}

func (r *runner) AddImport(t *testing.T, uri span.URI, expectedImport string) {
	cmd, err := command.NewListKnownPackagesCommand("List Known Packages", command.URIArg{
		URI: protocol.URIFromSpanURI(uri),
	})
	if err != nil {
		t.Fatal(err)
	}
	resp, err := r.server.executeCommand(r.ctx, &protocol.ExecuteCommandParams{
		Command:   cmd.Command,
		Arguments: cmd.Arguments,
	})
	if err != nil {
		t.Fatal(err)
	}
	res := resp.(command.ListKnownPackagesResult)
	var hasPkg bool
	for _, p := range res.Packages {
		if p == expectedImport {
			hasPkg = true
			break
		}
	}
	if !hasPkg {
		t.Fatalf("%s: got %v packages\nwant contains %q", command.ListKnownPackages, res.Packages, expectedImport)
	}
	cmd, err = command.NewAddImportCommand("Add Imports", command.AddImportArgs{
		URI:        protocol.URIFromSpanURI(uri),
		ImportPath: expectedImport,
	})
	if err != nil {
		t.Fatal(err)
	}
	_, err = r.server.executeCommand(r.ctx, &protocol.ExecuteCommandParams{
		Command:   cmd.Command,
		Arguments: cmd.Arguments,
	})
	if err != nil {
		t.Fatal(err)
	}
	got := (<-r.editRecv)[uri]
	want := r.data.Golden(t, "addimport", uri.Filename(), func() ([]byte, error) {
		return []byte(got), nil
	})
	if want == nil {
		t.Fatalf("golden file %q not found", uri.Filename())
	}
	if diff := compare.Bytes(want, got); diff != "" {
		t.Errorf("%s mismatch\n%s", command.AddImport, diff)
	}
}

func (r *runner) SelectionRanges(t *testing.T, spn span.Span) {
	uri := spn.URI()
	sm, err := r.data.Mapper(uri)
	if err != nil {
		t.Fatal(err)
	}
	loc, err := sm.SpanLocation(spn)
	if err != nil {
		t.Error(err)
	}

	ranges, err := r.server.selectionRange(r.ctx, &protocol.SelectionRangeParams{
		TextDocument: protocol.TextDocumentIdentifier{
			URI: protocol.URIFromSpanURI(uri),
		},
		Positions: []protocol.Position{loc.Range.Start},
	})
	if err != nil {
		t.Fatal(err)
	}

	sb := &strings.Builder{}
	for i, path := range ranges {
		fmt.Fprintf(sb, "Ranges %d: ", i)
		rng := path
		for {
			s, e, err := sm.RangeOffsets(rng.Range)
			if err != nil {
				t.Error(err)
			}

			var snippet string
			if e-s < 30 {
				snippet = string(sm.Content[s:e])
			} else {
				snippet = string(sm.Content[s:s+15]) + "..." + string(sm.Content[e-15:e])
			}

			fmt.Fprintf(sb, "\n\t%v %q", rng.Range, strings.ReplaceAll(snippet, "\n", "\\n"))

			if rng.Parent == nil {
				break
			}
			rng = *rng.Parent
		}
		sb.WriteRune('\n')
	}
	got := sb.String()

	testName := "selectionrange_" + tests.SpanName(spn)
	want := r.data.Golden(t, testName, uri.Filename(), func() ([]byte, error) {
		return []byte(got), nil
	})
	if want == nil {
		t.Fatalf("golden file %q not found", uri.Filename())
	}
	if diff := compare.Text(got, string(want)); diff != "" {
		t.Errorf("%s mismatch\n%s", testName, diff)
	}
}

func (r *runner) collectDiagnostics(view *cache.View) {
	if r.diagnostics != nil {
		return
	}
	r.diagnostics = make(map[span.URI][]*source.Diagnostic)

	snapshot, release, err := view.Snapshot()
	if err != nil {
		panic(err)
	}
	defer release()

	// Always run diagnostics with analysis.
	r.server.diagnose(r.ctx, snapshot, analyzeEverything)
	for uri, reports := range r.server.diagnostics {
		for _, report := range reports.reports {
			for _, d := range report.diags {
				r.diagnostics[uri] = append(r.diagnostics[uri], d)
			}
		}
	}
}
