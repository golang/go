// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lsp

import (
	"context"
	"fmt"
	"go/token"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	"golang.org/x/tools/gopls/internal/lsp/cache"
	"golang.org/x/tools/gopls/internal/lsp/command"
	"golang.org/x/tools/gopls/internal/lsp/protocol"
	"golang.org/x/tools/gopls/internal/lsp/source"
	"golang.org/x/tools/gopls/internal/lsp/tests"
	"golang.org/x/tools/gopls/internal/lsp/tests/compare"
	"golang.org/x/tools/gopls/internal/span"
	"golang.org/x/tools/internal/bug"
	"golang.org/x/tools/internal/event"
	"golang.org/x/tools/internal/testenv"
)

func TestMain(m *testing.M) {
	bug.PanicOnBugs = true
	testenv.ExitIfSmallMachine()

	// Set the global exporter to nil so that we don't log to stderr. This avoids
	// a lot of misleading noise in test output.
	//
	// TODO(rfindley): investigate whether we can/should capture logs scoped to
	// individual tests by passing in a context with a local exporter.
	event.SetExporter(nil)

	os.Exit(m.Run())
}

func TestLSP(t *testing.T) {
	tests.RunTests(t, "testdata", true, testLSP)
}

type runner struct {
	server      *Server
	data        *tests.Data
	diagnostics map[span.URI][]*source.Diagnostic
	ctx         context.Context
	normalizers []tests.Normalizer
	editRecv    chan map[span.URI]string
}

func testLSP(t *testing.T, datum *tests.Data) {
	ctx := tests.Context(t)

	cache := cache.New(nil, nil, nil)
	session := cache.NewSession(ctx)
	options := source.DefaultOptions().Clone()
	tests.DefaultOptions(options)
	session.SetOptions(options)
	options.SetEnvSlice(datum.Config.Env)
	view, snapshot, release, err := session.NewView(ctx, datum.Config.Dir, span.URIFromPath(datum.Config.Dir), options)
	if err != nil {
		t.Fatal(err)
	}

	defer view.Shutdown(ctx)

	// Enable type error analyses for tests.
	// TODO(golang/go#38212): Delete this once they are enabled by default.
	tests.EnableAllAnalyzers(view, options)
	view.SetOptions(ctx, options)

	// Enable all inlay hints for tests.
	tests.EnableAllInlayHints(view, options)

	// Only run the -modfile specific tests in module mode with Go 1.14 or above.
	datum.ModfileFlagAvailable = len(snapshot.ModFiles()) > 0 && testenv.Go1Point() >= 14
	release()

	var modifications []source.FileModification
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
		data:        datum,
		ctx:         ctx,
		normalizers: tests.CollectNormalizers(datum.Exported),
		editRecv:    make(chan map[span.URI]string, 1),
	}

	r.server = NewServer(session, testClient{runner: r})
	tests.Run(t, r, datum)
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
	loc, err := mapper.Location(spn)
	if err != nil {
		t.Fatalf("failed for %v: %v", spn, err)
	}

	params := &protocol.CallHierarchyPrepareParams{
		TextDocumentPositionParams: protocol.TextDocumentPositionParams{
			TextDocument: protocol.TextDocumentIdentifier{URI: loc.URI},
			Position:     loc.Range.Start,
		},
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
		t.Error(fmt.Sprintf("incoming calls: %s", msg))
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
		t.Error(fmt.Sprintf("outgoing calls: %s", msg))
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
	v := r.server.session.View(r.data.Config.Dir)
	r.collectDiagnostics(v)
	got := append([]*source.Diagnostic(nil), r.diagnostics[uri]...) // copy
	tests.CompareDiagnostics(t, uri, want, got)
}

func (r *runner) FoldingRanges(t *testing.T, spn span.Span) {
	uri := spn.URI()
	view, err := r.server.session.ViewOf(uri)
	if err != nil {
		t.Fatal(err)
	}
	original := view.Options()
	modified := original

	// Test all folding ranges.
	modified.LineFoldingOnly = false
	view, err = view.SetOptions(r.ctx, modified)
	if err != nil {
		t.Error(err)
		return
	}
	ranges, err := r.server.FoldingRange(r.ctx, &protocol.FoldingRangeParams{
		TextDocument: protocol.TextDocumentIdentifier{
			URI: protocol.URIFromSpanURI(uri),
		},
	})
	if err != nil {
		t.Error(err)
		return
	}
	r.foldingRanges(t, "foldingRange", uri, ranges)

	// Test folding ranges with lineFoldingOnly = true.
	modified.LineFoldingOnly = true
	view, err = view.SetOptions(r.ctx, modified)
	if err != nil {
		t.Error(err)
		return
	}
	ranges, err = r.server.FoldingRange(r.ctx, &protocol.FoldingRangeParams{
		TextDocument: protocol.TextDocumentIdentifier{
			URI: protocol.URIFromSpanURI(uri),
		},
	})
	if err != nil {
		t.Error(err)
		return
	}
	r.foldingRanges(t, "foldingRange-lineFolding", uri, ranges)
	view.SetOptions(r.ctx, original)
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

func foldRanges(m *protocol.ColumnMapper, contents string, ranges []protocol.FoldingRange) (string, error) {
	foldedText := "<>"
	res := contents
	// Apply the edits from the end of the file forward
	// to preserve the offsets
	// TODO(adonovan): factor to use diff.ApplyEdits, which validates the input.
	for i := len(ranges) - 1; i >= 0; i-- {
		r := ranges[i]
		start, err := m.Point(protocol.Position{r.StartLine, r.StartCharacter})
		if err != nil {
			return "", err
		}
		end, err := m.Point(protocol.Position{r.EndLine, r.EndCharacter})
		if err != nil {
			return "", err
		}
		res = res[:start.Offset()] + foldedText + res[end.Offset():]
	}
	return res, nil
}

func (r *runner) Format(t *testing.T, spn span.Span) {
	uri := spn.URI()
	filename := uri.Filename()
	gofmted := string(r.data.Golden(t, "gofmt", filename, func() ([]byte, error) {
		cmd := exec.Command("gofmt", filename)
		out, _ := cmd.Output() // ignore error, sometimes we have intentionally ungofmt-able files
		return out, nil
	}))

	edits, err := r.server.Formatting(r.ctx, &protocol.DocumentFormattingParams{
		TextDocument: protocol.TextDocumentIdentifier{
			URI: protocol.URIFromSpanURI(uri),
		},
	})
	if err != nil {
		if gofmted != "" {
			t.Error(err)
		}
		return
	}
	m, err := r.data.Mapper(uri)
	if err != nil {
		t.Fatal(err)
	}
	got, _, err := source.ApplyProtocolEdits(m, edits)
	if err != nil {
		t.Error(err)
	}
	if diff := compare.Text(gofmted, got); diff != "" {
		t.Errorf("format failed for %s (-want +got):\n%s", filename, diff)
	}
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

func (r *runner) Import(t *testing.T, spn span.Span) {
	uri := spn.URI()
	filename := uri.Filename()
	actions, err := r.server.CodeAction(r.ctx, &protocol.CodeActionParams{
		TextDocument: protocol.TextDocumentIdentifier{
			URI: protocol.URIFromSpanURI(uri),
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	m, err := r.data.Mapper(uri)
	if err != nil {
		t.Fatal(err)
	}
	got := string(m.Content)
	if len(actions) > 0 {
		res, err := applyTextDocumentEdits(r, actions[0].Edit.DocumentChanges)
		if err != nil {
			t.Fatal(err)
		}
		got = res[uri]
	}
	want := string(r.data.Golden(t, "goimports", filename, func() ([]byte, error) {
		return []byte(got), nil
	}))

	if d := compare.Text(want, got); d != "" {
		t.Errorf("import failed for %s:\n%s", filename, d)
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
	rng, err := m.Range(spn)
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
	var res map[span.URI]string
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
		want := string(r.data.Golden(t, "suggestedfix_"+tests.SpanName(spn), u.Filename(), func() ([]byte, error) {
			return []byte(got), nil
		}))
		if want != got {
			t.Errorf("suggested fixes failed for %s:\n%s", u.Filename(), compare.Text(want, got))
		}
	}
}

func (r *runner) FunctionExtraction(t *testing.T, start span.Span, end span.Span) {
	uri := start.URI()
	m, err := r.data.Mapper(uri)
	if err != nil {
		t.Fatal(err)
	}
	spn := span.New(start.URI(), start.Start(), end.End())
	rng, err := m.Range(spn)
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
		if action.Command.Title == "Extract function" {
			actions = append(actions, action)
		}
	}
	// Hack: We assume that we only get one code action per range.
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
		want := string(r.data.Golden(t, "functionextraction_"+tests.SpanName(spn), u.Filename(), func() ([]byte, error) {
			return []byte(got), nil
		}))
		if want != got {
			t.Errorf("function extraction failed for %s:\n%s", u.Filename(), compare.Text(want, got))
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
	rng, err := m.Range(spn)
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
		want := string(r.data.Golden(t, "methodextraction_"+tests.SpanName(spn), u.Filename(), func() ([]byte, error) {
			return []byte(got), nil
		}))
		if want != got {
			t.Errorf("method extraction failed for %s:\n%s", u.Filename(), compare.Text(want, got))
		}
	}
}

func (r *runner) Definition(t *testing.T, spn span.Span, d tests.Definition) {
	sm, err := r.data.Mapper(d.Src.URI())
	if err != nil {
		t.Fatal(err)
	}
	loc, err := sm.Location(d.Src)
	if err != nil {
		t.Fatalf("failed for %v: %v", d.Src, err)
	}
	tdpp := protocol.TextDocumentPositionParams{
		TextDocument: protocol.TextDocumentIdentifier{URI: loc.URI},
		Position:     loc.Range.Start,
	}
	var locs []protocol.Location
	var hover *protocol.Hover
	if d.IsType {
		params := &protocol.TypeDefinitionParams{
			TextDocumentPositionParams: tdpp,
		}
		locs, err = r.server.TypeDefinition(r.ctx, params)
	} else {
		params := &protocol.DefinitionParams{
			TextDocumentPositionParams: tdpp,
		}
		locs, err = r.server.Definition(r.ctx, params)
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
	if len(locs) != 1 {
		t.Errorf("got %d locations for definition, expected 1", len(locs))
	}
	didSomething := false
	if hover != nil {
		didSomething = true
		tag := fmt.Sprintf("%s-hoverdef", d.Name)
		expectHover := string(r.data.Golden(t, tag, d.Src.URI().Filename(), func() ([]byte, error) {
			return []byte(hover.Contents.Value), nil
		}))
		got := tests.StripSubscripts(hover.Contents.Value)
		expectHover = tests.StripSubscripts(expectHover)
		if got != expectHover {
			tests.CheckSameMarkdown(t, got, expectHover)
		}
	}
	if !d.OnlyHover {
		didSomething = true
		locURI := locs[0].URI.SpanURI()
		lm, err := r.data.Mapper(locURI)
		if err != nil {
			t.Fatal(err)
		}
		if def, err := lm.Span(locs[0]); err != nil {
			t.Fatalf("failed for %v: %v", locs[0], err)
		} else if def != d.Def {
			t.Errorf("for %v got %v want %v", d.Src, def, d.Def)
		}
	}
	if !didSomething {
		t.Errorf("no tests ran for %s", d.Src.URI())
	}
}

func (r *runner) Implementation(t *testing.T, spn span.Span, impls []span.Span) {
	sm, err := r.data.Mapper(spn.URI())
	if err != nil {
		t.Fatal(err)
	}
	loc, err := sm.Location(spn)
	if err != nil {
		t.Fatalf("failed for %v: %v", spn, err)
	}
	tdpp := protocol.TextDocumentPositionParams{
		TextDocument: protocol.TextDocumentIdentifier{URI: loc.URI},
		Position:     loc.Range.Start,
	}
	var locs []protocol.Location
	params := &protocol.ImplementationParams{
		TextDocumentPositionParams: tdpp,
	}
	locs, err = r.server.Implementation(r.ctx, params)
	if err != nil {
		t.Fatalf("failed for %v: %v", spn, err)
	}
	if len(locs) != len(impls) {
		t.Fatalf("got %d locations for implementation, expected %d", len(locs), len(impls))
	}

	var results []span.Span
	for i := range locs {
		locURI := locs[i].URI.SpanURI()
		lm, err := r.data.Mapper(locURI)
		if err != nil {
			t.Fatal(err)
		}
		imp, err := lm.Span(locs[i])
		if err != nil {
			t.Fatalf("failed for %v: %v", locs[i], err)
		}
		results = append(results, imp)
	}
	// Sort results and expected to make tests deterministic.
	sort.SliceStable(results, func(i, j int) bool {
		return span.Compare(results[i], results[j]) == -1
	})
	sort.SliceStable(impls, func(i, j int) bool {
		return span.Compare(impls[i], impls[j]) == -1
	})
	for i := range results {
		if results[i] != impls[i] {
			t.Errorf("for %dth implementation of %v got %v want %v", i, spn, results[i], impls[i])
		}
	}
}

func (r *runner) Highlight(t *testing.T, src span.Span, locations []span.Span) {
	m, err := r.data.Mapper(src.URI())
	if err != nil {
		t.Fatal(err)
	}
	loc, err := m.Location(src)
	if err != nil {
		t.Fatalf("failed for %v: %v", locations[0], err)
	}
	tdpp := protocol.TextDocumentPositionParams{
		TextDocument: protocol.TextDocumentIdentifier{URI: loc.URI},
		Position:     loc.Range.Start,
	}
	params := &protocol.DocumentHighlightParams{
		TextDocumentPositionParams: tdpp,
	}
	highlights, err := r.server.DocumentHighlight(r.ctx, params)
	if err != nil {
		t.Fatal(err)
	}
	if len(highlights) != len(locations) {
		t.Fatalf("got %d highlights for highlight at %v:%v:%v, expected %d", len(highlights), src.URI().Filename(), src.Start().Line(), src.Start().Column(), len(locations))
	}
	// Check to make sure highlights have a valid range.
	var results []span.Span
	for i := range highlights {
		h, err := m.RangeSpan(highlights[i].Range)
		if err != nil {
			t.Fatalf("failed for %v: %v", highlights[i], err)
		}
		results = append(results, h)
	}
	// Sort results to make tests deterministic since DocumentHighlight uses a map.
	sort.SliceStable(results, func(i, j int) bool {
		return span.Compare(results[i], results[j]) == -1
	})
	// Check to make sure all the expected highlights are found.
	for i := range results {
		if results[i] != locations[i] {
			t.Errorf("want %v, got %v\n", locations[i], results[i])
		}
	}
}

func (r *runner) Hover(t *testing.T, src span.Span, text string) {
	m, err := r.data.Mapper(src.URI())
	if err != nil {
		t.Fatal(err)
	}
	loc, err := m.Location(src)
	if err != nil {
		t.Fatalf("failed for %v", err)
	}
	tdpp := protocol.TextDocumentPositionParams{
		TextDocument: protocol.TextDocumentIdentifier{URI: loc.URI},
		Position:     loc.Range.Start,
	}
	params := &protocol.HoverParams{
		TextDocumentPositionParams: tdpp,
	}
	hover, err := r.server.Hover(r.ctx, params)
	if err != nil {
		t.Fatal(err)
	}
	if text == "" {
		if hover != nil {
			t.Errorf("want nil, got %v\n", hover)
		}
	} else {
		if hover == nil {
			t.Fatalf("want hover result to include %s, but got nil", text)
		}
		if got := hover.Contents.Value; got != text {
			t.Errorf("want %v, got %v\n", text, got)
		}
		if want, got := loc.Range, hover.Range; want != got {
			t.Errorf("want range %v, got %v instead", want, got)
		}
	}
}

func (r *runner) References(t *testing.T, src span.Span, itemList []span.Span) {
	sm, err := r.data.Mapper(src.URI())
	if err != nil {
		t.Fatal(err)
	}
	loc, err := sm.Location(src)
	if err != nil {
		t.Fatalf("failed for %v: %v", src, err)
	}
	for _, includeDeclaration := range []bool{true, false} {
		t.Run(fmt.Sprintf("refs-declaration-%v", includeDeclaration), func(t *testing.T) {
			want := make(map[protocol.Location]bool)
			for i, pos := range itemList {
				// We don't want the first result if we aren't including the declaration.
				if i == 0 && !includeDeclaration {
					continue
				}
				m, err := r.data.Mapper(pos.URI())
				if err != nil {
					t.Fatal(err)
				}
				loc, err := m.Location(pos)
				if err != nil {
					t.Fatalf("failed for %v: %v", src, err)
				}
				want[loc] = true
			}
			params := &protocol.ReferenceParams{
				TextDocumentPositionParams: protocol.TextDocumentPositionParams{
					TextDocument: protocol.TextDocumentIdentifier{URI: loc.URI},
					Position:     loc.Range.Start,
				},
				Context: protocol.ReferenceContext{
					IncludeDeclaration: includeDeclaration,
				},
			}
			got, err := r.server.References(r.ctx, params)
			if err != nil {
				t.Fatalf("failed for %v: %v", src, err)
			}
			if len(got) != len(want) {
				t.Errorf("references failed: different lengths got %v want %v", len(got), len(want))
			}
			for _, loc := range got {
				if !want[loc] {
					t.Errorf("references failed: incorrect references got %v want %v", loc, want)
				}
			}
		})
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
			Range:   protocol.Range{Start: *hint.Position, End: *hint.Position},
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

	withinlayHints := string(r.data.Golden(t, "inlayHint", filename, func() ([]byte, error) {
		return []byte(got), nil
	}))

	if withinlayHints != got {
		t.Errorf("inlay hints failed for %s, expected:\n%v\ngot:\n%v", filename, withinlayHints, got)
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
	loc, err := sm.Location(spn)
	if err != nil {
		t.Fatalf("failed for %v: %v", spn, err)
	}

	wedit, err := r.server.Rename(r.ctx, &protocol.RenameParams{
		TextDocument: protocol.TextDocumentIdentifier{
			URI: protocol.URIFromSpanURI(uri),
		},
		Position: loc.Range.Start,
		NewName:  newText,
	})
	if err != nil {
		renamed := string(r.data.Golden(t, tag, filename, func() ([]byte, error) {
			return []byte(err.Error()), nil
		}))
		if err.Error() != renamed {
			t.Errorf("rename failed for %s, expected:\n%v\ngot:\n%v\n", newText, renamed, err)
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

	var got string
	for i := 0; i < len(res); i++ {
		if i != 0 {
			got += "\n"
		}
		uri := span.URIFromURI(orderedURIs[i])
		if len(res) > 1 {
			got += filepath.Base(uri.Filename()) + ":\n"
		}
		val := res[uri]
		got += val
	}
	want := string(r.data.Golden(t, tag, filename, func() ([]byte, error) {
		return []byte(got), nil
	}))
	if want != got {
		t.Errorf("rename failed for %s:\n%s", newText, compare.Text(want, got))
	}
}

func (r *runner) PrepareRename(t *testing.T, src span.Span, want *source.PrepareItem) {
	m, err := r.data.Mapper(src.URI())
	if err != nil {
		t.Fatal(err)
	}
	loc, err := m.Location(src)
	if err != nil {
		t.Fatalf("failed for %v: %v", src, err)
	}
	tdpp := protocol.TextDocumentPositionParams{
		TextDocument: protocol.TextDocumentIdentifier{URI: loc.URI},
		Position:     loc.Range.Start,
	}
	params := &protocol.PrepareRenameParams{
		TextDocumentPositionParams: tdpp,
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

func applyTextDocumentEdits(r *runner, edits []protocol.DocumentChanges) (map[span.URI]string, error) {
	res := map[span.URI]string{}
	for _, docEdits := range edits {
		if docEdits.TextDocumentEdit != nil {
			uri := docEdits.TextDocumentEdit.TextDocument.URI.SpanURI()
			var m *protocol.ColumnMapper
			// If we have already edited this file, we use the edited version (rather than the
			// file in its original state) so that we preserve our initial changes.
			if content, ok := res[uri]; ok {
				m = protocol.NewColumnMapper(uri, []byte(content))
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

func (r *runner) Symbols(t *testing.T, uri span.URI, expectedSymbols []protocol.DocumentSymbol) {
	params := &protocol.DocumentSymbolParams{
		TextDocument: protocol.TextDocumentIdentifier{
			URI: protocol.URIFromSpanURI(uri),
		},
	}
	got, err := r.server.DocumentSymbol(r.ctx, params)
	if err != nil {
		t.Fatal(err)
	}

	symbols := make([]protocol.DocumentSymbol, len(got))
	for i, s := range got {
		s, ok := s.(protocol.DocumentSymbol)
		if !ok {
			t.Fatalf("%v: wanted []DocumentSymbols but got %v", uri, got)
		}
		symbols[i] = s
	}

	// Sort by position to make it easier to find errors.
	sortSymbols := func(s []protocol.DocumentSymbol) {
		sort.Slice(s, func(i, j int) bool {
			return protocol.CompareRange(s[i].SelectionRange, s[j].SelectionRange) < 0
		})
	}
	sortSymbols(expectedSymbols)
	sortSymbols(symbols)

	// Ignore 'Range' here as it is difficult (impossible?) to express
	// multi-line ranges in the packagestest framework.
	ignoreRange := cmpopts.IgnoreFields(protocol.DocumentSymbol{}, "Range")
	if diff := cmp.Diff(expectedSymbols, symbols, ignoreRange); diff != "" {
		t.Errorf("mismatching symbols (-want +got)\n%s", diff)
	}
}

func (r *runner) WorkspaceSymbols(t *testing.T, uri span.URI, query string, typ tests.WorkspaceSymbolsTestType) {
	matcher := tests.WorkspaceSymbolsTestTypeToMatcher(typ)

	original := r.server.session.Options()
	modified := original
	modified.SymbolMatcher = matcher
	r.server.session.SetOptions(modified)
	defer r.server.session.SetOptions(original)

	params := &protocol.WorkspaceSymbolParams{
		Query: query,
	}
	gotSymbols, err := r.server.Symbol(r.ctx, params)
	if err != nil {
		t.Fatal(err)
	}
	got, err := tests.WorkspaceSymbolsString(r.ctx, r.data, uri, gotSymbols)
	if err != nil {
		t.Fatal(err)
	}
	got = filepath.ToSlash(tests.Normalize(got, r.normalizers))
	want := string(r.data.Golden(t, fmt.Sprintf("workspace_symbol-%s-%s", strings.ToLower(string(matcher)), query), uri.Filename(), func() ([]byte, error) {
		return []byte(got), nil
	}))
	if diff := compare.Text(want, got); diff != "" {
		t.Error(diff)
	}
}

func (r *runner) SignatureHelp(t *testing.T, spn span.Span, want *protocol.SignatureHelp) {
	m, err := r.data.Mapper(spn.URI())
	if err != nil {
		t.Fatal(err)
	}
	loc, err := m.Location(spn)
	if err != nil {
		t.Fatalf("failed for %v: %v", loc, err)
	}
	tdpp := protocol.TextDocumentPositionParams{
		TextDocument: protocol.TextDocumentIdentifier{
			URI: protocol.URIFromSpanURI(spn.URI()),
		},
		Position: loc.Range.Start,
	}
	params := &protocol.SignatureHelpParams{
		TextDocumentPositionParams: tdpp,
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
	if diff := compare.Text(got, string(want)); diff != "" {
		t.Errorf("%s mismatch\n%s", command.AddImport, diff)
	}
}

func TestBytesOffset(t *testing.T) {
	tests := []struct {
		text string
		pos  protocol.Position
		want int
	}{
		{text: `a𐐀b`, pos: protocol.Position{Line: 0, Character: 0}, want: 0},
		{text: `a𐐀b`, pos: protocol.Position{Line: 0, Character: 1}, want: 1},
		{text: `a𐐀b`, pos: protocol.Position{Line: 0, Character: 2}, want: 1},
		{text: `a𐐀b`, pos: protocol.Position{Line: 0, Character: 3}, want: 5},
		{text: `a𐐀b`, pos: protocol.Position{Line: 0, Character: 4}, want: 6},
		{text: `a𐐀b`, pos: protocol.Position{Line: 0, Character: 5}, want: -1},
		{text: "aaa\nbbb\n", pos: protocol.Position{Line: 0, Character: 3}, want: 3},
		{text: "aaa\nbbb\n", pos: protocol.Position{Line: 0, Character: 4}, want: 3},
		{text: "aaa\nbbb\n", pos: protocol.Position{Line: 1, Character: 0}, want: 4},
		{text: "aaa\nbbb\n", pos: protocol.Position{Line: 1, Character: 3}, want: 7},
		{text: "aaa\nbbb\n", pos: protocol.Position{Line: 1, Character: 4}, want: 7},
		{text: "aaa\nbbb\n", pos: protocol.Position{Line: 2, Character: 0}, want: 8},
		{text: "aaa\nbbb\n", pos: protocol.Position{Line: 2, Character: 1}, want: -1},
		{text: "aaa\nbbb\n\n", pos: protocol.Position{Line: 2, Character: 0}, want: 8},
	}

	for i, test := range tests {
		fname := fmt.Sprintf("test %d", i)
		fset := token.NewFileSet()
		f := fset.AddFile(fname, -1, len(test.text))
		f.SetLinesForContent([]byte(test.text))
		uri := span.URIFromPath(fname)
		mapper := protocol.NewColumnMapper(uri, []byte(test.text))
		got, err := mapper.Point(test.pos)
		if err != nil && test.want != -1 {
			t.Errorf("unexpected error: %v", err)
		}
		if err == nil && got.Offset() != test.want {
			t.Errorf("want %d for %q(Line:%d,Character:%d), but got %d", test.want, test.text, int(test.pos.Line), int(test.pos.Character), got.Offset())
		}
	}
}

func (r *runner) collectDiagnostics(view source.View) {
	if r.diagnostics != nil {
		return
	}
	r.diagnostics = make(map[span.URI][]*source.Diagnostic)

	snapshot, release := view.Snapshot(r.ctx)
	defer release()

	// Always run diagnostics with analysis.
	r.server.diagnose(r.ctx, snapshot, true)
	for uri, reports := range r.server.diagnostics {
		for _, report := range reports.reports {
			for _, d := range report.diags {
				r.diagnostics[uri] = append(r.diagnostics[uri], d)
			}
		}
	}
}
