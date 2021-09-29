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

	"golang.org/x/tools/internal/lsp/cache"
	"golang.org/x/tools/internal/lsp/command"
	"golang.org/x/tools/internal/lsp/diff"
	"golang.org/x/tools/internal/lsp/diff/myers"
	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/lsp/tests"
	"golang.org/x/tools/internal/span"
	"golang.org/x/tools/internal/testenv"
)

func TestMain(m *testing.M) {
	testenv.ExitIfSmallMachine()
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

	cache := cache.New(nil)
	session := cache.NewSession(ctx)
	options := source.DefaultOptions().Clone()
	tests.DefaultOptions(options)
	session.SetOptions(options)
	options.SetEnvSlice(datum.Config.Env)
	view, snapshot, release, err := session.NewView(ctx, datum.Config.Dir, span.URIFromPath(datum.Config.Dir), "", options)
	if err != nil {
		t.Fatal(err)
	}

	defer view.Shutdown(ctx)

	// Enable type error analyses for tests.
	// TODO(golang/go#38212): Delete this once they are enabled by default.
	tests.EnableAllAnalyzers(view, options)
	view.SetOptions(ctx, options)

	// Only run the -modfile specific tests in module mode with Go 1.14 or above.
	datum.ModfileFlagAvailable = len(snapshot.ModFiles()) > 0 && testenv.Go1Point() >= 14
	release()

	var modifications []source.FileModification
	for filename, content := range datum.Config.Overlay {
		kind := source.DetectLanguage("", filename)
		if kind != source.Go {
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
	if source.DetectLanguage("", uri.Filename()) != source.Mod {
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
	d := r.diagnostics[uri]
	got := make([]*source.Diagnostic, len(d))
	copy(got, d)
	// A special case to test that there are no diagnostics for a file.
	if len(want) == 1 && want[0].Source == "no_diagnostics" {
		if len(got) != 0 {
			t.Errorf("expected no diagnostics for %s, got %v", uri, got)
		}
		return
	}
	if diff := tests.DiffDiagnostics(uri, want, got); diff != "" {
		t.Error(diff)
	}
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
		want := string(r.data.Golden(tag, uri.Filename(), func() ([]byte, error) {
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
			want := string(r.data.Golden(tag, uri.Filename(), func() ([]byte, error) {
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
	for i := len(ranges) - 1; i >= 0; i-- {
		fRange := ranges[i]
		spn, err := m.RangeSpan(protocol.Range{
			Start: protocol.Position{
				Line:      fRange.StartLine,
				Character: fRange.StartCharacter,
			},
			End: protocol.Position{
				Line:      fRange.EndLine,
				Character: fRange.EndCharacter,
			},
		})
		if err != nil {
			return "", err
		}
		start := spn.Start().Offset()
		end := spn.End().Offset()

		tmp := res[0:start] + foldedText
		res = tmp + res[end:]
	}
	return res, nil
}

func (r *runner) Format(t *testing.T, spn span.Span) {
	uri := spn.URI()
	filename := uri.Filename()
	gofmted := string(r.data.Golden("gofmt", filename, func() ([]byte, error) {
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
	sedits, err := source.FromProtocolEdits(m, edits)
	if err != nil {
		t.Error(err)
	}
	got := diff.ApplyEdits(string(m.Content), sedits)
	if gofmted != got {
		t.Errorf("format failed for %s, expected:\n%v\ngot:\n%v", filename, gofmted, got)
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
	want := string(r.data.Golden("goimports", filename, func() ([]byte, error) {
		return []byte(got), nil
	}))
	if want != got {
		d, err := myers.ComputeEdits(uri, want, got)
		if err != nil {
			t.Fatal(err)
		}
		t.Errorf("import failed for %s: %s", filename, diff.ToUnified("want", "got", want, d))
	}
}

func (r *runner) SuggestedFix(t *testing.T, spn span.Span, actionKinds []string, expectedActions int) {
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
	codeActionKinds := []protocol.CodeActionKind{}
	for _, k := range actionKinds {
		codeActionKinds = append(codeActionKinds, protocol.CodeActionKind(k))
	}
	actions, err := r.server.CodeAction(r.ctx, &protocol.CodeActionParams{
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
	if len(actions) != expectedActions {
		// Hack: We assume that we only get one code action per range.
		var cmds []string
		for _, a := range actions {
			cmds = append(cmds, fmt.Sprintf("%s (%s)", a.Command, a.Title))
		}
		t.Fatalf("unexpected number of code actions, want %d, got %d: %v", expectedActions, len(actions), cmds)
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
		t.Fatalf("unexpected kind for code action %s, expected one of %v, got %v", action.Title, codeActionKinds, action.Kind)
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
		want := string(r.data.Golden("suggestedfix_"+tests.SpanName(spn), u.Filename(), func() ([]byte, error) {
			return []byte(got), nil
		}))
		if want != got {
			t.Errorf("suggested fixes failed for %s:\n%s", u.Filename(), tests.Diff(t, want, got))
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
		want := string(r.data.Golden("functionextraction_"+tests.SpanName(spn), u.Filename(), func() ([]byte, error) {
			return []byte(got), nil
		}))
		if want != got {
			t.Errorf("function extraction failed for %s:\n%s", u.Filename(), tests.Diff(t, want, got))
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
		want := string(r.data.Golden("methodextraction_"+tests.SpanName(spn), u.Filename(), func() ([]byte, error) {
			return []byte(got), nil
		}))
		if want != got {
			t.Errorf("method extraction failed for %s:\n%s", u.Filename(), tests.Diff(t, want, got))
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
		expectHover := string(r.data.Golden(tag, d.Src.URI().Filename(), func() ([]byte, error) {
			return []byte(hover.Contents.Value), nil
		}))
		if hover.Contents.Value != expectHover {
			t.Errorf("%s:\n%s", d.Src, tests.Diff(t, expectHover, hover.Contents.Value))
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
		renamed := string(r.data.Golden(tag, filename, func() ([]byte, error) {
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
	want := string(r.data.Golden(tag, filename, func() ([]byte, error) {
		return []byte(got), nil
	}))
	if want != got {
		t.Errorf("rename failed for %s:\n%s", newText, tests.Diff(t, want, got))
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
	// we all love typed nils
	if got == nil {
		if want.Text != "" { // expected an ident.
			t.Errorf("prepare rename failed for %v: got nil", src)
		}
		return
	}
	if got.Start == got.End {
		// Special case for 0-length ranges. Marks can't specify a 0-length range,
		// so just compare the start.
		if got.Start != want.Range.Start {
			t.Errorf("prepare rename failed: incorrect point, got %v want %v", got.Start, want.Range.Start)
		}
	} else {
		if protocol.CompareRange(*got, want.Range) != 0 {
			t.Errorf("prepare rename failed: incorrect range got %v want %v", *got, want.Range)
		}
	}
}

func applyTextDocumentEdits(r *runner, edits []protocol.TextDocumentEdit) (map[span.URI]string, error) {
	res := map[span.URI]string{}
	for _, docEdits := range edits {
		uri := docEdits.TextDocument.URI.SpanURI()
		var m *protocol.ColumnMapper
		// If we have already edited this file, we use the edited version (rather than the
		// file in its original state) so that we preserve our initial changes.
		if content, ok := res[uri]; ok {
			m = &protocol.ColumnMapper{
				URI: uri,
				Converter: span.NewContentConverter(
					uri.Filename(), []byte(content)),
				Content: []byte(content),
			}
		} else {
			var err error
			if m, err = r.data.Mapper(uri); err != nil {
				return nil, err
			}
		}
		res[uri] = string(m.Content)
		sedits, err := source.FromProtocolEdits(m, docEdits.Edits)
		if err != nil {
			return nil, err
		}
		res[uri] = applyEdits(res[uri], sedits)
	}
	return res, nil
}

func applyEdits(contents string, edits []diff.TextEdit) string {
	res := contents

	// Apply the edits from the end of the file forward
	// to preserve the offsets
	for i := len(edits) - 1; i >= 0; i-- {
		edit := edits[i]
		start := edit.Span.Start().Offset()
		end := edit.Span.End().Offset()
		tmp := res[0:start] + edit.NewText
		res = tmp + res[end:]
	}
	return res
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
	if len(got) != len(expectedSymbols) {
		t.Errorf("want %d top-level symbols in %v, got %d", len(expectedSymbols), uri, len(got))
		return
	}
	symbols := make([]protocol.DocumentSymbol, len(got))
	for i, s := range got {
		s, ok := s.(protocol.DocumentSymbol)
		if !ok {
			t.Fatalf("%v: wanted []DocumentSymbols but got %v", uri, got)
		}
		symbols[i] = s
	}
	if diff := tests.DiffSymbols(t, uri, expectedSymbols, symbols); diff != "" {
		t.Error(diff)
	}
}

func (r *runner) WorkspaceSymbols(t *testing.T, uri span.URI, query string, typ tests.WorkspaceSymbolsTestType) {
	r.callWorkspaceSymbols(t, uri, query, typ)
}

func (r *runner) callWorkspaceSymbols(t *testing.T, uri span.URI, query string, typ tests.WorkspaceSymbolsTestType) {
	t.Helper()

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
	want := string(r.data.Golden(fmt.Sprintf("workspace_symbol-%s-%s", strings.ToLower(string(matcher)), query), uri.Filename(), func() ([]byte, error) {
		return []byte(got), nil
	}))
	if diff := tests.Diff(t, want, got); diff != "" {
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
	diff, err := tests.DiffSignatures(spn, want, got)
	if err != nil {
		t.Fatal(err)
	}
	if diff != "" {
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
	want := r.data.Golden("addimport", uri.Filename(), func() ([]byte, error) {
		return []byte(got), nil
	})
	if want == nil {
		t.Fatalf("golden file %q not found", uri.Filename())
	}
	if diff := tests.Diff(t, got, string(want)); diff != "" {
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
		converter := span.NewContentConverter(fname, []byte(test.text))
		mapper := &protocol.ColumnMapper{
			URI:       uri,
			Converter: converter,
			Content:   []byte(test.text),
		}
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
