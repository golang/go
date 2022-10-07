// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package cmdtest contains the test suite for the command line behavior of gopls.
package cmdtest

import (
	"bytes"
	"context"
	"flag"
	"fmt"
	"io"
	"os"
	"runtime"
	"sync"
	"testing"

	"golang.org/x/tools/gopls/internal/lsp/cache"
	"golang.org/x/tools/gopls/internal/lsp/cmd"
	"golang.org/x/tools/gopls/internal/lsp/debug"
	"golang.org/x/tools/gopls/internal/lsp/lsprpc"
	"golang.org/x/tools/gopls/internal/lsp/protocol"
	"golang.org/x/tools/gopls/internal/lsp/source"
	"golang.org/x/tools/gopls/internal/lsp/tests"
	"golang.org/x/tools/gopls/internal/span"
	"golang.org/x/tools/internal/jsonrpc2/servertest"
	"golang.org/x/tools/internal/tool"
)

type runner struct {
	data        *tests.Data
	ctx         context.Context
	options     func(*source.Options)
	normalizers []tests.Normalizer
	remote      string
}

func TestCommandLine(t *testing.T, testdata string, options func(*source.Options)) {
	// On Android, the testdata directory is not copied to the runner.
	if runtime.GOOS == "android" {
		t.Skip("testdata directory not present on android")
	}
	tests.RunTests(t, testdata, false, func(t *testing.T, datum *tests.Data) {
		ctx := tests.Context(t)
		ts := NewTestServer(ctx, options)
		tests.Run(t, NewRunner(datum, ctx, ts.Addr, options), datum)
		cmd.CloseTestConnections(ctx)
	})
}

func NewTestServer(ctx context.Context, options func(*source.Options)) *servertest.TCPServer {
	ctx = debug.WithInstance(ctx, "", "")
	cache := cache.New(nil, nil, options)
	ss := lsprpc.NewStreamServer(cache, false)
	return servertest.NewTCPServer(ctx, ss, nil)
}

func NewRunner(data *tests.Data, ctx context.Context, remote string, options func(*source.Options)) *runner {
	return &runner{
		data:        data,
		ctx:         ctx,
		options:     options,
		normalizers: tests.CollectNormalizers(data.Exported),
		remote:      remote,
	}
}

func (r *runner) CodeLens(t *testing.T, uri span.URI, want []protocol.CodeLens) {
	//TODO: add command line completions tests when it works
}

func (r *runner) Completion(t *testing.T, src span.Span, test tests.Completion, items tests.CompletionItems) {
	//TODO: add command line completions tests when it works
}

func (r *runner) CompletionSnippet(t *testing.T, src span.Span, expected tests.CompletionSnippet, placeholders bool, items tests.CompletionItems) {
	//TODO: add command line completions tests when it works
}

func (r *runner) UnimportedCompletion(t *testing.T, src span.Span, test tests.Completion, items tests.CompletionItems) {
	//TODO: add command line completions tests when it works
}

func (r *runner) DeepCompletion(t *testing.T, src span.Span, test tests.Completion, items tests.CompletionItems) {
	//TODO: add command line completions tests when it works
}

func (r *runner) FuzzyCompletion(t *testing.T, src span.Span, test tests.Completion, items tests.CompletionItems) {
	//TODO: add command line completions tests when it works
}

func (r *runner) CaseSensitiveCompletion(t *testing.T, src span.Span, test tests.Completion, items tests.CompletionItems) {
	//TODO: add command line completions tests when it works
}

func (r *runner) RankCompletion(t *testing.T, src span.Span, test tests.Completion, items tests.CompletionItems) {
	//TODO: add command line completions tests when it works
}

func (r *runner) FunctionExtraction(t *testing.T, start span.Span, end span.Span) {
	//TODO: function extraction not supported on command line
}

func (r *runner) MethodExtraction(t *testing.T, start span.Span, end span.Span) {
	//TODO: function extraction not supported on command line
}

func (r *runner) AddImport(t *testing.T, uri span.URI, expectedImport string) {
	//TODO: import addition not supported on command line
}

func (r *runner) Hover(t *testing.T, spn span.Span, info string) {
	//TODO: hovering not supported on command line
}

func (r *runner) InlayHints(t *testing.T, spn span.Span) {
	// TODO: inlayHints not supported on command line
}

func (r *runner) runGoplsCmd(t testing.TB, args ...string) (string, string) {
	rStdout, wStdout, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	oldStdout := os.Stdout
	rStderr, wStderr, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	oldStderr := os.Stderr
	stdout, stderr := &bytes.Buffer{}, &bytes.Buffer{}
	var wg sync.WaitGroup
	wg.Add(2)
	go func() {
		io.Copy(stdout, rStdout)
		wg.Done()
	}()
	go func() {
		io.Copy(stderr, rStderr)
		wg.Done()
	}()
	os.Stdout, os.Stderr = wStdout, wStderr
	app := cmd.New("gopls-test", r.data.Config.Dir, r.data.Exported.Config.Env, r.options)
	remote := r.remote
	s := flag.NewFlagSet(app.Name(), flag.ExitOnError)
	err = tool.Run(tests.Context(t), s,
		app,
		append([]string{fmt.Sprintf("-remote=internal@%s", remote)}, args...))
	if err != nil {
		fmt.Fprint(os.Stderr, err)
	}
	wStdout.Close()
	wStderr.Close()
	wg.Wait()
	os.Stdout, os.Stderr = oldStdout, oldStderr
	rStdout.Close()
	rStderr.Close()
	return stdout.String(), stderr.String()
}

// NormalizeGoplsCmd runs the gopls command and normalizes its output.
func (r *runner) NormalizeGoplsCmd(t testing.TB, args ...string) (string, string) {
	stdout, stderr := r.runGoplsCmd(t, args...)
	return r.Normalize(stdout), r.Normalize(stderr)
}

func (r *runner) Normalize(s string) string {
	return tests.Normalize(s, r.normalizers)
}

func (r *runner) NormalizePrefix(s string) string {
	return tests.NormalizePrefix(s, r.normalizers)
}
