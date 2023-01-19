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

// TestCommandLine runs the marker tests in files beneath testdata/ using
// implementations of each of the marker operations (e.g. @hover) that
// call the main function of the gopls command within this process.
func TestCommandLine(t *testing.T, testdata string, options func(*source.Options)) {
	// On Android, the testdata directory is not copied to the runner.
	if runtime.GOOS == "android" {
		t.Skip("testdata directory not present on android")
	}
	tests.RunTests(t, testdata, false, func(t *testing.T, datum *tests.Data) {
		ctx := tests.Context(t)
		ts := newTestServer(ctx, options)
		tests.Run(t, newRunner(datum, ctx, ts.Addr, options), datum)
		cmd.CloseTestConnections(ctx)
	})
}

func newTestServer(ctx context.Context, options func(*source.Options)) *servertest.TCPServer {
	ctx = debug.WithInstance(ctx, "", "")
	cache := cache.New(nil, nil)
	ss := lsprpc.NewStreamServer(cache, false, options)
	return servertest.NewTCPServer(ctx, ss, nil)
}

// runner implements tests.Tests by invoking the gopls command.
//
// TODO(golang/go#54845): We don't plan to implement all the methods
// of tests.Test.  Indeed, we'd like to delete the methods that are
// implemented because the two problems they solve are best addressed
// in other ways:
//
//  1. They provide coverage of the behavior of the server, but this
//     coverage is almost identical to the coverage provided by
//     executing the same tests by making LSP RPCs directly (see
//     lsp_test), and the latter is more efficient. When they do
//     differ, it is a pain for maintainers.
//
//  2. They provide coverage of the client-side code of the
//     command-line tool, which turns arguments into an LSP request and
//     prints the results. But this coverage could be more directly and
//     efficiently achieved by running a small number of tests tailored
//     to exercise the client-side code, not the server behavior.
//
// Once that's done, tests.Tests would have only a single
// implementation (LSP), and we could refactor the marker tests
// so that they more closely resemble self-contained regtests,
// as described in #54845.
type runner struct {
	data        *tests.Data
	ctx         context.Context
	options     func(*source.Options)
	normalizers []tests.Normalizer
	remote      string
}

func newRunner(data *tests.Data, ctx context.Context, remote string, options func(*source.Options)) *runner {
	return &runner{
		data:        data,
		ctx:         ctx,
		options:     options,
		normalizers: tests.CollectNormalizers(data.Exported),
		remote:      remote,
	}
}

// runGoplsCmd returns the stdout and stderr of a gopls command.
//
// It does not fork+exec gopls, but in effect calls its main function,
// and thus the subcommand's Application.Run method, within this process.
//
// Stdout and stderr are temporarily redirected: not concurrency-safe!
//
// The "exit code" is printed to stderr but not returned.
// Invalid flags cause process exit.
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

// Unimplemented methods of tests.Tests (see comment at runner):

func (*runner) CodeLens(t *testing.T, uri span.URI, want []protocol.CodeLens) {}
func (*runner) Completion(t *testing.T, src span.Span, test tests.Completion, items tests.CompletionItems) {
}
func (*runner) CompletionSnippet(t *testing.T, src span.Span, expected tests.CompletionSnippet, placeholders bool, items tests.CompletionItems) {
}
func (*runner) UnimportedCompletion(t *testing.T, src span.Span, test tests.Completion, items tests.CompletionItems) {
}
func (*runner) DeepCompletion(t *testing.T, src span.Span, test tests.Completion, items tests.CompletionItems) {
}
func (*runner) FuzzyCompletion(t *testing.T, src span.Span, test tests.Completion, items tests.CompletionItems) {
}
func (*runner) CaseSensitiveCompletion(t *testing.T, src span.Span, test tests.Completion, items tests.CompletionItems) {
}
func (*runner) RankCompletion(t *testing.T, src span.Span, test tests.Completion, items tests.CompletionItems) {
}
func (*runner) FunctionExtraction(t *testing.T, start span.Span, end span.Span) {}
func (*runner) MethodExtraction(t *testing.T, start span.Span, end span.Span)   {}
func (*runner) AddImport(t *testing.T, uri span.URI, expectedImport string)     {}
func (*runner) Hover(t *testing.T, spn span.Span, info string)                  {}
func (*runner) InlayHints(t *testing.T, spn span.Span)                          {}
func (*runner) SelectionRanges(t *testing.T, spn span.Span)                     {}
