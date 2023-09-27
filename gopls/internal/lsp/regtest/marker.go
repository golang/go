// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package regtest

import (
	"bytes"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"go/token"
	"go/types"
	"io/fs"
	"log"
	"os"
	"path"
	"path/filepath"
	"reflect"
	"regexp"
	"runtime"
	"sort"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	"golang.org/x/tools/go/expect"
	"golang.org/x/tools/gopls/internal/hooks"
	"golang.org/x/tools/gopls/internal/lsp/cache"
	"golang.org/x/tools/gopls/internal/lsp/debug"
	"golang.org/x/tools/gopls/internal/lsp/fake"
	"golang.org/x/tools/gopls/internal/lsp/lsprpc"
	"golang.org/x/tools/gopls/internal/lsp/protocol"
	"golang.org/x/tools/gopls/internal/lsp/safetoken"
	"golang.org/x/tools/gopls/internal/lsp/source"
	"golang.org/x/tools/gopls/internal/lsp/tests"
	"golang.org/x/tools/gopls/internal/lsp/tests/compare"
	"golang.org/x/tools/internal/jsonrpc2"
	"golang.org/x/tools/internal/jsonrpc2/servertest"
	"golang.org/x/tools/internal/testenv"
	"golang.org/x/tools/txtar"
)

var update = flag.Bool("update", false, "if set, update test data during marker tests")

// RunMarkerTests runs "marker" tests in the given test data directory.
// (In practice: ../../regtest/marker/testdata)
//
// Use this command to run the tests:
//
//	$ go test ./gopls/internal/regtest/marker [-update]
//
// A marker test uses the '//@' marker syntax of the x/tools/go/expect package
// to annotate source code with various information such as locations and
// arguments of LSP operations to be executed by the test. The syntax following
// '@' is parsed as a comma-separated list of ordinary Go function calls, for
// example
//
//	//@foo(a, "b", 3),bar(0)
//
// and delegates to a corresponding function to perform LSP-related operations.
// See the Marker types documentation below for a list of supported markers.
//
// Each call argument is converted to the type of the corresponding parameter of
// the designated function. The conversion logic may use the surrounding context,
// such as the position or nearby text. See the Argument conversion section below
// for the full set of special conversions. As a special case, the blank
// identifier '_' is treated as the zero value of the parameter type.
//
// The test runner collects test cases by searching the given directory for
// files with the .txt extension. Each file is interpreted as a txtar archive,
// which is extracted to a temporary directory. The relative path to the .txt
// file is used as the subtest name. The preliminary section of the file
// (before the first archive entry) is a free-form comment.
//
// These tests were inspired by (and in many places copied from) a previous
// iteration of the marker tests built on top of the packagestest framework.
// Key design decisions motivating this reimplementation are as follows:
//   - The old tests had a single global session, causing interaction at a
//     distance and several awkward workarounds.
//   - The old tests could not be safely parallelized, because certain tests
//     manipulated the server options
//   - Relatedly, the old tests did not have a logic grouping of assertions into
//     a single unit, resulting in clusters of files serving clusters of
//     entangled assertions.
//   - The old tests used locations in the source as test names and as the
//     identity of golden content, meaning that a single edit could change the
//     name of an arbitrary number of subtests, and making it difficult to
//     manually edit golden content.
//   - The old tests did not hew closely to LSP concepts, resulting in, for
//     example, each marker implementation doing its own position
//     transformations, and inventing its own mechanism for configuration.
//   - The old tests had an ad-hoc session initialization process. The regtest
//     environment has had more time devoted to its initialization, and has a
//     more convenient API.
//   - The old tests lacked documentation, and often had failures that were hard
//     to understand. By starting from scratch, we can revisit these aspects.
//
// # Special files
//
// There are several types of file within the test archive that are given special
// treatment by the test runner:
//   - "skip": the presence of this file causes the test to be skipped, with
//     the file content used as the skip message.
//   - "flags": this file is treated as a whitespace-separated list of flags
//     that configure the MarkerTest instance. Supported flags:
//     -min_go=go1.18 sets the minimum Go version for the test;
//     -cgo requires that CGO_ENABLED is set and the cgo tool is available
//     -write_sumfile=a,b,c instructs the test runner to generate go.sum files
//     in these directories before running the test.
//     -skip_goos=a,b,c instructs the test runner to skip the test for the
//     listed GOOS values.
//     -ignore_extra_diags suppresses errors for unmatched diagnostics
//     TODO(rfindley): using build constraint expressions for -skip_goos would
//     be clearer.
//     TODO(rfindley): support flag values containing whitespace.
//   - "settings.json": this file is parsed as JSON, and used as the
//     session configuration (see gopls/doc/settings.md)
//   - "capabilities.json": this file is parsed as JSON client capabilities,
//     and applied as an overlay over the default editor client capabilities.
//     see https://microsoft.github.io/language-server-protocol/specifications/lsp/3.17/specification/#clientCapabilities
//     for more details.
//   - "env": this file is parsed as a list of VAR=VALUE fields specifying the
//     editor environment.
//   - Golden files: Within the archive, file names starting with '@' are
//     treated as "golden" content, and are not written to disk, but instead are
//     made available to test methods expecting an argument of type *Golden,
//     using the identifier following '@'. For example, if the first parameter of
//     Foo were of type *Golden, the test runner would convert the identifier a
//     in the call @foo(a, "b", 3) into a *Golden by collecting golden file
//     data starting with "@a/".
//   - proxy files: any file starting with proxy/ is treated as a Go proxy
//     file. If present, these files are written to a separate temporary
//     directory and GOPROXY is set to file://<proxy directory>.
//
// # Marker types
//
// Markers are of two kinds. A few are "value markers" (e.g. @item), which are
// processed in a first pass and each computes a value that may be referred to
// by name later. Most are "action markers", which are processed in a second
// pass and take some action such as testing an LSP operation; they may refer
// to values computed by value markers.
//
// The following markers are supported within marker tests:
//
//   - acceptcompletion(location, label, golden): specifies that accepting the
//     completion candidate produced at the given location with provided label
//     results in the given golden state.
//
//   - codeaction(kind, start, end, golden): specifies a codeaction to request
//     for the given range. To support multi-line ranges, the range is defined
//     to be between start.Start and end.End. The golden directory contains
//     changed file content after the code action is applied.
//
//   - codeactionerr(kind, start, end, wantError): specifies a codeaction that
//     fails with an error that matches the expectation.
//
//   - codelens(location, title): specifies that a codelens is expected at the
//     given location, with given title. Must be used in conjunction with
//     @codelenses.
//
//   - codelenses(): specifies that textDocument/codeLens should be run for the
//     current document, with results compared to the @codelens annotations in
//     the current document.
//
//   - complete(location, ...items): specifies expected completion results at
//     the given location. Must be used in conjunction with @item.
//
//   - diag(location, regexp): specifies an expected diagnostic matching the
//     given regexp at the given location. The test runner requires
//     a 1:1 correspondence between observed diagnostics and diag annotations.
//     The diagnostics source and kind fields are ignored, to reduce fuss.
//
//     The specified location must match the start position of the diagnostic,
//     but end positions are ignored.
//
//     TODO(adonovan): in the older marker framework, the annotation asserted
//     two additional fields (source="compiler", kind="error"). Restore them?
//
//   - def(src, dst location): perform a textDocument/definition request at
//     the src location, and check the result points to the dst location.
//
//   - foldingrange(golden): perform a textDocument/foldingRange for the
//     current document, and compare with the golden content, which is the
//     original source annotated with numbered tags delimiting the resulting
//     ranges (e.g. <1 kind="..."> ... </1>).
//
//   - format(golden): perform a textDocument/format request for the enclosing
//     file, and compare against the named golden file. If the formatting
//     request succeeds, the golden file must contain the resulting formatted
//     source. If the formatting request fails, the golden file must contain
//     the error message.
//
//   - highlight(src location, dsts ...location): makes a
//     textDocument/highlight request at the given src location, which should
//     highlight the provided dst locations.
//
//   - hover(src, dst location, g Golden): perform a textDocument/hover at the
//     src location, and checks that the result is the dst location, with hover
//     content matching "hover.md" in the golden data g.
//
//   - implementations(src location, want ...location): makes a
//     textDocument/implementation query at the src location and
//     checks that the resulting set of locations matches want.
//
//   - item(label, details, kind): defines a completion item with the provided
//     fields. This information is not positional, and therefore @item markers
//     may occur anywhere in the source. Used in conjunction with @complete,
//     snippet, or rank.
//
//     TODO(rfindley): rethink whether floating @item annotations are the best
//     way to specify completion results.
//
//   - loc(name, location): specifies the name for a location in the source. These
//     locations may be referenced by other markers.
//
//   - rename(location, new, golden): specifies a renaming of the
//     identifier at the specified location to the new name.
//     The golden directory contains the transformed files.
//
//   - renameerr(location, new, wantError): specifies a renaming that
//     fails with an error that matches the expectation.
//
//   - suggestedfix(location, regexp, kind, golden): like diag, the location and
//     regexp identify an expected diagnostic. This diagnostic must
//     to have exactly one associated code action of the specified kind.
//     This action is executed for its editing effects on the source files.
//     Like rename, the golden directory contains the expected transformed files.
//
//   - rank(location, ...completionItem): executes a textDocument/completion
//     request at the given location, and verifies that each expected
//     completion item occurs in the results, in the expected order. Other
//     unexpected completion items may occur in the results.
//     TODO(rfindley): this exists for compatibility with the old marker tests.
//     Replace this with rankl, and rename.
//
//   - rankl(location, ...label): like rank, but only cares about completion
//     item labels.
//
//   - refs(location, want ...location): executes a textDocument/references
//     request at the first location and asserts that the result is the set of
//     'want' locations. The first want location must be the declaration
//     (assumedly unique).
//
//   - snippet(location, completionItem, snippet): executes a
//     textDocument/completion request at the location, and searches for a
//     result with label matching that of the provided completion item
//     (TODO(rfindley): accept a label rather than a completion item). Check
//     the the result snippet matches the provided snippet.
//
//   - symbol(golden): makes a textDocument/documentSymbol request
//     for the enclosing file, formats the response with one symbol
//     per line, sorts it, and compares against the named golden file.
//     Each line is of the form:
//
//     dotted.symbol.name kind "detail" +n lines
//
//     where the "+n lines" part indicates that the declaration spans
//     several lines. The test otherwise makes no attempt to check
//     location information. There is no point to using more than one
//     @symbol marker in a given file.
//
//   - workspacesymbol(query, golden): makes a workspace/symbol request for the
//     given query, formats the response with one symbol per line, and compares
//     against the named golden file. As workspace symbols are by definition a
//     workspace-wide request, the location of the workspace symbol marker does
//     not matter. Each line is of the form:
//
//     location name kind
//
// # Argument conversion
//
// Marker arguments are first parsed by the go/expect package, which accepts
// the following tokens as defined by the Go spec:
//   - string, int64, float64, and rune literals
//   - true and false
//   - nil
//   - identifiers (type expect.Identifier)
//   - regular expressions, denoted the two tokens re"abc" (type *regexp.Regexp)
//
// These values are passed as arguments to the corresponding parameter of the
// test function. Additional value conversions may occur for these argument ->
// parameter type pairs:
//   - string->regexp: the argument is parsed as a regular expressions.
//   - string->location: the argument is converted to the location of the first
//     instance of the argument in the partial line preceding the note.
//   - regexp->location: the argument is converted to the location of the first
//     match for the argument in the partial line preceding the note. If the
//     regular expression contains exactly one subgroup, the position of the
//     subgroup is used rather than the position of the submatch.
//   - name->location: the argument is replaced by the named location.
//   - name->Golden: the argument is used to look up golden content prefixed by
//     @<argument>.
//   - {string,regexp,identifier}->wantError: a wantError type specifies
//     an expected error message, either in the form of a substring that
//     must be present, a regular expression that it must match, or an
//     identifier (e.g. foo) such that the archive entry @foo
//     exists and contains the exact expected error.
//
// # Example
//
// Here is a complete example:
//
//	-- a.go --
//	package a
//
//	const abc = 0x2a //@hover("b", "abc", abc),hover(" =", "abc", abc)
//	-- @abc/hover.md --
//	```go
//	const abc untyped int = 42
//	```
//
//	@hover("b", "abc", abc),hover(" =", "abc", abc)
//
// In this example, the @hover annotation tells the test runner to run the
// hoverMarker function, which has parameters:
//
//	(mark marker, src, dsc protocol.Location, g *Golden).
//
// The first argument holds the test context, including fake editor with open
// files, and sandboxed directory.
//
// Argument converters translate the "b" and "abc" arguments into locations by
// interpreting each one as a regular expression and finding the location of
// its first match on the preceding portion of the line, and the abc identifier
// into a dictionary of golden content containing "hover.md". Then the
// hoverMarker method executes a textDocument/hover LSP request at the src
// position, and ensures the result spans "abc", with the markdown content from
// hover.md. (Note that the markdown content includes the expect annotation as
// the doc comment.)
//
// The next hover on the same line asserts the same result, but initiates the
// hover immediately after "abc" in the source. This tests that we find the
// preceding identifier when hovering.
//
// # Updating golden files
//
// To update golden content in the test archive, it is easier to regenerate
// content automatically rather than edit it by hand. To do this, run the
// tests with the -update flag. Only tests that actually run will be updated.
//
// In some cases, golden content will vary by Go version (for example, gopls
// produces different markdown at Go versions before the 1.19 go/doc update).
// By convention, the golden content in test archives should match the output
// at Go tip. Each test function can normalize golden content for older Go
// versions.
//
// Note that -update does not cause missing @diag or @loc markers to be added.
//
// # TODO
//
// This API is a work-in-progress, as we migrate existing marker tests from
// internal/lsp/tests.
//
// Remaining TODO:
//   - optimize test execution
//   - reorganize regtest packages (and rename to just 'test'?)
//   - Rename the files .txtar.
//   - Provide some means by which locations in the standard library
//     (or builtin.go) can be named, so that, for example, we can we
//     can assert that MyError implements the built-in error type.
//
// Existing marker tests (in ../testdata) to port:
//   - CallHierarchy
//   - Diagnostics
//   - CompletionItems
//   - Completions
//   - CompletionSnippets
//   - DeepCompletions
//   - FuzzyCompletions
//   - CaseSensitiveCompletions
//   - RankCompletions
//   - Formats
//   - Imports
//   - SemanticTokens
//   - FunctionExtractions
//   - MethodExtractions
//   - Renames
//   - PrepareRenames
//   - InlayHints
//   - WorkspaceSymbols
//   - Signatures
//   - Links
//   - AddImport
//   - SelectionRanges
func RunMarkerTests(t *testing.T, dir string) {
	// The marker tests must be able to run go/packages.Load.
	testenv.NeedsGoPackages(t)

	tests, err := loadMarkerTests(dir)
	if err != nil {
		t.Fatal(err)
	}

	// Opt: use a shared cache.
	// TODO(rfindley): opt: use a memoize store with no eviction.
	cache := cache.New(nil)

	for _, test := range tests {
		test := test
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			if test.skipReason != "" {
				t.Skip(test.skipReason)
			}
			for _, goos := range test.skipGOOS {
				if runtime.GOOS == goos {
					t.Skipf("skipping on %s due to -skip_goos", runtime.GOOS)
				}
			}

			// TODO(rfindley): it may be more useful to have full support for build
			// constraints.
			if test.minGoVersion != "" {
				var go1point int
				if _, err := fmt.Sscanf(test.minGoVersion, "go1.%d", &go1point); err != nil {
					t.Fatalf("parsing -min_go version: %v", err)
				}
				testenv.NeedsGo1Point(t, go1point)
			}
			if test.cgo {
				testenv.NeedsTool(t, "cgo")
			}
			config := fake.EditorConfig{
				Settings:         test.settings,
				CapabilitiesJSON: test.capabilities,
				Env:              test.env,
			}
			if _, ok := config.Settings["diagnosticsDelay"]; !ok {
				if config.Settings == nil {
					config.Settings = make(map[string]any)
				}
				config.Settings["diagnosticsDelay"] = "10ms"
			}
			// inv: config.Settings != nil

			run := &markerTestRun{
				test:       test,
				env:        newEnv(t, cache, test.files, test.proxyFiles, test.writeGoSum, config),
				settings:   config.Settings,
				values:     make(map[expect.Identifier]any),
				diags:      make(map[protocol.Location][]protocol.Diagnostic),
				extraNotes: make(map[protocol.DocumentURI]map[string][]*expect.Note),
			}
			// TODO(rfindley): make it easier to clean up the regtest environment.
			defer run.env.Editor.Shutdown(context.Background()) // ignore error
			defer run.env.Sandbox.Close()                       // ignore error

			// Open all files so that we operate consistently with LSP clients, and
			// (pragmatically) so that we have a Mapper available via the fake
			// editor.
			//
			// This also allows avoiding mutating the editor state in tests.
			for file := range test.files {
				run.env.OpenFile(file)
			}
			// Wait for the didOpen notifications to be processed, then collect
			// diagnostics.
			var diags map[string]*protocol.PublishDiagnosticsParams
			run.env.AfterChange(ReadAllDiagnostics(&diags))
			for path, params := range diags {
				uri := run.env.Sandbox.Workdir.URI(path)
				for _, diag := range params.Diagnostics {
					loc := protocol.Location{
						URI: uri,
						Range: protocol.Range{
							Start: diag.Range.Start,
							End:   diag.Range.Start, // ignore end positions
						},
					}
					run.diags[loc] = append(run.diags[loc], diag)
				}
			}

			var markers []marker
			for _, note := range test.notes {
				mark := marker{run: run, note: note}
				if fn, ok := valueMarkerFuncs[note.Name]; ok {
					fn(mark)
				} else if _, ok := actionMarkerFuncs[note.Name]; ok {
					markers = append(markers, mark) // save for later
				} else {
					uri := mark.uri()
					if run.extraNotes[uri] == nil {
						run.extraNotes[uri] = make(map[string][]*expect.Note)
					}
					run.extraNotes[uri][note.Name] = append(run.extraNotes[uri][note.Name], note)
				}
			}

			// Invoke each remaining marker in the test.
			for _, mark := range markers {
				actionMarkerFuncs[mark.note.Name](mark)
			}

			// Any remaining (un-eliminated) diagnostics are an error.
			if !test.ignoreExtraDiags {
				for loc, diags := range run.diags {
					for _, diag := range diags {
						t.Errorf("%s: unexpected diagnostic: %q", run.fmtLoc(loc), diag.Message)
					}
				}
			}

			// TODO(rfindley): use these for whole-file marker tests.
			for uri, extras := range run.extraNotes {
				for name, extra := range extras {
					if len(extra) > 0 {
						t.Errorf("%s: %d unused %q markers", run.env.Sandbox.Workdir.URIToPath(uri), len(extra), name)
					}
				}
			}

			formatted, err := formatTest(test)
			if err != nil {
				t.Errorf("formatTest: %v", err)
			} else if *update {
				filename := filepath.Join(dir, test.name)
				if err := os.WriteFile(filename, formatted, 0644); err != nil {
					t.Error(err)
				}
			} else {
				// On go 1.19 and later, verify that the testdata has not changed.
				//
				// On earlier Go versions, the golden test data varies due to different
				// markdown escaping.
				//
				// Only check this if the test hasn't already failed, otherwise we'd
				// report duplicate mismatches of golden data.
				if testenv.Go1Point() >= 19 && !t.Failed() {
					// Otherwise, verify that formatted content matches.
					if diff := compare.NamedText("formatted", "on-disk", string(formatted), string(test.content)); diff != "" {
						t.Errorf("formatted test does not match on-disk content:\n%s", diff)
					}
				}
			}
		})
	}

	if abs, err := filepath.Abs(dir); err == nil && t.Failed() {
		t.Logf("(Filenames are relative to %s.)", abs)
	}
}

// A marker holds state for the execution of a single @marker
// annotation in the source.
type marker struct {
	run  *markerTestRun
	note *expect.Note
}

// server returns the LSP server for the marker test run.
func (m marker) server() protocol.Server {
	return m.run.env.Editor.Server
}

// errorf reports an error with a prefix indicating the position of the marker note.
//
// It formats the error message using mark.sprintf.
func (mark marker) errorf(format string, args ...any) {
	msg := mark.sprintf(format, args...)
	// TODO(adonovan): consider using fmt.Fprintf(os.Stderr)+t.Fail instead of
	// t.Errorf to avoid reporting uninteresting positions in the Go source of
	// the driver. However, this loses the order of stderr wrt "FAIL: TestFoo"
	// subtest dividers.
	mark.run.env.T.Errorf("%s: %s", mark.run.fmtPos(mark.note.Pos), msg)
}

// valueMarkerFunc returns a wrapper around a function that allows it to be
// called during the processing of value markers (e.g. @value(v, 123)) with marker
// arguments converted to function parameters. The provided function's first
// parameter must be of type 'marker', and it must return a value.
//
// Unlike action markers, which are executed for actions such as test
// assertions, value markers are all evaluated first, and each computes
// a value that is recorded by its identifier, which is the marker's first
// argument. These values may be referred to from an action marker by
// this identifier, e.g. @action(... , v, ...).
//
// For example, given a fn with signature
//
//	func(mark marker, label, details, kind string) CompletionItem
//
// The result of valueMarkerFunc can associated with @item notes, and invoked
// as follows:
//
//	//@item(FooCompletion, "Foo", "func() int", "func")
//
// The provided fn should not mutate the test environment.
func valueMarkerFunc(fn any) func(marker) {
	ftype := reflect.TypeOf(fn)
	if ftype.NumIn() == 0 || ftype.In(0) != markerType {
		panic(fmt.Sprintf("value marker function %#v must accept marker as its first argument", ftype))
	}
	if ftype.NumOut() != 1 {
		panic(fmt.Sprintf("value marker function %#v must have exactly 1 result", ftype))
	}

	return func(mark marker) {
		if len(mark.note.Args) == 0 || !is[expect.Identifier](mark.note.Args[0]) {
			mark.errorf("first argument to a value marker function must be an identifier")
			return
		}
		id := mark.note.Args[0].(expect.Identifier)
		if alt, ok := mark.run.values[id]; ok {
			mark.errorf("%s already declared as %T", id, alt)
			return
		}
		args := append([]any{mark}, mark.note.Args[1:]...)
		argValues, err := convertArgs(mark, ftype, args)
		if err != nil {
			mark.errorf("converting args: %v", err)
			return
		}
		results := reflect.ValueOf(fn).Call(argValues)
		mark.run.values[id] = results[0].Interface()
	}
}

// actionMarkerFunc returns a wrapper around a function that allows it to be
// called during the processing of action markers (e.g. @action("abc", 123))
// with marker arguments converted to function parameters. The provided
// function's first parameter must be of type 'marker', and it must not return
// any values.
//
// The provided fn should not mutate the test environment.
func actionMarkerFunc(fn any) func(marker) {
	ftype := reflect.TypeOf(fn)
	if ftype.NumIn() == 0 || ftype.In(0) != markerType {
		panic(fmt.Sprintf("action marker function %#v must accept marker as its first argument", ftype))
	}
	if ftype.NumOut() != 0 {
		panic(fmt.Sprintf("action marker function %#v cannot have results", ftype))
	}

	return func(mark marker) {
		args := append([]any{mark}, mark.note.Args...)
		argValues, err := convertArgs(mark, ftype, args)
		if err != nil {
			mark.errorf("converting args: %v", err)
			return
		}
		reflect.ValueOf(fn).Call(argValues)
	}
}

func convertArgs(mark marker, ftype reflect.Type, args []any) ([]reflect.Value, error) {
	var (
		argValues []reflect.Value
		pnext     int          // next param index
		p         reflect.Type // current param
	)
	for i, arg := range args {
		if i < ftype.NumIn() {
			p = ftype.In(pnext)
			pnext++
		} else if p == nil || !ftype.IsVariadic() {
			// The actual number of arguments expected by the mark varies, depending
			// on whether this is a value marker or an action marker.
			//
			// Since this error indicates a bug, probably OK to have an imprecise
			// error message here.
			return nil, fmt.Errorf("too many arguments to %s", mark.note.Name)
		}
		elemType := p
		if ftype.IsVariadic() && pnext == ftype.NumIn() {
			elemType = p.Elem()
		}
		var v reflect.Value
		if id, ok := arg.(expect.Identifier); ok && id == "_" {
			v = reflect.Zero(elemType)
		} else {
			a, err := convert(mark, arg, elemType)
			if err != nil {
				return nil, err
			}
			v = reflect.ValueOf(a)
		}
		argValues = append(argValues, v)
	}
	// Check that we have sufficient arguments. If the function is variadic, we
	// do not need arguments for the final parameter.
	if pnext < ftype.NumIn()-1 || pnext == ftype.NumIn()-1 && !ftype.IsVariadic() {
		// Same comment as above: OK to be vague here.
		return nil, fmt.Errorf("not enough arguments to %s", mark.note.Name)
	}
	return argValues, nil
}

// is reports whether arg is a T.
func is[T any](arg any) bool {
	_, ok := arg.(T)
	return ok
}

// Supported value marker functions. See [valueMarkerFunc] for more details.
var valueMarkerFuncs = map[string]func(marker){
	"loc":  valueMarkerFunc(locMarker),
	"item": valueMarkerFunc(completionItemMarker),
}

// Supported action marker functions. See [actionMarkerFunc] for more details.
var actionMarkerFuncs = map[string]func(marker){
	"acceptcompletion": actionMarkerFunc(acceptCompletionMarker),
	"codeaction":       actionMarkerFunc(codeActionMarker),
	"codeactionerr":    actionMarkerFunc(codeActionErrMarker),
	"codelenses":       actionMarkerFunc(codeLensesMarker),
	"complete":         actionMarkerFunc(completeMarker),
	"def":              actionMarkerFunc(defMarker),
	"diag":             actionMarkerFunc(diagMarker),
	"foldingrange":     actionMarkerFunc(foldingRangeMarker),
	"format":           actionMarkerFunc(formatMarker),
	"highlight":        actionMarkerFunc(highlightMarker),
	"hover":            actionMarkerFunc(hoverMarker),
	"implementation":   actionMarkerFunc(implementationMarker),
	"rank":             actionMarkerFunc(rankMarker),
	"rankl":            actionMarkerFunc(ranklMarker),
	"refs":             actionMarkerFunc(refsMarker),
	"rename":           actionMarkerFunc(renameMarker),
	"renameerr":        actionMarkerFunc(renameErrMarker),
	"signature":        actionMarkerFunc(signatureMarker),
	"snippet":          actionMarkerFunc(snippetMarker),
	"suggestedfix":     actionMarkerFunc(suggestedfixMarker),
	"symbol":           actionMarkerFunc(symbolMarker),
	"typedef":          actionMarkerFunc(typedefMarker),
	"workspacesymbol":  actionMarkerFunc(workspaceSymbolMarker),
}

// markerTest holds all the test data extracted from a test txtar archive.
//
// See the documentation for RunMarkerTests for more information on the archive
// format.
type markerTest struct {
	name         string                        // relative path to the txtar file in the testdata dir
	fset         *token.FileSet                // fileset used for parsing notes
	content      []byte                        // raw test content
	archive      *txtar.Archive                // original test archive
	settings     map[string]any                // gopls settings
	capabilities []byte                        // content of capabilities.json file
	env          map[string]string             // editor environment
	proxyFiles   map[string][]byte             // proxy content
	files        map[string][]byte             // data files from the archive (excluding special files)
	notes        []*expect.Note                // extracted notes from data files
	golden       map[expect.Identifier]*Golden // extracted golden content, by identifier name

	skipReason string   // the skip reason extracted from the "skip" archive file
	flags      []string // flags extracted from the special "flags" archive file.

	// Parsed flags values.
	minGoVersion     string
	cgo              bool
	writeGoSum       []string // comma separated dirs to write go sum for
	skipGOOS         []string // comma separated GOOS values to skip
	ignoreExtraDiags bool
}

// flagSet returns the flagset used for parsing the special "flags" file in the
// test archive.
func (t *markerTest) flagSet() *flag.FlagSet {
	flags := flag.NewFlagSet(t.name, flag.ContinueOnError)
	flags.StringVar(&t.minGoVersion, "min_go", "", "if set, the minimum go1.X version required for this test")
	flags.BoolVar(&t.cgo, "cgo", false, "if set, requires cgo (both the cgo tool and CGO_ENABLED=1)")
	flags.Var((*stringListValue)(&t.writeGoSum), "write_sumfile", "if set, write the sumfile for these directories")
	flags.Var((*stringListValue)(&t.skipGOOS), "skip_goos", "if set, skip this test on these GOOS values")
	flags.BoolVar(&t.ignoreExtraDiags, "ignore_extra_diags", false, "if set, suppress errors for unmatched diagnostics")
	return flags
}

// stringListValue implements flag.Value.
type stringListValue []string

func (l *stringListValue) Set(s string) error {
	if s != "" {
		for _, d := range strings.Split(s, ",") {
			*l = append(*l, strings.TrimSpace(d))
		}
	}
	return nil
}

func (l stringListValue) String() string {
	return strings.Join([]string(l), ",")
}

func (t *markerTest) getGolden(id expect.Identifier) *Golden {
	golden, ok := t.golden[id]
	// If there was no golden content for this identifier, we must create one
	// to handle the case where -update is set: we need a place to store
	// the updated content.
	if !ok {
		golden = &Golden{id: id}

		// TODO(adonovan): the separation of markerTest (the
		// static aspects) from markerTestRun (the dynamic
		// ones) is evidently bogus because here we modify
		// markerTest during execution. Let's merge the two.
		t.golden[id] = golden
	}
	return golden
}

// Golden holds extracted golden content for a single @<name> prefix.
//
// When -update is set, golden captures the updated golden contents for later
// writing.
type Golden struct {
	id      expect.Identifier
	data    map[string][]byte // key "" => @id itself
	updated map[string][]byte
}

// Get returns golden content for the given name, which corresponds to the
// relative path following the golden prefix @<name>/. For example, to access
// the content of @foo/path/to/result.json from the Golden associated with
// @foo, name should be "path/to/result.json".
//
// If -update is set, the given update function will be called to get the
// updated golden content that should be written back to testdata.
//
// Marker functions must use this method instead of accessing data entries
// directly otherwise the -update operation will delete those entries.
//
// TODO(rfindley): rethink the logic here. We may want to separate Get and Set,
// and not delete golden content that isn't set.
func (g *Golden) Get(t testing.TB, name string, updated []byte) ([]byte, bool) {
	if existing, ok := g.updated[name]; ok {
		// Multiple tests may reference the same golden data, but if they do they
		// must agree about its expected content.
		if diff := compare.NamedText("existing", "updated", string(existing), string(updated)); diff != "" {
			t.Errorf("conflicting updates for golden data %s/%s:\n%s", g.id, name, diff)
		}
	}
	if g.updated == nil {
		g.updated = make(map[string][]byte)
	}
	g.updated[name] = updated
	if *update {
		return updated, true
	}

	res, ok := g.data[name]
	return res, ok
}

// loadMarkerTests walks the given dir looking for .txt files, which it
// interprets as a txtar archive.
//
// See the documentation for RunMarkerTests for more details on the test data
// archive.
func loadMarkerTests(dir string) ([]*markerTest, error) {
	var tests []*markerTest
	err := filepath.WalkDir(dir, func(path string, d fs.DirEntry, err error) error {
		if strings.HasSuffix(path, ".txt") {
			content, err := os.ReadFile(path)
			if err != nil {
				return err
			}

			name := strings.TrimPrefix(path, dir+string(filepath.Separator))
			test, err := loadMarkerTest(name, content)
			if err != nil {
				return fmt.Errorf("%s: %v", path, err)
			}
			tests = append(tests, test)
		}
		return nil
	})
	return tests, err
}

func loadMarkerTest(name string, content []byte) (*markerTest, error) {
	archive := txtar.Parse(content)
	if len(archive.Files) == 0 {
		return nil, fmt.Errorf("txtar file has no '-- filename --' sections")
	}
	if bytes.Contains(archive.Comment, []byte("\n-- ")) {
		// This check is conservative, but the comment is only a comment.
		return nil, fmt.Errorf("ill-formed '-- filename --' header in comment")
	}
	test := &markerTest{
		name:    name,
		fset:    token.NewFileSet(),
		content: content,
		archive: archive,
		files:   make(map[string][]byte),
		golden:  make(map[expect.Identifier]*Golden),
	}
	for _, file := range archive.Files {
		switch {
		case file.Name == "skip":
			reason := strings.ReplaceAll(string(file.Data), "\n", " ")
			reason = strings.TrimSpace(reason)
			test.skipReason = reason

		case file.Name == "flags":
			test.flags = strings.Fields(string(file.Data))
			if err := test.flagSet().Parse(test.flags); err != nil {
				return nil, fmt.Errorf("parsing flags: %v", err)
			}

		case file.Name == "settings.json":
			if err := json.Unmarshal(file.Data, &test.settings); err != nil {
				return nil, err
			}

		case file.Name == "capabilities.json":
			test.capabilities = file.Data // lazily unmarshalled by the editor

		case file.Name == "env":
			test.env = make(map[string]string)
			fields := strings.Fields(string(file.Data))
			for _, field := range fields {
				key, value, ok := strings.Cut(field, "=")
				if !ok {
					return nil, fmt.Errorf("env vars must be formatted as var=value, got %q", field)
				}
				test.env[key] = value
			}

		case strings.HasPrefix(file.Name, "@"): // golden content
			idstring, name, _ := strings.Cut(file.Name[len("@"):], "/")
			id := expect.Identifier(idstring)
			// Note that a file.Name of just "@id" gives (id, name) = ("id", "").
			if _, ok := test.golden[id]; !ok {
				test.golden[id] = &Golden{
					id:   id,
					data: make(map[string][]byte),
				}
			}
			test.golden[id].data[name] = file.Data

		case strings.HasPrefix(file.Name, "proxy/"):
			name := file.Name[len("proxy/"):]
			if test.proxyFiles == nil {
				test.proxyFiles = make(map[string][]byte)
			}
			test.proxyFiles[name] = file.Data

		default: // ordinary file content
			notes, err := expect.Parse(test.fset, file.Name, file.Data)
			if err != nil {
				return nil, fmt.Errorf("parsing notes in %q: %v", file.Name, err)
			}

			// Reject common misspelling: "// @mark".
			// TODO(adonovan): permit "// @" within a string. Detect multiple spaces.
			if i := bytes.Index(file.Data, []byte("// @")); i >= 0 {
				line := 1 + bytes.Count(file.Data[:i], []byte("\n"))
				return nil, fmt.Errorf("%s:%d: unwanted space before marker (// @)", file.Name, line)
			}

			test.notes = append(test.notes, notes...)
			test.files[file.Name] = file.Data
		}

		// Print a warning if we see what looks like "-- filename --"
		// without the second "--". It's not necessarily wrong,
		// but it should almost never appear in our test inputs.
		if bytes.Contains(file.Data, []byte("\n-- ")) {
			log.Printf("ill-formed '-- filename --' header in %s?", file.Name)
		}
	}

	return test, nil
}

// formatTest formats the test as a txtar archive.
func formatTest(test *markerTest) ([]byte, error) {
	arch := &txtar.Archive{
		Comment: test.archive.Comment,
	}

	updatedGolden := make(map[string][]byte)
	for id, g := range test.golden {
		for name, data := range g.updated {
			filename := "@" + path.Join(string(id), name) // name may be ""
			updatedGolden[filename] = data
		}
	}

	// Preserve the original ordering of archive files.
	for _, file := range test.archive.Files {
		switch file.Name {
		// Preserve configuration files exactly as they were. They must have parsed
		// if we got this far.
		case "skip", "flags", "settings.json", "capabilities.json", "env":
			arch.Files = append(arch.Files, file)
		default:
			if _, ok := test.files[file.Name]; ok { // ordinary file
				arch.Files = append(arch.Files, file)
			} else if strings.HasPrefix(file.Name, "proxy/") { // proxy file
				arch.Files = append(arch.Files, file)
			} else if data, ok := updatedGolden[file.Name]; ok { // golden file
				arch.Files = append(arch.Files, txtar.File{Name: file.Name, Data: data})
				delete(updatedGolden, file.Name)
			}
		}
	}

	// ...followed by any new golden files.
	var newGoldenFiles []txtar.File
	for filename, data := range updatedGolden {
		newGoldenFiles = append(newGoldenFiles, txtar.File{Name: filename, Data: data})
	}
	// Sort new golden files lexically.
	sort.Slice(newGoldenFiles, func(i, j int) bool {
		return newGoldenFiles[i].Name < newGoldenFiles[j].Name
	})
	arch.Files = append(arch.Files, newGoldenFiles...)

	return txtar.Format(arch), nil
}

// newEnv creates a new environment for a marker test.
//
// TODO(rfindley): simplify and refactor the construction of testing
// environments across regtests, marker tests, and benchmarks.
func newEnv(t *testing.T, cache *cache.Cache, files, proxyFiles map[string][]byte, writeGoSum []string, config fake.EditorConfig) *Env {
	sandbox, err := fake.NewSandbox(&fake.SandboxConfig{
		RootDir:    t.TempDir(),
		Files:      files,
		ProxyFiles: proxyFiles,
	})
	if err != nil {
		t.Fatal(err)
	}

	for _, dir := range writeGoSum {
		if err := sandbox.RunGoCommand(context.Background(), dir, "list", []string{"-mod=mod", "..."}, []string{"GOWORK=off"}, true); err != nil {
			t.Fatal(err)
		}
	}

	// Put a debug instance in the context to prevent logging to stderr.
	// See associated TODO in runner.go: we should revisit this pattern.
	ctx := context.Background()
	ctx = debug.WithInstance(ctx, "", "off")

	awaiter := NewAwaiter(sandbox.Workdir)
	ss := lsprpc.NewStreamServer(cache, false, hooks.Options)
	server := servertest.NewPipeServer(ss, jsonrpc2.NewRawStream)
	const skipApplyEdits = true // capture edits but don't apply them
	editor, err := fake.NewEditor(sandbox, config).Connect(ctx, server, awaiter.Hooks(), skipApplyEdits)
	if err != nil {
		sandbox.Close() // ignore error
		t.Fatal(err)
	}
	if err := awaiter.Await(ctx, InitialWorkspaceLoad); err != nil {
		sandbox.Close() // ignore error
		t.Fatal(err)
	}
	return &Env{
		T:       t,
		Ctx:     ctx,
		Editor:  editor,
		Sandbox: sandbox,
		Awaiter: awaiter,
	}
}

// A markerTestRun holds the state of one run of a marker test archive.
type markerTestRun struct {
	test     *markerTest
	env      *Env
	settings map[string]any

	// Collected information.
	// Each @diag/@suggestedfix marker eliminates an entry from diags.
	values map[expect.Identifier]any
	diags  map[protocol.Location][]protocol.Diagnostic // diagnostics by position; location end == start

	// Notes that weren't associated with a top-level marker func. They may be
	// consumed by another marker (e.g. @codelenses collects @codelens markers).
	// Any notes that aren't consumed are flagged as an error.
	extraNotes map[protocol.DocumentURI]map[string][]*expect.Note
}

// sprintf returns a formatted string after applying pre-processing to
// arguments of the following types:
//   - token.Pos: formatted using (*markerTestRun).fmtPos
//   - protocol.Location: formatted using (*markerTestRun).fmtLoc
func (c *marker) sprintf(format string, args ...any) string {
	if false {
		_ = fmt.Sprintf(format, args...) // enable vet printf checker
	}
	var args2 []any
	for _, arg := range args {
		switch arg := arg.(type) {
		case token.Pos:
			args2 = append(args2, c.run.fmtPos(arg))
		case protocol.Location:
			args2 = append(args2, c.run.fmtLoc(arg))
		default:
			args2 = append(args2, arg)
		}
	}
	return fmt.Sprintf(format, args2...)
}

// uri returns the URI of the file containing the marker.
func (mark marker) uri() protocol.DocumentURI {
	return mark.run.env.Sandbox.Workdir.URI(mark.run.test.fset.File(mark.note.Pos).Name())
}

// path returns the relative path to the file containing the marker.
func (mark marker) path() string {
	return mark.run.env.Sandbox.Workdir.RelPath(mark.run.test.fset.File(mark.note.Pos).Name())
}

// fmtLoc formats the given pos in the context of the test, using
// archive-relative paths for files and including the line number in the full
// archive file.
func (run *markerTestRun) fmtPos(pos token.Pos) string {
	file := run.test.fset.File(pos)
	if file == nil {
		run.env.T.Errorf("position %d not in test fileset", pos)
		return "<invalid location>"
	}
	m, err := run.env.Editor.Mapper(file.Name())
	if err != nil {
		run.env.T.Errorf("%s", err)
		return "<invalid location>"
	}
	loc, err := m.PosLocation(file, pos, pos)
	if err != nil {
		run.env.T.Errorf("Mapper(%s).PosLocation failed: %v", file.Name(), err)
	}
	return run.fmtLoc(loc)
}

// fmtLoc formats the given location in the context of the test, using
// archive-relative paths for files and including the line number in the full
// archive file.
func (run *markerTestRun) fmtLoc(loc protocol.Location) string {
	formatted := run.fmtLocDetails(loc, true)
	if formatted == "" {
		run.env.T.Errorf("unable to find %s in test archive", loc)
		return "<invalid location>"
	}
	return formatted
}

// See fmtLoc. If includeTxtPos is not set, the position in the full archive
// file is omitted.
//
// If the location cannot be found within the archive, fmtLocDetails returns "".
func (run *markerTestRun) fmtLocDetails(loc protocol.Location, includeTxtPos bool) string {
	if loc == (protocol.Location{}) {
		return ""
	}
	lines := bytes.Count(run.test.archive.Comment, []byte("\n"))
	var name string
	for _, f := range run.test.archive.Files {
		lines++ // -- separator --
		uri := run.env.Sandbox.Workdir.URI(f.Name)
		if uri == loc.URI {
			name = f.Name
			break
		}
		lines += bytes.Count(f.Data, []byte("\n"))
	}
	if name == "" {
		return ""
	}
	m, err := run.env.Editor.Mapper(name)
	if err != nil {
		run.env.T.Errorf("internal error: %v", err)
		return "<invalid location>"
	}
	s, err := m.LocationSpan(loc)
	if err != nil {
		run.env.T.Errorf("error formatting location %s: %v", loc, err)
		return "<invalid location>"
	}

	innerSpan := fmt.Sprintf("%d:%d", s.Start().Line(), s.Start().Column())       // relative to the embedded file
	outerSpan := fmt.Sprintf("%d:%d", lines+s.Start().Line(), s.Start().Column()) // relative to the archive file
	if s.Start() != s.End() {
		if s.End().Line() == s.Start().Line() {
			innerSpan += fmt.Sprintf("-%d", s.End().Column())
			outerSpan += fmt.Sprintf("-%d", s.End().Column())
		} else {
			innerSpan += fmt.Sprintf("-%d:%d", s.End().Line(), s.End().Column())
			innerSpan += fmt.Sprintf("-%d:%d", lines+s.End().Line(), s.End().Column())
		}
	}

	if includeTxtPos {
		return fmt.Sprintf("%s:%s (%s:%s)", name, innerSpan, run.test.name, outerSpan)
	} else {
		return fmt.Sprintf("%s:%s", name, innerSpan)
	}
}

// ---- converters ----

// converter is the signature of argument converters.
// A converter should return an error rather than calling marker.errorf().
type converter func(marker, any) (any, error)

// Types with special conversions.
var (
	goldenType    = reflect.TypeOf(&Golden{})
	locationType  = reflect.TypeOf(protocol.Location{})
	markerType    = reflect.TypeOf(marker{})
	regexpType    = reflect.TypeOf(&regexp.Regexp{})
	wantErrorType = reflect.TypeOf(wantError{})
)

func convert(mark marker, arg any, paramType reflect.Type) (any, error) {
	if paramType == goldenType {
		id, ok := arg.(expect.Identifier)
		if !ok {
			return nil, fmt.Errorf("invalid input type %T: golden key must be an identifier", arg)
		}
		return mark.run.test.getGolden(id), nil
	}
	if id, ok := arg.(expect.Identifier); ok {
		if arg, ok := mark.run.values[id]; ok {
			if !reflect.TypeOf(arg).AssignableTo(paramType) {
				return nil, fmt.Errorf("cannot convert %v to %s", arg, paramType)
			}
			return arg, nil
		}
	}
	if reflect.TypeOf(arg).AssignableTo(paramType) {
		return arg, nil // no conversion required
	}
	switch paramType {
	case locationType:
		return convertLocation(mark, arg)
	case wantErrorType:
		return convertWantError(mark, arg)
	default:
		return nil, fmt.Errorf("cannot convert %v to %s", arg, paramType)
	}
}

// convertLocation converts a string or regexp argument into the protocol
// location corresponding to the first position of the string (or first match
// of the regexp) in the line preceding the note.
func convertLocation(mark marker, arg any) (protocol.Location, error) {
	switch arg := arg.(type) {
	case string:
		startOff, preceding, m, err := linePreceding(mark.run, mark.note.Pos)
		if err != nil {
			return protocol.Location{}, err
		}
		idx := bytes.Index(preceding, []byte(arg))
		if idx < 0 {
			return protocol.Location{}, fmt.Errorf("substring %q not found in %q", arg, preceding)
		}
		off := startOff + idx
		return m.OffsetLocation(off, off+len(arg))
	case *regexp.Regexp:
		return findRegexpInLine(mark.run, mark.note.Pos, arg)
	default:
		return protocol.Location{}, fmt.Errorf("cannot convert argument type %T to location (must be a string to match the preceding line)", arg)
	}
}

// findRegexpInLine searches the partial line preceding pos for a match for the
// regular expression re, returning a location spanning the first match. If re
// contains exactly one subgroup, the position of this subgroup match is
// returned rather than the position of the full match.
func findRegexpInLine(run *markerTestRun, pos token.Pos, re *regexp.Regexp) (protocol.Location, error) {
	startOff, preceding, m, err := linePreceding(run, pos)
	if err != nil {
		return protocol.Location{}, err
	}

	matches := re.FindSubmatchIndex(preceding)
	if len(matches) == 0 {
		return protocol.Location{}, fmt.Errorf("no match for regexp %q found in %q", re, string(preceding))
	}
	var start, end int
	switch len(matches) {
	case 2:
		// no subgroups: return the range of the regexp expression
		start, end = matches[0], matches[1]
	case 4:
		// one subgroup: return its range
		start, end = matches[2], matches[3]
	default:
		return protocol.Location{}, fmt.Errorf("invalid location regexp %q: expect either 0 or 1 subgroups, got %d", re, len(matches)/2-1)
	}

	return m.OffsetLocation(start+startOff, end+startOff)
}

func linePreceding(run *markerTestRun, pos token.Pos) (int, []byte, *protocol.Mapper, error) {
	file := run.test.fset.File(pos)
	posn := safetoken.Position(file, pos)
	lineStart := file.LineStart(posn.Line)
	startOff, endOff, err := safetoken.Offsets(file, lineStart, pos)
	if err != nil {
		return 0, nil, nil, err
	}
	m, err := run.env.Editor.Mapper(file.Name())
	if err != nil {
		return 0, nil, nil, err
	}
	return startOff, m.Content[startOff:endOff], m, nil
}

// convertWantError converts a string, regexp, or identifier
// argument into a wantError. The string is a substring of the
// expected error, the regexp is a pattern than matches the expected
// error, and the identifier is a golden file containing the expected
// error.
func convertWantError(mark marker, arg any) (wantError, error) {
	switch arg := arg.(type) {
	case string:
		return wantError{substr: arg}, nil
	case *regexp.Regexp:
		return wantError{pattern: arg}, nil
	case expect.Identifier:
		golden := mark.run.test.getGolden(arg)
		return wantError{golden: golden}, nil
	default:
		return wantError{}, fmt.Errorf("cannot convert %T to wantError (want: string, regexp, or identifier)", arg)
	}
}

// A wantError represents an expectation of a specific error message.
//
// It may be indicated in one of three ways, in 'expect' notation:
// - an identifier 'foo', to compare with the contents of the golden section @foo;
// - a pattern expression re"ab.*c", to match against a regular expression;
// - a string literal "abc", to check for a substring.
type wantError struct {
	golden  *Golden
	pattern *regexp.Regexp
	substr  string
}

func (we wantError) String() string {
	if we.golden != nil {
		return fmt.Sprintf("error from @%s entry", we.golden.id)
	} else if we.pattern != nil {
		return fmt.Sprintf("error matching %#q", we.pattern)
	} else {
		return fmt.Sprintf("error with substring %q", we.substr)
	}
}

// check asserts that 'err' matches the wantError's expectations.
func (we wantError) check(mark marker, err error) {
	if err == nil {
		mark.errorf("@%s succeeded unexpectedly, want %v", mark.note.Name, we)
		return
	}
	got := err.Error()

	if we.golden != nil {
		// Error message must match @id golden file.
		wantBytes, ok := we.golden.Get(mark.run.env.T, "", []byte(got))
		if !ok {
			mark.errorf("@%s: missing @%s entry", mark.note.Name, we.golden.id)
			return
		}
		want := strings.TrimSpace(string(wantBytes))
		if got != want {
			// (ignore leading/trailing space)
			mark.errorf("@%s failed with wrong error: got:\n%s\nwant:\n%s\ndiff:\n%s",
				mark.note.Name, got, want, compare.Text(want, got))
		}

	} else if we.pattern != nil {
		// Error message must match regular expression pattern.
		if !we.pattern.MatchString(got) {
			mark.errorf("got error %q, does not match pattern %#q", got, we.pattern)
		}

	} else if !strings.Contains(got, we.substr) {
		// Error message must contain expected substring.
		mark.errorf("got error %q, want substring %q", got, we.substr)
	}
}

// checkChangedFiles compares the files changed by an operation with their expected (golden) state.
func checkChangedFiles(mark marker, changed map[string][]byte, golden *Golden) {
	// Check changed files match expectations.
	for filename, got := range changed {
		if want, ok := golden.Get(mark.run.env.T, filename, got); !ok {
			mark.errorf("%s: unexpected change to file %s; got:\n%s",
				mark.note.Name, filename, got)

		} else if string(got) != string(want) {
			mark.errorf("%s: wrong file content for %s: got:\n%s\nwant:\n%s\ndiff:\n%s",
				mark.note.Name, filename, got, want,
				compare.Bytes(want, got))
		}
	}

	// Report unmet expectations.
	for filename := range golden.data {
		if _, ok := changed[filename]; !ok {
			want, _ := golden.Get(mark.run.env.T, filename, nil)
			mark.errorf("%s: missing change to file %s; want:\n%s",
				mark.note.Name, filename, want)
		}
	}
}

// ---- marker functions ----

// TODO(rfindley): consolidate documentation of these markers. They are already
// documented above, so much of the documentation here is redundant.

// completionItem is a simplified summary of a completion item.
type completionItem struct {
	Label, Detail, Kind, Documentation string
}

func completionItemMarker(mark marker, label string, other ...string) completionItem {
	if len(other) > 3 {
		mark.errorf("too many arguments to @item: expect at most 4")
	}
	item := completionItem{
		Label: label,
	}
	if len(other) > 0 {
		item.Detail = other[0]
	}
	if len(other) > 1 {
		item.Kind = other[1]
	}
	if len(other) > 2 {
		item.Documentation = other[2]
	}
	return item
}

func rankMarker(mark marker, src protocol.Location, items ...completionItem) {
	list := mark.run.env.Completion(src)
	var got []string
	// Collect results that are present in items, preserving their order.
	for _, g := range list.Items {
		for _, w := range items {
			if g.Label == w.Label {
				got = append(got, g.Label)
				break
			}
		}
	}
	var want []string
	for _, w := range items {
		want = append(want, w.Label)
	}
	if diff := cmp.Diff(want, got); diff != "" {
		mark.errorf("completion rankings do not match (-want +got):\n%s", diff)
	}
}

func ranklMarker(mark marker, src protocol.Location, labels ...string) {
	list := mark.run.env.Completion(src)
	var got []string
	// Collect results that are present in items, preserving their order.
	for _, g := range list.Items {
		for _, label := range labels {
			if g.Label == label {
				got = append(got, g.Label)
				break
			}
		}
	}
	if diff := cmp.Diff(labels, got); diff != "" {
		mark.errorf("completion rankings do not match (-want +got):\n%s", diff)
	}
}

func snippetMarker(mark marker, src protocol.Location, item completionItem, want string) {
	list := mark.run.env.Completion(src)
	var (
		found bool
		got   string
		all   []string // for errors
	)
	items := filterBuiltinsAndKeywords(list.Items)
	for _, i := range items {
		all = append(all, i.Label)
		if i.Label == item.Label {
			found = true
			if i.TextEdit != nil {
				got = i.TextEdit.NewText
			}
			break
		}
	}
	if !found {
		mark.errorf("no completion item found matching %s (got: %v)", item.Label, all)
		return
	}
	if got != want {
		mark.errorf("snippets do not match: got %q, want %q", got, want)
	}
}

// completeMarker implements the @complete marker, running
// textDocument/completion at the given src location and asserting that the
// results match the expected results.
func completeMarker(mark marker, src protocol.Location, want ...completionItem) {
	list := mark.run.env.Completion(src)
	items := filterBuiltinsAndKeywords(list.Items)
	var got []completionItem
	for i, item := range items {
		simplified := completionItem{
			Label:  item.Label,
			Detail: item.Detail,
			Kind:   fmt.Sprint(item.Kind),
		}
		if item.Documentation != nil {
			switch v := item.Documentation.Value.(type) {
			case string:
				simplified.Documentation = v
			case protocol.MarkupContent:
				simplified.Documentation = strings.TrimSpace(v.Value) // trim newlines
			}
		}
		// Support short-hand notation: if Detail, Kind, or Documentation are omitted from the
		// item, don't match them.
		if i < len(want) {
			if want[i].Detail == "" {
				simplified.Detail = ""
			}
			if want[i].Kind == "" {
				simplified.Kind = ""
			}
			if want[i].Documentation == "" {
				simplified.Documentation = ""
			}
		}
		got = append(got, simplified)
	}
	if len(want) == 0 {
		want = nil // got is nil if empty
	}
	if diff := cmp.Diff(want, got); diff != "" {
		mark.errorf("Completion(...) returned unexpect results (-want +got):\n%s", diff)
	}
}

// filterBuiltinsAndKeywords filters out builtins and keywords from completion
// results.
//
// It over-approximates, and does not detect if builtins are shadowed.
func filterBuiltinsAndKeywords(items []protocol.CompletionItem) []protocol.CompletionItem {
	keep := 0
	for _, item := range items {
		if types.Universe.Lookup(item.Label) == nil && token.Lookup(item.Label) == token.IDENT {
			items[keep] = item
			keep++
		}
	}
	return items[:keep]
}

// acceptCompletionMarker implements the @acceptCompletion marker, running
// textDocument/completion at the given src location and accepting the
// candidate with the given label. The resulting source must match the provided
// golden content.
func acceptCompletionMarker(mark marker, src protocol.Location, label string, golden *Golden) {
	list := mark.run.env.Completion(src)
	var selected *protocol.CompletionItem
	for _, item := range list.Items {
		if item.Label == label {
			selected = &item
			break
		}
	}
	if selected == nil {
		mark.errorf("Completion(...) did not return an item labeled %q", label)
		return
	}
	filename := mark.path()
	mapper, err := mark.run.env.Editor.Mapper(filename)
	if err != nil {
		mark.errorf("Editor.Mapper(%s) failed: %v", filename, err)
		return
	}

	patched, _, err := source.ApplyProtocolEdits(mapper, append([]protocol.TextEdit{
		*selected.TextEdit,
	}, selected.AdditionalTextEdits...))

	if err != nil {
		mark.errorf("ApplyProtocolEdits failed: %v", err)
		return
	}
	changes := map[string][]byte{filename: patched}
	// Check the file state.
	checkChangedFiles(mark, changes, golden)
}

// defMarker implements the @def marker, running textDocument/definition at
// the given src location and asserting that there is exactly one resulting
// location, matching dst.
//
// TODO(rfindley): support a variadic destination set.
func defMarker(mark marker, src, dst protocol.Location) {
	got := mark.run.env.GoToDefinition(src)
	if got != dst {
		mark.errorf("definition location does not match:\n\tgot: %s\n\twant %s",
			mark.run.fmtLoc(got), mark.run.fmtLoc(dst))
	}
}

func typedefMarker(mark marker, src, dst protocol.Location) {
	got := mark.run.env.TypeDefinition(src)
	if got != dst {
		mark.errorf("type definition location does not match:\n\tgot: %s\n\twant %s",
			mark.run.fmtLoc(got), mark.run.fmtLoc(dst))
	}
}

func foldingRangeMarker(mark marker, g *Golden) {
	env := mark.run.env
	ranges, err := mark.server().FoldingRange(env.Ctx, &protocol.FoldingRangeParams{
		TextDocument: protocol.TextDocumentIdentifier{URI: mark.uri()},
	})
	if err != nil {
		mark.errorf("foldingRange failed: %v", err)
		return
	}
	var edits []protocol.TextEdit
	insert := func(line, char uint32, text string) {
		pos := protocol.Position{Line: line, Character: char}
		edits = append(edits, protocol.TextEdit{
			Range: protocol.Range{
				Start: pos,
				End:   pos,
			},
			NewText: text,
		})
	}
	for i, rng := range ranges {
		insert(rng.StartLine, rng.StartCharacter, fmt.Sprintf("<%d kind=%q>", i, rng.Kind))
		insert(rng.EndLine, rng.EndCharacter, fmt.Sprintf("</%d>", i))
	}
	filename := mark.path()
	mapper, err := env.Editor.Mapper(filename)
	if err != nil {
		mark.errorf("Editor.Mapper(%s) failed: %v", filename, err)
		return
	}
	got, _, err := source.ApplyProtocolEdits(mapper, edits)
	if err != nil {
		mark.errorf("ApplyProtocolEdits failed: %v", err)
		return
	}
	want, _ := g.Get(mark.run.env.T, "", got)
	if diff := compare.Bytes(want, got); diff != "" {
		mark.errorf("foldingRange mismatch:\n%s", diff)
	}
}

// formatMarker implements the @format marker.
func formatMarker(mark marker, golden *Golden) {
	edits, err := mark.server().Formatting(mark.run.env.Ctx, &protocol.DocumentFormattingParams{
		TextDocument: protocol.TextDocumentIdentifier{URI: mark.uri()},
	})
	var got []byte
	if err != nil {
		got = []byte(err.Error() + "\n") // all golden content is newline terminated
	} else {
		env := mark.run.env
		filename := mark.path()
		mapper, err := env.Editor.Mapper(filename)
		if err != nil {
			mark.errorf("Editor.Mapper(%s) failed: %v", filename, err)
		}

		got, _, err = source.ApplyProtocolEdits(mapper, edits)
		if err != nil {
			mark.errorf("ApplyProtocolEdits failed: %v", err)
			return
		}
	}

	want, ok := golden.Get(mark.run.env.T, "", got)
	if !ok {
		mark.errorf("missing golden file @%s", golden.id)
		return
	}

	if diff := compare.Bytes(want, got); diff != "" {
		mark.errorf("golden file @%s does not match format results:\n%s", golden.id, diff)
	}
}

func highlightMarker(mark marker, src protocol.Location, dsts ...protocol.Location) {
	highlights := mark.run.env.DocumentHighlight(src)
	var got []protocol.Range
	for _, h := range highlights {
		got = append(got, h.Range)
	}

	var want []protocol.Range
	for _, d := range dsts {
		want = append(want, d.Range)
	}

	sortRanges := func(s []protocol.Range) {
		sort.Slice(s, func(i, j int) bool {
			return protocol.CompareRange(s[i], s[j]) < 0
		})
	}

	sortRanges(got)
	sortRanges(want)

	if diff := cmp.Diff(want, got); diff != "" {
		mark.errorf("DocumentHighlight(%v) mismatch (-want +got):\n%s", src, diff)
	}
}

// hoverMarker implements the @hover marker, running textDocument/hover at the
// given src location and asserting that the resulting hover is over the dst
// location (typically a span surrounding src), and that the markdown content
// matches the golden content.
func hoverMarker(mark marker, src, dst protocol.Location, golden *Golden) {
	content, gotDst := mark.run.env.Hover(src)
	if gotDst != dst {
		mark.errorf("hover location does not match:\n\tgot: %s\n\twant %s)", mark.run.fmtLoc(gotDst), mark.run.fmtLoc(dst))
	}
	gotMD := ""
	if content != nil {
		gotMD = content.Value
	}
	wantMD := ""
	if golden != nil {
		wantBytes, _ := golden.Get(mark.run.env.T, "hover.md", []byte(gotMD))
		wantMD = string(wantBytes)
	}
	// Normalize newline termination: archive files can't express non-newline
	// terminated files.
	if strings.HasSuffix(wantMD, "\n") && !strings.HasSuffix(gotMD, "\n") {
		gotMD += "\n"
	}
	if diff := tests.DiffMarkdown(wantMD, gotMD); diff != "" {
		mark.errorf("hover markdown mismatch (-want +got):\n%s", diff)
	}
}

// locMarker implements the @loc marker. It is executed before other
// markers, so that locations are available.
func locMarker(mark marker, loc protocol.Location) protocol.Location { return loc }

// diagMarker implements the @diag marker. It eliminates diagnostics from
// the observed set in mark.test.
func diagMarker(mark marker, loc protocol.Location, re *regexp.Regexp) {
	if _, ok := removeDiagnostic(mark, loc, re); !ok {
		mark.errorf("no diagnostic at %v matches %q", loc, re)
	}
}

// removeDiagnostic looks for a diagnostic matching loc at the given position.
//
// If found, it returns (diag, true), and eliminates the matched diagnostic
// from the unmatched set.
//
// If not found, it returns (protocol.Diagnostic{}, false).
func removeDiagnostic(mark marker, loc protocol.Location, re *regexp.Regexp) (protocol.Diagnostic, bool) {
	loc.Range.End = loc.Range.Start // diagnostics ignore end position.
	diags := mark.run.diags[loc]
	for i, diag := range diags {
		if re.MatchString(diag.Message) {
			mark.run.diags[loc] = append(diags[:i], diags[i+1:]...)
			return diag, true
		}
	}
	return protocol.Diagnostic{}, false
}

// renameMarker implements the @rename(location, new, golden) marker.
func renameMarker(mark marker, loc protocol.Location, newName string, golden *Golden) {
	changed, err := rename(mark.run.env, loc, newName)
	if err != nil {
		mark.errorf("rename failed: %v. (Use @renameerr for expected errors.)", err)
		return
	}
	checkChangedFiles(mark, changed, golden)
}

// renameErrMarker implements the @renamererr(location, new, error) marker.
func renameErrMarker(mark marker, loc protocol.Location, newName string, wantErr wantError) {
	_, err := rename(mark.run.env, loc, newName)
	wantErr.check(mark, err)
}

func signatureMarker(mark marker, src protocol.Location, want string) {
	got := mark.run.env.SignatureHelp(src)
	if got == nil || len(got.Signatures) != 1 {
		mark.errorf("signatureHelp = %v, want exactly 1 signature", got)
		return
	}
	if got := got.Signatures[0].Label; got != want {
		mark.errorf("signatureHelp: got %q, want %q", got, want)
	}
}

// rename returns the new contents of the files that would be modified
// by renaming the identifier at loc to newName.
func rename(env *Env, loc protocol.Location, newName string) (map[string][]byte, error) {
	// We call Server.Rename directly, instead of
	//   env.Editor.Rename(env.Ctx, loc, newName)
	// to isolate Rename from PrepareRename, and because we don't
	// want to modify the file system in a scenario with multiple
	// @rename markers.

	editMap, err := env.Editor.Server.Rename(env.Ctx, &protocol.RenameParams{
		TextDocument: protocol.TextDocumentIdentifier{URI: loc.URI},
		Position:     loc.Range.Start,
		NewName:      string(newName),
	})
	if err != nil {
		return nil, err
	}

	fileChanges := make(map[string][]byte)
	if err := applyDocumentChanges(env, editMap.DocumentChanges, fileChanges); err != nil {
		return nil, fmt.Errorf("applying document changes: %v", err)
	}
	return fileChanges, nil
}

// applyDocumentChanges applies the given document changes to the editor buffer
// content, recording the resulting contents in the fileChanges map. It is an
// error for a change to an edit a file that is already present in the
// fileChanges map.
func applyDocumentChanges(env *Env, changes []protocol.DocumentChanges, fileChanges map[string][]byte) error {
	getMapper := func(path string) (*protocol.Mapper, error) {
		if _, ok := fileChanges[path]; ok {
			return nil, fmt.Errorf("internal error: %s is already edited", path)
		}
		return env.Editor.Mapper(path)
	}

	for _, change := range changes {
		if change.RenameFile != nil {
			// rename
			oldFile := env.Sandbox.Workdir.URIToPath(change.RenameFile.OldURI)
			mapper, err := getMapper(oldFile)
			if err != nil {
				return err
			}
			newFile := env.Sandbox.Workdir.URIToPath(change.RenameFile.NewURI)
			fileChanges[newFile] = mapper.Content
		} else {
			// edit
			filename := env.Sandbox.Workdir.URIToPath(change.TextDocumentEdit.TextDocument.URI)
			mapper, err := getMapper(filename)
			if err != nil {
				return err
			}
			patched, _, err := source.ApplyProtocolEdits(mapper, change.TextDocumentEdit.Edits)
			if err != nil {
				return err
			}
			fileChanges[filename] = patched
		}
	}

	return nil
}

func codeActionMarker(mark marker, actionKind string, start, end protocol.Location, golden *Golden) {
	// Request the range from start.Start to end.End.
	loc := start
	loc.Range.End = end.Range.End

	// Apply the fix it suggests.
	changed, err := codeAction(mark.run.env, loc.URI, loc.Range, actionKind, nil)
	if err != nil {
		mark.errorf("codeAction failed: %v", err)
		return
	}

	// Check the file state.
	checkChangedFiles(mark, changed, golden)
}

func codeActionErrMarker(mark marker, actionKind string, start, end protocol.Location, wantErr wantError) {
	loc := start
	loc.Range.End = end.Range.End
	_, err := codeAction(mark.run.env, loc.URI, loc.Range, actionKind, nil)
	wantErr.check(mark, err)
}

// codeLensesMarker runs the @codelenses() marker, collecting @codelens marks
// in the current file and comparing with the result of the
// textDocument/codeLens RPC.
func codeLensesMarker(mark marker) {
	type codeLens struct {
		Range protocol.Range
		Title string
	}

	lenses := mark.run.env.CodeLens(mark.path())
	var got []codeLens
	for _, lens := range lenses {
		title := ""
		if lens.Command != nil {
			title = lens.Command.Title
		}
		got = append(got, codeLens{lens.Range, title})
	}

	var want []codeLens
	mark.consumeExtraNotes("codelens", actionMarkerFunc(func(mark marker, loc protocol.Location, title string) {
		want = append(want, codeLens{loc.Range, title})
	}))

	for _, s := range [][]codeLens{got, want} {
		sort.Slice(s, func(i, j int) bool {
			li, lj := s[i], s[j]
			if c := protocol.CompareRange(li.Range, lj.Range); c != 0 {
				return c < 0
			}
			return li.Title < lj.Title
		})
	}

	if diff := cmp.Diff(want, got); diff != "" {
		mark.errorf("codelenses: unexpected diff (-want +got):\n%s", diff)
	}
}

// consumeExtraNotes runs the provided func for each extra note with the given
// name, and deletes all matching notes.
func (mark marker) consumeExtraNotes(name string, f func(marker)) {
	uri := mark.uri()
	notes := mark.run.extraNotes[uri][name]
	delete(mark.run.extraNotes[uri], name)

	for _, note := range notes {
		f(marker{run: mark.run, note: note})
	}
}

// suggestedfixMarker implements the @suggestedfix(location, regexp,
// kind, golden) marker. It acts like @diag(location, regexp), to set
// the expectation of a diagnostic, but then it applies the first code
// action of the specified kind suggested by the matched diagnostic.
func suggestedfixMarker(mark marker, loc protocol.Location, re *regexp.Regexp, actionKind string, golden *Golden) {
	loc.Range.End = loc.Range.Start // diagnostics ignore end position.
	// Find and remove the matching diagnostic.
	diag, ok := removeDiagnostic(mark, loc, re)
	if !ok {
		mark.errorf("no diagnostic at %v matches %q", loc, re)
		return
	}

	// Apply the fix it suggests.
	changed, err := codeAction(mark.run.env, loc.URI, diag.Range, actionKind, &diag)
	if err != nil {
		mark.errorf("suggestedfix failed: %v. (Use @suggestedfixerr for expected errors.)", err)
		return
	}

	// Check the file state.
	checkChangedFiles(mark, changed, golden)
}

// codeAction executes a textDocument/codeAction request for the specified
// location and kind. If diag is non-nil, it is used as the code action
// context.
//
// The resulting map contains resulting file contents after the code action is
// applied. Currently, this function does not support code actions that return
// edits directly; it only supports code action commands.
func codeAction(env *Env, uri protocol.DocumentURI, rng protocol.Range, actionKind string, diag *protocol.Diagnostic) (map[string][]byte, error) {
	// Request all code actions that apply to the diagnostic.
	// (The protocol supports filtering using Context.Only={actionKind}
	// but we can give a better error if we don't filter.)
	params := &protocol.CodeActionParams{
		TextDocument: protocol.TextDocumentIdentifier{URI: uri},
		Range:        rng,
		Context: protocol.CodeActionContext{
			Only: nil, // => all kinds
		},
	}
	if diag != nil {
		params.Context.Diagnostics = []protocol.Diagnostic{*diag}
	}

	actions, err := env.Editor.Server.CodeAction(env.Ctx, params)
	if err != nil {
		return nil, err
	}

	// Find the sole candidates CodeAction of the specified kind (e.g. refactor.rewrite).
	var candidates []protocol.CodeAction
	for _, act := range actions {
		if act.Kind == protocol.CodeActionKind(actionKind) {
			candidates = append(candidates, act)
		}
	}
	if len(candidates) != 1 {
		for _, act := range actions {
			env.T.Logf("found CodeAction Kind=%s Title=%q", act.Kind, act.Title)
		}
		return nil, fmt.Errorf("found %d CodeActions of kind %s for this diagnostic, want 1", len(candidates), actionKind)
	}
	action := candidates[0]

	// Apply the codeAction.
	//
	// Spec:
	//  "If a code action provides an edit and a command, first the edit is
	//  executed and then the command."
	fileChanges := make(map[string][]byte)
	// An action may specify an edit and/or a command, to be
	// applied in that order. But since applyDocumentChanges(env,
	// action.Edit.DocumentChanges) doesn't compose, for now we
	// assert that all commands used in the @suggestedfix tests
	// return only a command.
	if action.Edit != nil {
		if action.Edit.Changes != nil {
			env.T.Errorf("internal error: discarding unexpected CodeAction{Kind=%s, Title=%q}.Edit.Changes", action.Kind, action.Title)
		}
		if action.Edit.DocumentChanges != nil {
			if err := applyDocumentChanges(env, action.Edit.DocumentChanges, fileChanges); err != nil {
				return nil, fmt.Errorf("applying document changes: %v", err)
			}
		}
	}

	if action.Command != nil {
		// This is a typical CodeAction command:
		//
		//   Title:     "Implement error"
		//   Command:   gopls.apply_fix
		//   Arguments: [{"Fix":"stub_methods","URI":".../a.go","Range":...}}]
		//
		// The client makes an ExecuteCommand RPC to the server,
		// which dispatches it to the ApplyFix handler.
		// ApplyFix dispatches to the "stub_methods" suggestedfix hook (the meat).
		// The server then makes an ApplyEdit RPC to the client,
		// whose Awaiter hook gathers the edits instead of applying them.

		_ = env.Awaiter.takeDocumentChanges() // reset (assuming Env is confined to this thread)

		if _, err := env.Editor.Server.ExecuteCommand(env.Ctx, &protocol.ExecuteCommandParams{
			Command:   action.Command.Command,
			Arguments: action.Command.Arguments,
		}); err != nil {
			env.T.Fatalf("error converting command %q to edits: %v", action.Command.Command, err)
		}

		if err := applyDocumentChanges(env, env.Awaiter.takeDocumentChanges(), fileChanges); err != nil {
			return nil, fmt.Errorf("applying document changes from command: %v", err)
		}
	}

	return fileChanges, nil
}

// TODO(adonovan): suggestedfixerr

// refsMarker implements the @refs marker.
func refsMarker(mark marker, src protocol.Location, want ...protocol.Location) {
	refs := func(includeDeclaration bool, want []protocol.Location) error {
		got, err := mark.server().References(mark.run.env.Ctx, &protocol.ReferenceParams{
			TextDocumentPositionParams: protocol.LocationTextDocumentPositionParams(src),
			Context: protocol.ReferenceContext{
				IncludeDeclaration: includeDeclaration,
			},
		})
		if err != nil {
			return err
		}

		return compareLocations(mark, got, want)
	}

	for _, includeDeclaration := range []bool{false, true} {
		// Ignore first 'want' location if we didn't request the declaration.
		// TODO(adonovan): don't assume a single declaration:
		// there may be >1 if corresponding methods are considered.
		want := want
		if !includeDeclaration && len(want) > 0 {
			want = want[1:]
		}
		if err := refs(includeDeclaration, want); err != nil {
			mark.errorf("refs(includeDeclaration=%t) failed: %v",
				includeDeclaration, err)
		}
	}
}

// implementationMarker implements the @implementation marker.
func implementationMarker(mark marker, src protocol.Location, want ...protocol.Location) {
	got, err := mark.server().Implementation(mark.run.env.Ctx, &protocol.ImplementationParams{
		TextDocumentPositionParams: protocol.LocationTextDocumentPositionParams(src),
	})
	if err != nil {
		mark.errorf("implementation at %s failed: %v", src, err)
		return
	}
	if err := compareLocations(mark, got, want); err != nil {
		mark.errorf("implementation: %v", err)
	}
}

// symbolMarker implements the @symbol marker.
func symbolMarker(mark marker, golden *Golden) {
	// Retrieve information about all symbols in this file.
	symbols, err := mark.server().DocumentSymbol(mark.run.env.Ctx, &protocol.DocumentSymbolParams{
		TextDocument: protocol.TextDocumentIdentifier{URI: mark.uri()},
	})
	if err != nil {
		mark.errorf("DocumentSymbol request failed: %v", err)
		return
	}

	// Format symbols one per line, sorted (in effect) by first column, a dotted name.
	var lines []string
	for _, symbol := range symbols {
		// Each result element is a union of (legacy)
		// SymbolInformation and (new) DocumentSymbol,
		// so we ascertain which one and then transcode.
		data, err := json.Marshal(symbol)
		if err != nil {
			mark.run.env.T.Fatal(err)
		}
		if _, ok := symbol.(map[string]any)["location"]; ok {
			// This case is not reached because Editor initialization
			// enables HierarchicalDocumentSymbolSupport.
			// TODO(adonovan): test this too.
			var sym protocol.SymbolInformation
			if err := json.Unmarshal(data, &sym); err != nil {
				mark.run.env.T.Fatal(err)
			}
			mark.errorf("fake Editor doesn't support SymbolInformation")

		} else {
			var sym protocol.DocumentSymbol // new hierarchical hotness
			if err := json.Unmarshal(data, &sym); err != nil {
				mark.run.env.T.Fatal(err)
			}

			// Print each symbol in the response tree.
			var visit func(sym protocol.DocumentSymbol, prefix []string)
			visit = func(sym protocol.DocumentSymbol, prefix []string) {
				var out strings.Builder
				out.WriteString(strings.Join(prefix, "."))
				fmt.Fprintf(&out, " %q", sym.Detail)
				if delta := sym.Range.End.Line - sym.Range.Start.Line; delta > 0 {
					fmt.Fprintf(&out, " +%d lines", delta)
				}
				lines = append(lines, out.String())

				for _, child := range sym.Children {
					visit(child, append(prefix, child.Name))
				}
			}
			visit(sym, []string{sym.Name})
		}
	}
	sort.Strings(lines)
	lines = append(lines, "") // match trailing newline in .txtar file
	got := []byte(strings.Join(lines, "\n"))

	// Compare with golden.
	want, ok := golden.Get(mark.run.env.T, "", got)
	if !ok {
		mark.errorf("%s: missing golden file @%s", mark.note.Name, golden.id)
	} else if diff := cmp.Diff(string(got), string(want)); diff != "" {
		mark.errorf("%s: unexpected output: got:\n%s\nwant:\n%s\ndiff:\n%s",
			mark.note.Name, got, want, diff)
	}
}

// compareLocations returns an error message if got and want are not
// the same set of locations. The marker is used only for fmtLoc.
func compareLocations(mark marker, got, want []protocol.Location) error {
	toStrings := func(locs []protocol.Location) []string {
		strs := make([]string, len(locs))
		for i, loc := range locs {
			strs[i] = mark.run.fmtLoc(loc)
		}
		sort.Strings(strs)
		return strs
	}
	if diff := cmp.Diff(toStrings(want), toStrings(got)); diff != "" {
		return fmt.Errorf("incorrect result locations: (got %d, want %d):\n%s",
			len(got), len(want), diff)
	}
	return nil
}

func workspaceSymbolMarker(mark marker, query string, golden *Golden) {
	params := &protocol.WorkspaceSymbolParams{
		Query: query,
	}

	gotSymbols, err := mark.server().Symbol(mark.run.env.Ctx, params)
	if err != nil {
		mark.errorf("Symbol(%q) failed: %v", query, err)
		return
	}
	var got bytes.Buffer
	for _, s := range gotSymbols {
		// Omit the txtar position of the symbol location; otherwise edits to the
		// txtar archive lead to unexpected failures.
		loc := mark.run.fmtLocDetails(s.Location, false)
		// TODO(rfindley): can we do better here, by detecting if the location is
		// relative to GOROOT?
		if loc == "" {
			loc = "<unknown>"
		}
		fmt.Fprintf(&got, "%s %s %s\n", loc, s.Name, s.Kind)
	}

	want, ok := golden.Get(mark.run.env.T, "", got.Bytes())
	if !ok {
		mark.errorf("missing golden file @%s", golden.id)
		return
	}

	if diff := compare.Bytes(want, got.Bytes()); diff != "" {
		mark.errorf("Symbol(%q) mismatch:\n%s", query, diff)
	}
}
