// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package regtest

import (
	"bytes"
	"context"
	"flag"
	"fmt"
	"go/token"
	"io/fs"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
	"testing"

	"golang.org/x/tools/go/expect"
	"golang.org/x/tools/gopls/internal/hooks"
	"golang.org/x/tools/gopls/internal/lsp/cache"
	"golang.org/x/tools/gopls/internal/lsp/debug"
	"golang.org/x/tools/gopls/internal/lsp/fake"
	"golang.org/x/tools/gopls/internal/lsp/lsprpc"
	"golang.org/x/tools/gopls/internal/lsp/protocol"
	"golang.org/x/tools/gopls/internal/lsp/safetoken"
	"golang.org/x/tools/gopls/internal/lsp/tests"
	"golang.org/x/tools/gopls/internal/lsp/tests/compare"
	"golang.org/x/tools/internal/jsonrpc2"
	"golang.org/x/tools/internal/jsonrpc2/servertest"
	"golang.org/x/tools/internal/testenv"
	"golang.org/x/tools/txtar"
)

var updateGolden = flag.Bool("update", false, "if set, update test data during marker tests")

// RunMarkerTests runs "marker" tests in the given test data directory.
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
// Each call argument is coerced to the type of the corresponding parameter of
// the designated function. The coercion logic may use the surrounding context,
// such as the position or nearby text. See the Argument coercion section below
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
// There are three types of file within the test archive that are given special
// treatment by the test runner:
//   - "flags": this file is parsed as flags configuring the MarkerTest
//     instance. For example, -min_go=go1.18 sets the minimum required Go version
//     for the test.
//   - "settings.json": (*) this file is parsed as JSON, and used as the
//     session configuration (see gopls/doc/settings.md)
//   - Golden files: Within the archive, file names starting with '@' are
//     treated as "golden" content, and are not written to disk, but instead are
//     made available to test methods expecting an argument of type *Golden,
//     using the identifier following '@'. For example, if the first parameter of
//     Foo were of type *Golden, the test runner would coerce the identifier a in
//     the call @foo(a, "b", 3) into a *Golden by collecting golden file data
//     starting with "@a/".
//
// # Marker types
//
// The following markers are supported within marker tests:
//   - @diag(location, regexp): (***) see Special markers below.
//   - @hover(src, dst location, g Golden): perform a textDocument/hover at the
//     src location, and check that the result spans the dst location, with hover
//     content matching "hover.md" in the golden data g.
//   - @loc(name, location): (**) see [Special markers] below.
//
// # Argument conversion
//
// In additon to passing through literals as basic types, the marker test
// runner supports the following coercions into non-basic types:
//   - string->regexp: strings are parsed as regular expressions
//   - string->location: strings are parsed as regular expressions and used to
//     match the first location in the line preceding the note
//   - name->location: identifiers may reference named locations created using
//     the @loc marker.
//   - name->Golden: identifiers match the golden content contained in archive
//     files prefixed by @<name>.
//
// # Special markers
//
// There are two markers that have additional special handling, rather than
// just invoking the test method of the same name:
//   - @loc(name, location): (**) specifies a named location in the source. These
//     locations may be referenced by other markers.
//   - @diag(location, regexp): (***) specifies an expected diagnostic
//     matching the given regexp at the given location. The test runner requires
//     a 1:1 correspondence between observed diagnostics and diag annotations:
//     it is an error if the test runner receives a publishDiagnostics
//     notification for a diagnostic that is not annotated, or if a diagnostic
//     annotation does not match an existing diagnostic.
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
//	(env *Env, src, dsc protocol.Location, g *Golden).
//
// The env argument holds the implicit test environment, including fake editor
// with open files, and sandboxed directory.
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
// # TODO
//
// This API is a work-in-progress, as we migrate existing marker tests from
// internal/lsp/tests.
//
// Remaining TODO:
//   - parallelize/optimize test execution
//   - actually support regexp locations?
//   - (*) add support for per-test editor settings (via a settings.json file)
//   - (**) add support for locs
//   - (***) add special handling for diagnostics
//   - add support for per-test environment?
//   - reorganize regtest packages (and rename to just 'test'?)
//
// Existing marker tests to port:
//   - CallHierarchy
//   - CodeLens
//   - Diagnostics
//   - CompletionItems
//   - Completions
//   - CompletionSnippets
//   - UnimportedCompletions
//   - DeepCompletions
//   - FuzzyCompletions
//   - CaseSensitiveCompletions
//   - RankCompletions
//   - FoldingRanges
//   - Formats
//   - Imports
//   - SemanticTokens
//   - SuggestedFixes
//   - FunctionExtractions
//   - MethodExtractions
//   - Definitions
//   - Implementations
//   - Highlights
//   - References
//   - Renames
//   - PrepareRenames
//   - Symbols
//   - InlayHints
//   - WorkspaceSymbols
//   - Signatures
//   - Links
//   - AddImport
//   - SelectionRanges
func RunMarkerTests(t *testing.T, dir string) {
	tests, err := loadMarkerTests(dir)
	if err != nil {
		t.Fatal(err)
	}

	// Opt: use a shared cache.
	// TODO: opt: use a memoize store with no eviction.
	cache := cache.New(nil, nil)

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			// TODO(rfindley): it may be more useful to have full support for build
			// constraints.
			if test.minGoVersion != "" {
				var go1point int
				if _, err := fmt.Sscanf(test.minGoVersion, "go1.%d", &go1point); err != nil {
					t.Fatalf("parsing -min_go version: %v", err)
				}
				testenv.NeedsGo1Point(t, 18)
			}
			test.executed = true
			env := newEnv(t, cache, test.files)
			// TODO(rfindley): make it easier to clean up the regtest environment.
			defer env.Editor.Shutdown(context.Background()) // ignore error
			defer env.Sandbox.Close()                       // ignore error

			// Open all files so that we operate consistently with LSP clients, and
			// (pragmatically) so that we have a Mapper available via the fake
			// editor.
			//
			// This also allows avoiding mutating the editor state in tests.
			for file := range test.files {
				env.OpenFile(file)
			}

			// Invoke each method in the test.
			for _, note := range test.notes {
				posn := safetoken.StartPosition(test.fset, note.Pos)
				mi, ok := markers[note.Name]
				if !ok {
					t.Errorf("%s: no marker function named %s", posn, note.Name)
					continue
				}

				// The first converter corresponds to the *Env argument. All others
				// must be coerced from the marker syntax.
				if got, want := len(note.Args), len(mi.converters); got != want {
					t.Errorf("%s: got %d argumentsto %s, expect %d", posn, got, note.Name, want)
					continue
				}

				args := []reflect.Value{reflect.ValueOf(env)}
				hasErrors := false
				for i, in := range note.Args {
					// Special handling for the blank identifier: treat it as the zero
					// value.
					if ident, ok := in.(expect.Identifier); ok && ident == "_" {
						zero := reflect.Zero(mi.paramTypes[i])
						args = append(args, zero)
						continue
					}
					out, err := mi.converters[i](env, test, note, in)
					if err != nil {
						t.Errorf("%s: converting argument #%d of %s (%v): %v", posn, i, note.Name, in, err)
						hasErrors = true
					}
					args = append(args, reflect.ValueOf(out))
				}

				if !hasErrors {
					mi.fn.Call(args)
				}
			}
		})
	}

	// If updateGolden is set, golden content was updated during text execution,
	// so we can now update the test data.
	// TODO(rfindley): even when -update is not set, compare updated content with
	// actual content.
	if *updateGolden {
		if err := writeMarkerTests(dir, tests); err != nil {
			t.Fatalf("failed to -update: %v", err)
		}
	}
}

// supported markers, excluding @loc and @diag which are handled separately.
//
// Each marker func must accept an *Env as its first argument, with subsequent
// arguments coerced from the arguments to the marker annotation.
//
// Marker funcs should not mutate the test environment (e.g. via opening files
// or applying edits in the editor).
var markers = map[string]markerInfo{
	"hover": makeMarker(hoverMarker),
}

// MarkerTest holds all the test data extracted from a test txtar archive.
//
// See the documentation for RunMarkerTests for more information on the archive
// format.
type MarkerTest struct {
	name   string         // relative path to the txtar file in the testdata dir
	fset   *token.FileSet // fileset used for parsing notes
	files  map[string][]byte
	notes  []*expect.Note
	golden map[string]*Golden

	// executed tracks whether the test was executed.
	//
	// When -update is set, only tests that were actually executed are written.
	executed bool

	// flags holds flags extracted from the special "flags" archive file.
	flags []string

	// Parsed flags values.
	minGoVersion string
}

// flagSet returns the flagset used for parsing the special "flags" file in the
// test archive.
func (t *MarkerTest) flagSet() *flag.FlagSet {
	flags := flag.NewFlagSet(t.name, flag.ContinueOnError)
	flags.StringVar(&t.minGoVersion, "min_go", "", "if set, the minimum go1.X version required for this test")
	return flags
}

// Golden holds extracted golden content for a single @<name> prefix. The
//
// When -update is set, golden captures the updated golden contents for later
// writing.
type Golden struct {
	data    map[string][]byte
	updated map[string][]byte
}

// Get returns golden content for the given name, which corresponds to the
// relative path following the golden prefix @<name>/. For example, to access
// the content of @foo/path/to/result.json from the Golden associated with
// @foo, name should be "path/to/result.json".
//
// If -update is set, the given update function will be called to get the
// updated golden content that should be written back to testdata.
func (g *Golden) Get(t testing.TB, name string, update func() []byte) []byte {
	if *updateGolden {
		d := update()
		if existing, ok := g.updated[name]; ok {
			// Multiple tests may reference the same golden data, but if they do they
			// must agree about its expected content.
			if diff := compare.Text(string(existing), string(d)); diff != "" {
				t.Fatalf("conflicting updates for golden data %s:\n%s", name, diff)
			}
		}
		if g.updated == nil {
			g.updated = make(map[string][]byte)
		}
		g.updated[name] = d
		return d
	}
	return g.data[name]
}

// loadMarkerTests walks the given dir looking for .txt files, which it
// interprets as a txtar archive.
//
// See the documentation for RunMarkerTests for more details on the test data
// archive.
//
// TODO(rfindley): this test could sanity check the results. For example, it is
// too easy to write "// @" instead of "//@", which we will happy skip silently.
func loadMarkerTests(dir string) ([]*MarkerTest, error) {
	var tests []*MarkerTest
	err := filepath.WalkDir(dir, func(path string, d fs.DirEntry, err error) error {
		if strings.HasSuffix(path, ".txt") {
			content, err := os.ReadFile(path)
			if err != nil {
				return err
			}
			archive := txtar.Parse(content)
			name := strings.TrimPrefix(path, dir+string(filepath.Separator))
			test, err := loadMarkerTest(name, archive)
			if err != nil {
				return fmt.Errorf("%s: %v", path, err)
			}
			tests = append(tests, test)
		}
		return nil
	})
	return tests, err
}

func loadMarkerTest(name string, archive *txtar.Archive) (*MarkerTest, error) {
	test := &MarkerTest{
		name:   name,
		fset:   token.NewFileSet(),
		files:  make(map[string][]byte),
		golden: make(map[string]*Golden),
	}
	for _, file := range archive.Files {
		if file.Name == "flags" {
			test.flags = strings.Fields(string(file.Data))
			if err := test.flagSet().Parse(test.flags); err != nil {
				return nil, fmt.Errorf("parsing flags: %v", err)
			}
			continue
		}
		if strings.HasPrefix(file.Name, "@") {
			// golden content
			// TODO: use strings.Cut once we are on 1.18+.
			idx := strings.IndexByte(file.Name, '/')
			if idx < 0 {
				return nil, fmt.Errorf("golden file path %q must contain '/'", file.Name)
			}
			goldenID := file.Name[len("@"):idx]
			if _, ok := test.golden[goldenID]; !ok {
				test.golden[goldenID] = &Golden{
					data: make(map[string][]byte),
				}
			}
			test.golden[goldenID].data[file.Name[idx+len("/"):]] = file.Data
		} else {
			// ordinary file content
			notes, err := expect.Parse(test.fset, file.Name, file.Data)
			if err != nil {
				return nil, fmt.Errorf("parsing notes in %q: %v", file.Name, err)
			}
			test.notes = append(test.notes, notes...)
			test.files[file.Name] = file.Data
		}
	}
	return test, nil
}

// writeMarkerTests writes the updated golden content to the test data files.
func writeMarkerTests(dir string, tests []*MarkerTest) error {
	for _, test := range tests {
		if !test.executed {
			continue
		}
		arch := &txtar.Archive{}

		// Special configuration files go first.
		if len(test.flags) > 0 {
			flags := strings.Join(test.flags, " ")
			arch.Files = append(arch.Files, txtar.File{Name: "flags", Data: []byte(flags)})
		}

		// ...followed by ordinary files
		var files []txtar.File
		for name, data := range test.files {
			files = append(files, txtar.File{Name: name, Data: data})
		}
		sort.Slice(files, func(i, j int) bool {
			return files[i].Name < files[j].Name
		})
		arch.Files = append(arch.Files, files...)

		// ...followed by golden files
		var goldenFiles []txtar.File
		for id, golden := range test.golden {
			for name, data := range golden.updated {
				fullName := "@" + id + "/" + name
				goldenFiles = append(goldenFiles, txtar.File{Name: fullName, Data: data})
			}
		}
		sort.Slice(goldenFiles, func(i, j int) bool {
			return goldenFiles[i].Name < goldenFiles[j].Name
		})
		arch.Files = append(arch.Files, goldenFiles...)

		data := txtar.Format(arch)
		filename := filepath.Join(dir, test.name)
		if err := os.WriteFile(filename, data, 0644); err != nil {
			return err
		}
	}
	return nil
}

// newEnv creates a new environment for a marker test.
//
// TODO(rfindley): simplify and refactor the construction of testing
// environments across regtests, marker tests, and benchmarks.
func newEnv(t *testing.T, cache *cache.Cache, files map[string][]byte) *Env {
	sandbox, err := fake.NewSandbox(&fake.SandboxConfig{
		RootDir: t.TempDir(),
		GOPROXY: "https://proxy.golang.org",
		Files:   files,
	})
	if err != nil {
		t.Fatal(err)
	}

	// Put a debug instance in the context to prevent logging to stderr.
	// See associated TODO in runner.go: we should revisit this pattern.
	ctx := context.Background()
	ctx = debug.WithInstance(ctx, "", "off")

	awaiter := NewAwaiter(sandbox.Workdir)
	ss := lsprpc.NewStreamServer(cache, false, hooks.Options)
	server := servertest.NewPipeServer(ss, jsonrpc2.NewRawStream)
	editor, err := fake.NewEditor(sandbox, fake.EditorConfig{}).Connect(ctx, server, awaiter.Hooks())
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

type markerInfo struct {
	fn         reflect.Value  // the func to invoke
	paramTypes []reflect.Type // parameter types, for zero values
	converters []converter    // to convert non-blank arguments
}

type converter func(*Env, *MarkerTest, *expect.Note, interface{}) (interface{}, error)

// makeMarker uses reflection to load markerInfo for the given func value.
func makeMarker(fn interface{}) markerInfo {
	mi := markerInfo{
		fn: reflect.ValueOf(fn),
	}
	mtyp := mi.fn.Type()
	if mtyp.NumIn() == 0 || mtyp.In(0) != envType {
		panic(fmt.Sprintf("marker function %#v must accept *Env as its first argument", mi.fn))
	}
	if mtyp.NumOut() != 0 {
		panic(fmt.Sprintf("marker function %#v must not have results", mi.fn))
	}
	for a := 1; a < mtyp.NumIn(); a++ {
		in := mtyp.In(a)
		mi.paramTypes = append(mi.paramTypes, in)
		c := makeConverter(in)
		mi.converters = append(mi.converters, c)
	}
	return mi
}

// Types with special conversions.
var (
	envType      = reflect.TypeOf(&Env{})
	locationType = reflect.TypeOf(protocol.Location{})
	goldenType   = reflect.TypeOf(&Golden{})
)

func makeConverter(paramType reflect.Type) converter {
	switch paramType {
	case locationType:
		return locationConverter
	case goldenType:
		return goldenConverter
	default:
		return func(_ *Env, _ *MarkerTest, _ *expect.Note, arg interface{}) (interface{}, error) {
			if argType := reflect.TypeOf(arg); argType != paramType {
				return nil, fmt.Errorf("cannot convert type %s to %s", argType, paramType)
			}
			return arg, nil
		}
	}
}

// locationConverter converts a string argument into the protocol location
// corresponding to the first position of the string in the line preceding the
// note.
func locationConverter(env *Env, test *MarkerTest, note *expect.Note, arg interface{}) (interface{}, error) {
	file := test.fset.File(note.Pos)
	posn := safetoken.StartPosition(test.fset, note.Pos)
	lineStart := file.LineStart(posn.Line)
	startOff, endOff, err := safetoken.Offsets(file, lineStart, note.Pos)
	if err != nil {
		return nil, err
	}
	m := env.Editor.Mapper(file.Name())
	substr, ok := arg.(string)
	if !ok {
		return nil, fmt.Errorf("cannot convert argument type %T to location (must be a string to match the preceding line)", arg)
	}

	preceding := m.Content[startOff:endOff]
	idx := bytes.Index(preceding, []byte(substr))
	if idx < 0 {
		return nil, fmt.Errorf("substring %q not found in %q", substr, preceding)
	}
	off := startOff + idx
	loc, err := m.OffsetLocation(off, off+len(substr))
	return loc, err
}

// goldenConverter converts an identifier into the Golden directory of content
// prefixed by @<ident> in the test archive file.
func goldenConverter(_ *Env, test *MarkerTest, note *expect.Note, arg interface{}) (interface{}, error) {
	switch arg := arg.(type) {
	case expect.Identifier:
		golden := test.golden[string(arg)]
		// If there was no golden content for this identifier, we must create one
		// to handle the case where -update_golden is set: we need a place to store
		// the updated content.
		if golden == nil {
			golden = new(Golden)
			test.golden[string(arg)] = golden
		}
		return golden, nil
	default:
		return nil, fmt.Errorf("invalid input type %T: golden key must be an identifier", arg)
	}
}

// hoverMarker implements the @hover marker, running textDocument/hover at the
// given src location and asserting that the resulting hover is over the dst
// location (typically a span surrounding src), and that the markdown content
// matches the golden content.
func hoverMarker(env *Env, src, dst protocol.Location, golden *Golden) {
	content, gotDst := env.Hover(src)
	if gotDst != dst {
		env.T.Errorf("%s: hover location does not match:\n\tgot: %s\n\twant %s)", src, gotDst, dst)
	}
	gotMD := ""
	if content != nil {
		gotMD = content.Value
	}
	wantMD := ""
	if golden != nil {
		wantMD = string(golden.Get(env.T, "hover.md", func() []byte { return []byte(gotMD) }))
	}
	// Normalize newline termination: archive files can't express non-newline
	// terminated files.
	if strings.HasSuffix(wantMD, "\n") && !strings.HasSuffix(gotMD, "\n") {
		gotMD += "\n"
	}
	if diff := tests.DiffMarkdown(wantMD, gotMD); diff != "" {
		env.T.Errorf("%s: hover markdown mismatch (-want +got):\n%s", src, diff)
	}
}
