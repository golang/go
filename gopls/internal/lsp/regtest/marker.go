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
	"io/fs"
	"os"
	"path/filepath"
	"reflect"
	"regexp"
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

var update = flag.Bool("update", false, "if set, update test data during marker tests")

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
//   - "flags": this file is treated as a whitespace-separated list of flags
//     that configure the MarkerTest instance. For example, -min_go=go1.18 sets
//     the minimum required Go version for the test.
//     TODO(rfindley): support flag values containing whitespace.
//   - "settings.json": this file is parsed as JSON, and used as the
//     session configuration (see gopls/doc/settings.md)
//   - "env": this file is parsed as a list of VAR=VALUE fields specifying the
//     editor environment.
//   - Golden files: Within the archive, file names starting with '@' are
//     treated as "golden" content, and are not written to disk, but instead are
//     made available to test methods expecting an argument of type *Golden,
//     using the identifier following '@'. For example, if the first parameter of
//     Foo were of type *Golden, the test runner would convert the identifier a
//     in the call @foo(a, "b", 3) into a *Golden by collecting golden file
//     data starting with "@a/".
//
// # Marker types
//
// The following markers are supported within marker tests:
//   - diag(location, regexp): specifies an expected diagnostic matching the
//     given regexp at the given location. The test runner requires
//     a 1:1 correspondence between observed diagnostics and diag annotations
//   - def(src, dst location): perform a textDocument/definition request at
//     the src location, and check the the result points to the dst location.
//   - hover(src, dst location, g Golden): perform a textDocument/hover at the
//     src location, and checks that the result is the dst location, with hover
//     content matching "hover.md" in the golden data g.
//   - loc(name, location): specifies the name for a location in the source. These
//     locations may be referenced by other markers.
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
//	(c *markerContext, src, dsc protocol.Location, g *Golden).
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
//   - parallelize/optimize test execution
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
			config := fake.EditorConfig{
				Settings: test.settings,
				Env:      test.env,
			}
			c := &markerContext{
				test: test,
				env:  newEnv(t, cache, test.files, config),

				locations: make(map[expect.Identifier]protocol.Location),
				diags:     make(map[protocol.Location][]protocol.Diagnostic),
			}
			// TODO(rfindley): make it easier to clean up the regtest environment.
			defer c.env.Editor.Shutdown(context.Background()) // ignore error
			defer c.env.Sandbox.Close()                       // ignore error

			// Open all files so that we operate consistently with LSP clients, and
			// (pragmatically) so that we have a Mapper available via the fake
			// editor.
			//
			// This also allows avoiding mutating the editor state in tests.
			for file := range test.files {
				c.env.OpenFile(file)
			}

			// Pre-process locations.
			var notes []*expect.Note
			for _, note := range test.notes {
				switch note.Name {
				case "loc":
					mi := markers[note.Name]
					if err := runMarker(c, mi, note); err != nil {
						t.Error(err)
					}
				default:
					notes = append(notes, note)
				}
			}

			// Wait for the didOpen notifications to be processed, then collect
			// diagnostics.
			var diags map[string]*protocol.PublishDiagnosticsParams
			c.env.AfterChange(ReadAllDiagnostics(&diags))
			for path, params := range diags {
				uri := c.env.Sandbox.Workdir.URI(path)
				for _, diag := range params.Diagnostics {
					loc := protocol.Location{
						URI:   uri,
						Range: diag.Range,
					}
					c.diags[loc] = append(c.diags[loc], diag)
				}
			}

			// Invoke each remaining note function in the test.
			for _, note := range notes {
				mi, ok := markers[note.Name]
				if !ok {
					t.Errorf("%s: no marker function named %s", c.fmtPos(note.Pos), note.Name)
					continue
				}
				if err := runMarker(c, mi, note); err != nil {
					t.Error(err)
				}
			}

			// Any remaining (un-eliminated) diagnostics are an error.
			for loc, diags := range c.diags {
				for _, diag := range diags {
					t.Errorf("%s: unexpected diagnostic: %q", c.fmtLoc(loc), diag.Message)
				}
			}
		})
	}

	// If updateGolden is set, golden content was updated during text execution,
	// so we can now update the test data.
	// TODO(rfindley): even when -update is not set, compare updated content with
	// actual content.
	if *update {
		if err := writeMarkerTests(dir, tests); err != nil {
			t.Fatalf("failed to -update: %v", err)
		}
	}
}

// runMarker calls mi.fn with the arguments coerced from note.
func runMarker(c *markerContext, mi markerInfo, note *expect.Note) error {
	// The first converter corresponds to the *Env argument. All others
	// must be coerced from the marker syntax.
	if got, want := len(note.Args), len(mi.converters); got != want {
		return fmt.Errorf("%s: got %d arguments to %s, expect %d", c.fmtPos(note.Pos), got, note.Name, want)
	}

	args := []reflect.Value{reflect.ValueOf(c)}
	for i, in := range note.Args {
		// Special handling for the blank identifier: treat it as the zero
		// value.
		if ident, ok := in.(expect.Identifier); ok && ident == "_" {
			zero := reflect.Zero(mi.paramTypes[i])
			args = append(args, zero)
			continue
		}
		out, err := mi.converters[i](c, note, in)
		if err != nil {
			return fmt.Errorf("%s: converting argument #%d of %s (%v): %v", c.fmtPos(note.Pos), i, note.Name, in, err)
		}
		args = append(args, reflect.ValueOf(out))
	}

	mi.fn.Call(args)
	return nil
}

// Supported markers.
//
// Each marker func must accept an markerContext as its first argument, with
// subsequent arguments coerced from the marker arguments.
//
// Marker funcs should not mutate the test environment (e.g. via opening files
// or applying edits in the editor).
var markers = map[string]markerInfo{
	"def":   makeMarker(defMarker),
	"diag":  makeMarker(diagMarker),
	"hover": makeMarker(hoverMarker),
	"loc":   makeMarker(locMarker),
}

// MarkerTest holds all the test data extracted from a test txtar archive.
//
// See the documentation for RunMarkerTests for more information on the archive
// format.
type MarkerTest struct {
	name     string                 // relative path to the txtar file in the testdata dir
	fset     *token.FileSet         // fileset used for parsing notes
	archive  *txtar.Archive         // original test archive
	settings map[string]interface{} // gopls settings
	env      map[string]string      // editor environment
	files    map[string][]byte      // data files from the archive (excluding special files)
	notes    []*expect.Note         // extracted notes from data files
	golden   map[string]*Golden     // extracted golden content, by identifier name

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
	id      string
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
func (g *Golden) Get(t testing.TB, name string, getUpdate func() []byte) []byte {
	if *update {
		d := getUpdate()
		if existing, ok := g.updated[name]; ok {
			// Multiple tests may reference the same golden data, but if they do they
			// must agree about its expected content.
			if diff := compare.Text(string(existing), string(d)); diff != "" {
				t.Errorf("conflicting updates for golden data %s/%s:\n%s", g.id, name, diff)
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
		name:    name,
		fset:    token.NewFileSet(),
		archive: archive,
		files:   make(map[string][]byte),
		golden:  make(map[string]*Golden),
	}
	for _, file := range archive.Files {
		switch {
		case file.Name == "flags":
			test.flags = strings.Fields(string(file.Data))
			if err := test.flagSet().Parse(test.flags); err != nil {
				return nil, fmt.Errorf("parsing flags: %v", err)
			}

		case file.Name == "settings.json":
			if err := json.Unmarshal(file.Data, &test.settings); err != nil {
				return nil, err
			}

		case file.Name == "env":
			test.env = make(map[string]string)
			fields := strings.Fields(string(file.Data))
			for _, field := range fields {
				// TODO: use strings.Cut once we are on 1.18+.
				key, value, ok := cut(field, "=")
				if !ok {
					return nil, fmt.Errorf("env vars must be formatted as var=value, got %q", field)
				}
				test.env[key] = value
			}

		case strings.HasPrefix(file.Name, "@"): // golden content
			prefix, name, ok := cut(file.Name, "/")
			if !ok {
				return nil, fmt.Errorf("golden file path %q must contain '/'", file.Name)
			}
			goldenID := prefix[len("@"):]
			if _, ok := test.golden[goldenID]; !ok {
				test.golden[goldenID] = &Golden{
					id:   goldenID,
					data: make(map[string][]byte),
				}
			}
			test.golden[goldenID].data[name] = file.Data

		default: // ordinary file content
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

// cut is a copy of strings.Cut.
//
// TODO: once we only support Go 1.18+, just use strings.Cut.
func cut(s, sep string) (before, after string, found bool) {
	if i := strings.Index(s, sep); i >= 0 {
		return s[:i], s[i+len(sep):], true
	}
	return s, "", false
}

// writeMarkerTests writes the updated golden content to the test data files.
func writeMarkerTests(dir string, tests []*MarkerTest) error {
	for _, test := range tests {
		if !test.executed {
			continue
		}
		arch := &txtar.Archive{
			Comment: test.archive.Comment,
		}

		// Special configuration files go first.
		if len(test.flags) > 0 {
			flags := strings.Join(test.flags, " ")
			arch.Files = append(arch.Files, txtar.File{Name: "flags", Data: []byte(flags)})
		}
		if len(test.settings) > 0 {
			data, err := json.MarshalIndent(test.settings, "", "\t")
			if err != nil {
				return err
			}
			arch.Files = append(arch.Files, txtar.File{Name: "settings.json", Data: data})
		}
		if len(test.env) > 0 {
			var vars []string
			for k, v := range test.env {
				vars = append(vars, fmt.Sprintf("%s=%s", k, v))
			}
			sort.Strings(vars)
			data := []byte(strings.Join(vars, "\n"))
			arch.Files = append(arch.Files, txtar.File{Name: "env", Data: data})
		}

		// ...followed by ordinary files. Preserve the order they appeared in the
		// original archive.
		for _, file := range test.archive.Files {
			if _, ok := test.files[file.Name]; ok { // ordinary file
				arch.Files = append(arch.Files, file)
			}
		}

		// ...followed by golden files.
		var goldenFiles []txtar.File
		for id, golden := range test.golden {
			for name, data := range golden.updated {
				fullName := "@" + id + "/" + name
				goldenFiles = append(goldenFiles, txtar.File{Name: fullName, Data: data})
			}
		}
		// Unlike ordinary files, golden content is usually not manually edited, so
		// we sort lexically.
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
func newEnv(t *testing.T, cache *cache.Cache, files map[string][]byte, config fake.EditorConfig) *Env {
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
	editor, err := fake.NewEditor(sandbox, config).Connect(ctx, server, awaiter.Hooks())
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

type markerContext struct {
	test *MarkerTest
	env  *Env

	// Collected information.
	locations map[expect.Identifier]protocol.Location
	diags     map[protocol.Location][]protocol.Diagnostic
}

// fmtLoc formats the given pos in the context of the test, using
// archive-relative paths for files and including the line number in the full
// archive file.
func (c markerContext) fmtPos(pos token.Pos) string {
	file := c.test.fset.File(pos)
	if file == nil {
		c.env.T.Errorf("position %d not in test fileset", pos)
		return "<invalid location>"
	}
	m := c.env.Editor.Mapper(file.Name())
	if m == nil {
		c.env.T.Errorf("%s is not open", file.Name())
		return "<invalid location>"
	}
	loc, err := m.PosLocation(file, pos, pos)
	if err != nil {
		c.env.T.Errorf("Mapper(%s).PosLocation failed: %v", file.Name(), err)
	}
	return c.fmtLoc(loc)
}

// fmtLoc formats the given location in the context of the test, using
// archive-relative paths for files and including the line number in the full
// archive file.
func (c markerContext) fmtLoc(loc protocol.Location) string {
	if loc == (protocol.Location{}) {
		return "<missing location>"
	}
	lines := bytes.Count(c.test.archive.Comment, []byte("\n"))
	var name string
	for _, f := range c.test.archive.Files {
		lines++ // -- separator --
		uri := c.env.Sandbox.Workdir.URI(f.Name)
		if uri == loc.URI {
			name = f.Name
			break
		}
		lines += bytes.Count(f.Data, []byte("\n"))
	}
	if name == "" {
		c.env.T.Errorf("unable to find %s in test archive", loc)
		return "<invalid location>"
	}
	m := c.env.Editor.Mapper(name)
	s, err := m.LocationSpan(loc)
	if err != nil {
		c.env.T.Errorf("error formatting location %s: %v", loc, err)
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

	return fmt.Sprintf("%s:%s (%s:%s)", name, innerSpan, c.test.name, outerSpan)
}

// converter is the signature of argument converters.
type converter func(*markerContext, *expect.Note, interface{}) (interface{}, error)

// makeMarker uses reflection to load markerInfo for the given func value.
func makeMarker(fn interface{}) markerInfo {
	mi := markerInfo{
		fn: reflect.ValueOf(fn),
	}
	mtyp := mi.fn.Type()
	if mtyp.NumIn() == 0 || mtyp.In(0) != markerContextType {
		panic(fmt.Sprintf("marker function %#v must accept markerContext as its first argument", mi.fn))
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
	goldenType        = reflect.TypeOf(&Golden{})
	locationType      = reflect.TypeOf(protocol.Location{})
	markerContextType = reflect.TypeOf(&markerContext{})
	regexpType        = reflect.TypeOf(&regexp.Regexp{})
)

func makeConverter(paramType reflect.Type) converter {
	switch paramType {
	case goldenType:
		return goldenConverter
	case locationType:
		return locationConverter
	default:
		return func(_ *markerContext, _ *expect.Note, arg interface{}) (interface{}, error) {
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
func locationConverter(c *markerContext, note *expect.Note, arg interface{}) (interface{}, error) {
	switch arg := arg.(type) {
	case string:
		startOff, preceding, m, err := linePreceding(c, note.Pos)
		if err != nil {
			return protocol.Location{}, err
		}
		idx := bytes.Index(preceding, []byte(arg))
		if idx < 0 {
			return nil, fmt.Errorf("substring %q not found in %q", arg, preceding)
		}
		off := startOff + idx
		return m.OffsetLocation(off, off+len(arg))
	case *regexp.Regexp:
		return findRegexpInLine(c, note.Pos, arg)
	case expect.Identifier:
		loc, ok := c.locations[arg]
		if !ok {
			return nil, fmt.Errorf("no location named %q", arg)
		}
		return loc, nil
	default:
		return nil, fmt.Errorf("cannot convert argument type %T to location (must be a string to match the preceding line)", arg)
	}
}

// findRegexpInLine searches the partial line preceding pos for a match for the
// regular expression re, returning a location spanning the first match. If re
// contains exactly one subgroup, the position of this subgroup match is
// returned rather than the position of the full match.
func findRegexpInLine(c *markerContext, pos token.Pos, re *regexp.Regexp) (protocol.Location, error) {
	startOff, preceding, m, err := linePreceding(c, pos)
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

func linePreceding(c *markerContext, pos token.Pos) (int, []byte, *protocol.Mapper, error) {
	file := c.test.fset.File(pos)
	posn := safetoken.Position(file, pos)
	lineStart := file.LineStart(posn.Line)
	startOff, endOff, err := safetoken.Offsets(file, lineStart, pos)
	if err != nil {
		return 0, nil, nil, err
	}
	m := c.env.Editor.Mapper(file.Name())
	return startOff, m.Content[startOff:endOff], m, nil
}

// goldenConverter convers an identifier into the Golden directory of content
// prefixed by @<ident> in the test archive file.
func goldenConverter(c *markerContext, note *expect.Note, arg interface{}) (interface{}, error) {
	switch arg := arg.(type) {
	case expect.Identifier:
		golden := c.test.golden[string(arg)]
		// If there was no golden content for this identifier, we must create one
		// to handle the case where -update is set: we need a place to store
		// the updated content.
		if golden == nil {
			golden = new(Golden)
			c.test.golden[string(arg)] = golden
		}
		return golden, nil
	default:
		return nil, fmt.Errorf("invalid input type %T: golden key must be an identifier", arg)
	}
}

// defMarker implements the @godef marker, running textDocument/definition at
// the given src location and asserting that there is exactly one resulting
// location, matching dst.
//
// TODO(rfindley): support a variadic destination set.
func defMarker(c *markerContext, src, dst protocol.Location) {
	got := c.env.GoToDefinition(src)
	if got != dst {
		c.env.T.Errorf("%s: definition location does not match:\n\tgot: %s\n\twant %s", c.fmtLoc(src), c.fmtLoc(got), c.fmtLoc(dst))
	}
}

// hoverMarker implements the @hover marker, running textDocument/hover at the
// given src location and asserting that the resulting hover is over the dst
// location (typically a span surrounding src), and that the markdown content
// matches the golden content.
func hoverMarker(c *markerContext, src, dst protocol.Location, golden *Golden) {
	content, gotDst := c.env.Hover(src)
	if gotDst != dst {
		c.env.T.Errorf("%s: hover location does not match:\n\tgot: %s\n\twant %s)", c.fmtLoc(src), c.fmtLoc(gotDst), c.fmtLoc(dst))
	}
	gotMD := ""
	if content != nil {
		gotMD = content.Value
	}
	wantMD := ""
	if golden != nil {
		wantMD = string(golden.Get(c.env.T, "hover.md", func() []byte { return []byte(gotMD) }))
	}
	// Normalize newline termination: archive files can't express non-newline
	// terminated files.
	if strings.HasSuffix(wantMD, "\n") && !strings.HasSuffix(gotMD, "\n") {
		gotMD += "\n"
	}
	if diff := tests.DiffMarkdown(wantMD, gotMD); diff != "" {
		c.env.T.Errorf("%s: hover markdown mismatch (-want +got):\n%s", c.fmtLoc(src), diff)
	}
}

// locMarker implements the @loc hover marker. It is executed before other
// markers, so that locations are available.
func locMarker(c *markerContext, name expect.Identifier, loc protocol.Location) {
	c.locations[name] = loc
}

// diagMarker implements the @diag hover marker. It eliminates diagnostics from
// the observed set in the markerContext.
func diagMarker(c *markerContext, loc protocol.Location, re *regexp.Regexp) {
	idx := -1
	diags := c.diags[loc]
	for i, diag := range diags {
		if re.MatchString(diag.Message) {
			idx = i
			break
		}
	}
	if idx >= 0 {
		c.diags[loc] = append(diags[:idx], diags[idx+1:]...)
	} else {
		c.env.T.Errorf("%s: no diagnostic matches %q", c.fmtLoc(loc), re)
	}
}
