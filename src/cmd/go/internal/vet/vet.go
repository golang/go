// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package vet implements the “go vet” and “go fix” commands.
package vet

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"slices"
	"strconv"
	"strings"
	"sync"

	"cmd/go/internal/base"
	"cmd/go/internal/cfg"
	"cmd/go/internal/load"
	"cmd/go/internal/modload"
	"cmd/go/internal/trace"
	"cmd/go/internal/work"
)

var CmdVet = &base.Command{
	CustomFlags: true,
	UsageLine:   "go vet [build flags] [-vettool prog] [vet flags] [packages]",
	Short:       "report likely mistakes in packages",
	Long: `
Vet runs the Go vet tool (cmd/vet) on the named packages
and reports diagnostics.

It supports these flags:

  -c int
	display offending line with this many lines of context (default -1)
  -json
	emit JSON output
  -fix
	instead of printing each diagnostic, apply its first fix (if any)
  -diff
	instead of applying each fix, print the patch as a unified diff

The -vettool=prog flag selects a different analysis tool with
alternative or additional checks. For example, the 'shadow' analyzer
can be built and run using these commands:

  go install golang.org/x/tools/go/analysis/passes/shadow/cmd/shadow@latest
  go vet -vettool=$(which shadow)

Alternative vet tools should be built atop golang.org/x/tools/go/analysis/unitchecker,
which handles the interaction with go vet.

For more about specifying packages, see 'go help packages'.
For a list of checkers and their flags, see 'go tool vet help'.
For details of a specific checker such as 'printf', see 'go tool vet help printf'.

The build flags supported by go vet are those that control package resolution
and execution, such as -C, -n, -x, -v, -tags, and -toolexec.
For more about these flags, see 'go help build'.

See also: go fmt, go fix.
	`,
}

var CmdFix = &base.Command{
	CustomFlags: true,
	UsageLine:   "go fix [build flags] [-fixtool prog] [fix flags] [packages]",
	Short:       "apply fixes suggested by static checkers",
	Long: `
Fix runs the Go fix tool (cmd/vet) on the named packages
and applies suggested fixes.

It supports these flags:

  -diff
	instead of applying each fix, print the patch as a unified diff

The -fixtool=prog flag selects a different analysis tool with
alternative or additional fixes; see the documentation for go vet's
-vettool flag for details.

For more about specifying packages, see 'go help packages'.

For a list of fixers and their flags, see 'go tool fix help'.

For details of a specific fixer such as 'hostport',
see 'go tool fix help hostport'.

The build flags supported by go fix are those that control package resolution
and execution, such as -C, -n, -x, -v, -tags, and -toolexec.
For more about these flags, see 'go help build'.

See also: go fmt, go vet.
	`,
}

func init() {
	// avoid initialization cycle
	CmdVet.Run = run
	CmdFix.Run = run

	addFlags(CmdVet)
	addFlags(CmdFix)
}

var (
	// "go vet -fix" causes fixes to be applied.
	vetFixFlag = CmdVet.Flag.Bool("fix", false, "apply the first fix (if any) for each diagnostic")

	// The "go fix -fix=name,..." flag is an obsolete flag formerly
	// used to pass a list of names to the old "cmd/fix -r".
	fixFixFlag = CmdFix.Flag.String("fix", "", "obsolete; no effect")
)

// run implements both "go vet" and "go fix".
func run(ctx context.Context, cmd *base.Command, args []string) {
	// Compute flags for the vet/fix tool (e.g. cmd/{vet,fix}).
	toolFlags, pkgArgs := toolFlags(cmd, args)

	// The vet/fix commands do custom flag processing;
	// initialize workspaces after that.
	modload.InitWorkfile()

	if cfg.DebugTrace != "" {
		var close func() error
		var err error
		ctx, close, err = trace.Start(ctx, cfg.DebugTrace)
		if err != nil {
			base.Fatalf("failed to start trace: %v", err)
		}
		defer func() {
			if err := close(); err != nil {
				base.Fatalf("failed to stop trace: %v", err)
			}
		}()
	}

	ctx, span := trace.StartSpan(ctx, fmt.Sprint("Running ", cmd.Name(), " command"))
	defer span.Done()

	work.BuildInit()

	// Flag theory:
	//
	// All flags supported by unitchecker are accepted by go {vet,fix}.
	// Some arise from each analyzer in the tool (both to enable it
	// and to configure it), whereas others [-V -c -diff -fix -flags -json]
	// are core to unitchecker itself.
	//
	// Most are passed through to toolFlags, but not all:
	// * -V and -flags are used by the handshake in the [toolFlags] function;
	// * these old flags have no effect: [-all -source -tags -v]; and
	// * the [-c -fix -diff -json] flags are handled specially
	//   as described below:
	//
	// command args                 tool args
	// go vet               =>      cmd/vet -json           Parse stdout, print diagnostics to stderr.
	// go vet -json         =>      cmd/vet -json           Pass stdout through.
	// go vet -fix [-diff]  =>      cmd/vet -fix [-diff]    Pass stdout through.
	// go fix [-diff]       =>      cmd/fix -fix [-diff]    Pass stdout through.
	// go fix -json         =>      cmd/fix -json           Pass stdout through.
	//
	// Notes:
	// * -diff requires "go vet -fix" or "go fix", and no -json.
	// * -json output is the same in "vet" and "fix" modes,
	//   and describes both diagnostics and fixes (but does not apply them).
	// * -c=n is supported by the unitchecker, but we reimplement it
	//   here (see printDiagnostics), and do not pass the flag through.

	work.VetExplicit = len(toolFlags) > 0

	if cmd.Name() == "fix" || *vetFixFlag {
		// fix mode: 'go fix' or 'go vet -fix'
		if jsonFlag {
			if diffFlag {
				base.Fatalf("-json and -diff cannot be used together")
			}
		} else {
			toolFlags = append(toolFlags, "-fix")
			if diffFlag {
				toolFlags = append(toolFlags, "-diff")
			}
		}
		if contextFlag != -1 {
			base.Fatalf("-c flag cannot be used when applying fixes")
		}
	} else {
		// vet mode: 'go vet' without -fix
		if !jsonFlag {
			// Post-process the JSON diagnostics on stdout and format
			// it as "file:line: message" diagnostics on stderr.
			// (JSON reliably frames diagnostics, fixes, and errors so
			// that we don't have to parse stderr or interpret non-zero
			// exit codes, and interacts better with the action cache.)
			toolFlags = append(toolFlags, "-json")
			work.VetHandleStdout = printJSONDiagnostics
		}
		if diffFlag {
			base.Fatalf("go vet -diff flag requires -fix")
		}
	}

	// Implement legacy "go fix -fix=name,..." flag.
	if *fixFixFlag != "" {
		fmt.Fprintf(os.Stderr, "go %s: the -fix=%s flag is obsolete and has no effect", cmd.Name(), *fixFixFlag)

		// The buildtag fixer is now implemented by cmd/fix.
		if slices.Contains(strings.Split(*fixFixFlag, ","), "buildtag") {
			fmt.Fprintf(os.Stderr, "go %s: to enable the buildtag check, use -buildtag", cmd.Name())
		}
	}

	work.VetFlags = toolFlags

	pkgOpts := load.PackageOpts{ModResolveTests: true}
	pkgs := load.PackagesAndErrors(ctx, pkgOpts, pkgArgs)
	load.CheckPackageErrors(pkgs)
	if len(pkgs) == 0 {
		base.Fatalf("no packages to %s", cmd.Name())
	}

	b := work.NewBuilder("")
	defer func() {
		if err := b.Close(); err != nil {
			base.Fatal(err)
		}
	}()

	// To avoid file corruption from duplicate application of
	// fixes (in fix mode), and duplicate reporting of diagnostics
	// (in vet mode), we must run the tool only once for each
	// source file. We achieve that by running on ptest (below)
	// instead of p.
	//
	// As a side benefit, this also allows analyzers to make
	// "closed world" assumptions and report diagnostics (such as
	// "this symbol is unused") that might be false if computed
	// from just the primary package p, falsified by the
	// additional declarations in test files.
	//
	// We needn't worry about intermediate test variants, as they
	// will only be executed in VetxOnly mode, for facts but not
	// diagnostics.

	root := &work.Action{Mode: "go " + cmd.Name()}
	for _, p := range pkgs {
		_, ptest, pxtest, perr := load.TestPackagesFor(ctx, pkgOpts, p, nil)
		if perr != nil {
			base.Errorf("%v", perr.Error)
			continue
		}
		if len(ptest.GoFiles) == 0 && len(ptest.CgoFiles) == 0 && pxtest == nil {
			base.Errorf("go: can't %s %s: no Go files in %s", cmd.Name(), p.ImportPath, p.Dir)
			continue
		}
		if len(ptest.GoFiles) > 0 || len(ptest.CgoFiles) > 0 {
			// The test package includes all the files of primary package.
			root.Deps = append(root.Deps, b.VetAction(work.ModeBuild, work.ModeBuild, ptest))
		}
		if pxtest != nil {
			root.Deps = append(root.Deps, b.VetAction(work.ModeBuild, work.ModeBuild, pxtest))
		}
	}
	b.Do(ctx, root)
}

// printJSONDiagnostics parses JSON (from the tool's stdout) and
// prints it (to stderr) in "file:line: message" form.
// It also ensures that we exit nonzero if there were diagnostics.
func printJSONDiagnostics(r io.Reader) error {
	stdout, err := io.ReadAll(r)
	if err != nil {
		return err
	}
	if len(stdout) > 0 {
		// unitchecker emits a JSON map of the form:
		// output maps Package ID -> Analyzer.Name -> (error | []Diagnostic);
		var tree jsonTree
		if err := json.Unmarshal([]byte(stdout), &tree); err != nil {
			return fmt.Errorf("parsing JSON: %v", err)
		}
		for _, units := range tree {
			for analyzer, msg := range units {
				if msg[0] == '[' {
					// []Diagnostic
					var diags []jsonDiagnostic
					if err := json.Unmarshal([]byte(msg), &diags); err != nil {
						return fmt.Errorf("parsing JSON diagnostics: %v", err)
					}
					for _, diag := range diags {
						base.SetExitStatus(1)
						printJSONDiagnostic(analyzer, diag)
					}
				} else {
					// error
					var e jsonError
					if err := json.Unmarshal([]byte(msg), &e); err != nil {
						return fmt.Errorf("parsing JSON error: %v", err)
					}

					base.SetExitStatus(1)
					return errors.New(e.Err)
				}
			}
		}
	}
	return nil
}

var stderrMu sync.Mutex // serializes concurrent writes to stdout

func printJSONDiagnostic(analyzer string, diag jsonDiagnostic) {
	stderrMu.Lock()
	defer stderrMu.Unlock()

	type posn struct {
		file      string
		line, col int
	}
	parsePosn := func(s string) (_ posn, _ bool) {
		colon2 := strings.LastIndexByte(s, ':')
		if colon2 < 0 {
			return
		}
		colon1 := strings.LastIndexByte(s[:colon2], ':')
		if colon1 < 0 {
			return
		}
		line, err := strconv.Atoi(s[colon1+len(":") : colon2])
		if err != nil {
			return
		}
		col, err := strconv.Atoi(s[colon2+len(":"):])
		if err != nil {
			return
		}
		return posn{s[:colon1], line, col}, true
	}

	print := func(start, end, message string) {
		if posn, ok := parsePosn(start); ok {
			// The (*work.Shell).reportCmd method relativizes the
			// prefix of each line of the subprocess's stdout;
			// but filenames in JSON aren't at the start of the line,
			// so we need to apply ShortPath here too.
			fmt.Fprintf(os.Stderr, "%s:%d:%d: %v\n", base.ShortPath(posn.file), posn.line, posn.col, message)
		} else {
			fmt.Fprintf(os.Stderr, "%s: %v\n", start, message)
		}

		// -c=n: show offending line plus N lines of context.
		// (Duplicates logic in unitchecker; see analysisflags.PrintPlain.)
		if contextFlag >= 0 {
			if end == "" {
				end = start
			}
			var (
				startPosn, ok1 = parsePosn(start)
				endPosn, ok2   = parsePosn(end)
			)
			if ok1 && ok2 {
				// TODO(adonovan): respect overlays (like unitchecker does).
				data, _ := os.ReadFile(startPosn.file)
				lines := strings.Split(string(data), "\n")
				for i := startPosn.line - contextFlag; i <= endPosn.line+contextFlag; i++ {
					if 1 <= i && i <= len(lines) {
						fmt.Fprintf(os.Stderr, "%d\t%s\n", i, lines[i-1])
					}
				}
			}
		}
	}

	// TODO(adonovan): append  " [analyzer]" to message. But we must first relax
	// x/tools/go/analysis/internal/versiontest.TestVettool and revendor; sigh.
	_ = analyzer
	print(diag.Posn, diag.End, diag.Message)
	for _, rel := range diag.Related {
		print(rel.Posn, rel.End, "\t"+rel.Message)
	}
}

// -- JSON schema --

// (populated by golang.org/x/tools/go/analysis/internal/analysisflags/flags.go)

// A jsonTree is a mapping from package ID to analysis name to result.
// Each result is either a jsonError or a list of jsonDiagnostic.
type jsonTree map[string]map[string]json.RawMessage

type jsonError struct {
	Err string `json:"error"`
}

// A TextEdit describes the replacement of a portion of a file.
// Start and End are zero-based half-open indices into the original byte
// sequence of the file, and New is the new text.
type jsonTextEdit struct {
	Filename string `json:"filename"`
	Start    int    `json:"start"`
	End      int    `json:"end"`
	New      string `json:"new"`
}

// A jsonSuggestedFix describes an edit that should be applied as a whole or not
// at all. It might contain multiple TextEdits/text_edits if the SuggestedFix
// consists of multiple non-contiguous edits.
type jsonSuggestedFix struct {
	Message string         `json:"message"`
	Edits   []jsonTextEdit `json:"edits"`
}

// A jsonDiagnostic describes the json schema of an analysis.Diagnostic.
type jsonDiagnostic struct {
	Category       string                   `json:"category,omitempty"`
	Posn           string                   `json:"posn"` // e.g. "file.go:line:column"
	End            string                   `json:"end"`
	Message        string                   `json:"message"`
	SuggestedFixes []jsonSuggestedFix       `json:"suggested_fixes,omitempty"`
	Related        []jsonRelatedInformation `json:"related,omitempty"`
}

// A jsonRelated describes a secondary position and message related to
// a primary diagnostic.
type jsonRelatedInformation struct {
	Posn    string `json:"posn"` // e.g. "file.go:line:column"
	End     string `json:"end"`
	Message string `json:"message"`
}
