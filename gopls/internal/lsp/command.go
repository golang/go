// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lsp

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"runtime/pprof"
	"sort"
	"strings"

	"golang.org/x/mod/modfile"
	"golang.org/x/tools/go/ast/astutil"
	"golang.org/x/tools/gopls/internal/bug"
	"golang.org/x/tools/gopls/internal/lsp/cache"
	"golang.org/x/tools/gopls/internal/lsp/command"
	"golang.org/x/tools/gopls/internal/lsp/debug"
	"golang.org/x/tools/gopls/internal/lsp/progress"
	"golang.org/x/tools/gopls/internal/lsp/protocol"
	"golang.org/x/tools/gopls/internal/lsp/source"
	"golang.org/x/tools/gopls/internal/span"
	"golang.org/x/tools/gopls/internal/telemetry"
	"golang.org/x/tools/gopls/internal/vulncheck"
	"golang.org/x/tools/gopls/internal/vulncheck/scan"
	"golang.org/x/tools/internal/event"
	"golang.org/x/tools/internal/gocommand"
	"golang.org/x/tools/internal/tokeninternal"
	"golang.org/x/tools/internal/xcontext"
)

func (s *Server) executeCommand(ctx context.Context, params *protocol.ExecuteCommandParams) (interface{}, error) {
	ctx, done := event.Start(ctx, "lsp.Server.executeCommand")
	defer done()

	var found bool
	for _, name := range s.Options().SupportedCommands {
		if name == params.Command {
			found = true
			break
		}
	}
	if !found {
		return nil, fmt.Errorf("%s is not a supported command", params.Command)
	}

	handler := &commandHandler{
		s:      s,
		params: params,
	}
	return command.Dispatch(ctx, params, handler)
}

type commandHandler struct {
	s      *Server
	params *protocol.ExecuteCommandParams
}

func (h *commandHandler) MaybePromptForTelemetry(ctx context.Context) error {
	go h.s.maybePromptForTelemetry(ctx, true)
	return nil
}

func (*commandHandler) AddTelemetryCounters(_ context.Context, args command.AddTelemetryCountersArgs) error {
	if len(args.Names) != len(args.Values) {
		return fmt.Errorf("Names and Values must have the same length")
	}
	// invalid counter update requests will be silently dropped. (no audience)
	telemetry.AddForwardedCounters(args.Names, args.Values)
	return nil
}

// commandConfig configures common command set-up and execution.
type commandConfig struct {
	async       bool                 // whether to run the command asynchronously. Async commands can only return errors.
	requireSave bool                 // whether all files must be saved for the command to work
	progress    string               // title to use for progress reporting. If empty, no progress will be reported.
	forView     string               // view to resolve to a snapshot; incompatible with forURI
	forURI      protocol.DocumentURI // URI to resolve to a snapshot. If unset, snapshot will be nil.
}

// commandDeps is evaluated from a commandConfig. Note that not all fields may
// be populated, depending on which configuration is set. See comments in-line
// for details.
type commandDeps struct {
	snapshot source.Snapshot    // present if cfg.forURI was set
	fh       source.FileHandle  // present if cfg.forURI was set
	work     *progress.WorkDone // present cfg.progress was set
}

type commandFunc func(context.Context, commandDeps) error

// run performs command setup for command execution, and invokes the given run
// function. If cfg.async is set, run executes the given func in a separate
// goroutine, and returns as soon as setup is complete and the goroutine is
// scheduled.
//
// Invariant: if the resulting error is non-nil, the given run func will
// (eventually) be executed exactly once.
func (c *commandHandler) run(ctx context.Context, cfg commandConfig, run commandFunc) (err error) {
	if cfg.requireSave {
		var unsaved []string
		for _, overlay := range c.s.session.Overlays() {
			if !overlay.SameContentsOnDisk() {
				unsaved = append(unsaved, overlay.URI().Filename())
			}
		}
		if len(unsaved) > 0 {
			return fmt.Errorf("All files must be saved first (unsaved: %v).", unsaved)
		}
	}
	var deps commandDeps
	if cfg.forURI != "" && cfg.forView != "" {
		return bug.Errorf("internal error: forURI=%q, forView=%q", cfg.forURI, cfg.forView)
	}
	if cfg.forURI != "" {
		var ok bool
		var release func()
		deps.snapshot, deps.fh, ok, release, err = c.s.beginFileRequest(ctx, cfg.forURI, source.UnknownKind)
		defer release()
		if !ok {
			if err != nil {
				return err
			}
			return fmt.Errorf("invalid file URL: %v", cfg.forURI)
		}
	} else if cfg.forView != "" {
		view, err := c.s.session.View(cfg.forView)
		if err != nil {
			return err
		}
		var release func()
		deps.snapshot, release, err = view.Snapshot()
		if err != nil {
			return err
		}
		defer release()
	}
	ctx, cancel := context.WithCancel(xcontext.Detach(ctx))
	if cfg.progress != "" {
		deps.work = c.s.progress.Start(ctx, cfg.progress, "Running...", c.params.WorkDoneToken, cancel)
	}
	runcmd := func() error {
		defer cancel()
		err := run(ctx, deps)
		if deps.work != nil {
			switch {
			case errors.Is(err, context.Canceled):
				deps.work.End(ctx, "canceled")
			case err != nil:
				event.Error(ctx, "command error", err)
				deps.work.End(ctx, "failed")
			default:
				deps.work.End(ctx, "completed")
			}
		}
		return err
	}
	if cfg.async {
		go func() {
			if err := runcmd(); err != nil {
				if showMessageErr := c.s.client.ShowMessage(ctx, &protocol.ShowMessageParams{
					Type:    protocol.Error,
					Message: err.Error(),
				}); showMessageErr != nil {
					event.Error(ctx, fmt.Sprintf("failed to show message: %q", err.Error()), showMessageErr)
				}
			}
		}()
		return nil
	}
	return runcmd()
}

func (c *commandHandler) ApplyFix(ctx context.Context, args command.ApplyFixArgs) error {
	return c.run(ctx, commandConfig{
		// Note: no progress here. Applying fixes should be quick.
		forURI: args.URI,
	}, func(ctx context.Context, deps commandDeps) error {
		edits, err := source.ApplyFix(ctx, args.Fix, deps.snapshot, deps.fh, args.Range)
		if err != nil {
			return err
		}
		changes := []protocol.DocumentChanges{} // must be a slice
		for _, edit := range edits {
			edit := edit
			changes = append(changes, protocol.DocumentChanges{
				TextDocumentEdit: &edit,
			})
		}
		r, err := c.s.client.ApplyEdit(ctx, &protocol.ApplyWorkspaceEditParams{
			Edit: protocol.WorkspaceEdit{
				DocumentChanges: changes,
			},
		})
		if err != nil {
			return err
		}
		if !r.Applied {
			return errors.New(r.FailureReason)
		}
		return nil
	})
}

func (c *commandHandler) RegenerateCgo(ctx context.Context, args command.URIArg) error {
	return c.run(ctx, commandConfig{
		progress: "Regenerating Cgo",
	}, func(ctx context.Context, deps commandDeps) error {
		mod := source.FileModification{
			URI:    args.URI.SpanURI(),
			Action: source.InvalidateMetadata,
		}
		return c.s.didModifyFiles(ctx, []source.FileModification{mod}, FromRegenerateCgo)
	})
}

func (c *commandHandler) CheckUpgrades(ctx context.Context, args command.CheckUpgradesArgs) error {
	return c.run(ctx, commandConfig{
		forURI:   args.URI,
		progress: "Checking for upgrades",
	}, func(ctx context.Context, deps commandDeps) error {
		upgrades, err := c.s.getUpgrades(ctx, deps.snapshot, args.URI.SpanURI(), args.Modules)
		if err != nil {
			return err
		}
		deps.snapshot.View().RegisterModuleUpgrades(args.URI.SpanURI(), upgrades)
		// Re-diagnose the snapshot to publish the new module diagnostics.
		c.s.diagnoseSnapshot(deps.snapshot, nil, false, 0)
		return nil
	})
}

func (c *commandHandler) AddDependency(ctx context.Context, args command.DependencyArgs) error {
	return c.GoGetModule(ctx, args)
}

func (c *commandHandler) UpgradeDependency(ctx context.Context, args command.DependencyArgs) error {
	return c.GoGetModule(ctx, args)
}

func (c *commandHandler) ResetGoModDiagnostics(ctx context.Context, args command.ResetGoModDiagnosticsArgs) error {
	return c.run(ctx, commandConfig{
		forURI: args.URI,
	}, func(ctx context.Context, deps commandDeps) error {
		// Clear all diagnostics coming from the upgrade check source and vulncheck.
		// This will clear the diagnostics in all go.mod files, but they
		// will be re-calculated when the snapshot is diagnosed again.
		if args.DiagnosticSource == "" || args.DiagnosticSource == string(source.UpgradeNotification) {
			deps.snapshot.View().ClearModuleUpgrades(args.URI.SpanURI())
			c.s.clearDiagnosticSource(modCheckUpgradesSource)
		}

		if args.DiagnosticSource == "" || args.DiagnosticSource == string(source.Govulncheck) {
			deps.snapshot.View().SetVulnerabilities(args.URI.SpanURI(), nil)
			c.s.clearDiagnosticSource(modVulncheckSource)
		}

		// Re-diagnose the snapshot to remove the diagnostics.
		c.s.diagnoseSnapshot(deps.snapshot, nil, false, 0)
		return nil
	})
}

func (c *commandHandler) GoGetModule(ctx context.Context, args command.DependencyArgs) error {
	return c.run(ctx, commandConfig{
		progress: "Running go get",
		forURI:   args.URI,
	}, func(ctx context.Context, deps commandDeps) error {
		return c.s.runGoModUpdateCommands(ctx, deps.snapshot, args.URI.SpanURI(), func(invoke func(...string) (*bytes.Buffer, error)) error {
			return runGoGetModule(invoke, args.AddRequire, args.GoCmdArgs)
		})
	})
}

// TODO(rFindley): UpdateGoSum, Tidy, and Vendor could probably all be one command.
func (c *commandHandler) UpdateGoSum(ctx context.Context, args command.URIArgs) error {
	return c.run(ctx, commandConfig{
		progress: "Updating go.sum",
	}, func(ctx context.Context, deps commandDeps) error {
		for _, uri := range args.URIs {
			snapshot, fh, ok, release, err := c.s.beginFileRequest(ctx, uri, source.UnknownKind)
			defer release()
			if !ok {
				return err
			}
			if err := c.s.runGoModUpdateCommands(ctx, snapshot, fh.URI(), func(invoke func(...string) (*bytes.Buffer, error)) error {
				_, err := invoke("list", "all")
				return err
			}); err != nil {
				return err
			}
		}
		return nil
	})
}

func (c *commandHandler) Tidy(ctx context.Context, args command.URIArgs) error {
	return c.run(ctx, commandConfig{
		requireSave: true,
		progress:    "Running go mod tidy",
	}, func(ctx context.Context, deps commandDeps) error {
		for _, uri := range args.URIs {
			snapshot, fh, ok, release, err := c.s.beginFileRequest(ctx, uri, source.UnknownKind)
			defer release()
			if !ok {
				return err
			}
			if err := c.s.runGoModUpdateCommands(ctx, snapshot, fh.URI(), func(invoke func(...string) (*bytes.Buffer, error)) error {
				_, err := invoke("mod", "tidy")
				return err
			}); err != nil {
				return err
			}
		}
		return nil
	})
}

func (c *commandHandler) Vendor(ctx context.Context, args command.URIArg) error {
	return c.run(ctx, commandConfig{
		requireSave: true,
		progress:    "Running go mod vendor",
		forURI:      args.URI,
	}, func(ctx context.Context, deps commandDeps) error {
		// Use RunGoCommandPiped here so that we don't compete with any other go
		// command invocations. go mod vendor deletes modules.txt before recreating
		// it, and therefore can run into file locking issues on Windows if that
		// file is in use by another process, such as go list.
		//
		// If golang/go#44119 is resolved, go mod vendor will instead modify
		// modules.txt in-place. In that case we could theoretically allow this
		// command to run concurrently.
		err := deps.snapshot.RunGoCommandPiped(ctx, source.Normal|source.AllowNetwork, &gocommand.Invocation{
			Verb:       "mod",
			Args:       []string{"vendor"},
			WorkingDir: filepath.Dir(args.URI.SpanURI().Filename()),
		}, &bytes.Buffer{}, &bytes.Buffer{})
		return err
	})
}

func (c *commandHandler) EditGoDirective(ctx context.Context, args command.EditGoDirectiveArgs) error {
	return c.run(ctx, commandConfig{
		requireSave: true, // if go.mod isn't saved it could cause a problem
		forURI:      args.URI,
	}, func(ctx context.Context, deps commandDeps) error {
		snapshot, fh, ok, release, err := c.s.beginFileRequest(ctx, args.URI, source.UnknownKind)
		defer release()
		if !ok {
			return err
		}
		if err := c.s.runGoModUpdateCommands(ctx, snapshot, fh.URI(), func(invoke func(...string) (*bytes.Buffer, error)) error {
			_, err := invoke("mod", "edit", "-go", args.Version)
			return err
		}); err != nil {
			return err
		}
		return nil
	})
}

func (c *commandHandler) RemoveDependency(ctx context.Context, args command.RemoveDependencyArgs) error {
	return c.run(ctx, commandConfig{
		progress: "Removing dependency",
		forURI:   args.URI,
	}, func(ctx context.Context, deps commandDeps) error {
		// See the documentation for OnlyDiagnostic.
		//
		// TODO(rfindley): In Go 1.17+, we will be able to use the go command
		// without checking if the module is tidy.
		if args.OnlyDiagnostic {
			return c.s.runGoModUpdateCommands(ctx, deps.snapshot, args.URI.SpanURI(), func(invoke func(...string) (*bytes.Buffer, error)) error {
				if err := runGoGetModule(invoke, false, []string{args.ModulePath + "@none"}); err != nil {
					return err
				}
				_, err := invoke("mod", "tidy")
				return err
			})
		}
		pm, err := deps.snapshot.ParseMod(ctx, deps.fh)
		if err != nil {
			return err
		}
		edits, err := dropDependency(deps.snapshot, pm, args.ModulePath)
		if err != nil {
			return err
		}
		response, err := c.s.client.ApplyEdit(ctx, &protocol.ApplyWorkspaceEditParams{
			Edit: protocol.WorkspaceEdit{
				DocumentChanges: []protocol.DocumentChanges{
					{
						TextDocumentEdit: &protocol.TextDocumentEdit{
							TextDocument: protocol.OptionalVersionedTextDocumentIdentifier{
								Version: deps.fh.Version(),
								TextDocumentIdentifier: protocol.TextDocumentIdentifier{
									URI: protocol.URIFromSpanURI(deps.fh.URI()),
								},
							},
							Edits: nonNilSliceTextEdit(edits),
						},
					},
				},
			},
		})
		if err != nil {
			return err
		}
		if !response.Applied {
			return fmt.Errorf("edits not applied because of %s", response.FailureReason)
		}
		return nil
	})
}

// dropDependency returns the edits to remove the given require from the go.mod
// file.
func dropDependency(snapshot source.Snapshot, pm *source.ParsedModule, modulePath string) ([]protocol.TextEdit, error) {
	// We need a private copy of the parsed go.mod file, since we're going to
	// modify it.
	copied, err := modfile.Parse("", pm.Mapper.Content, nil)
	if err != nil {
		return nil, err
	}
	if err := copied.DropRequire(modulePath); err != nil {
		return nil, err
	}
	copied.Cleanup()
	newContent, err := copied.Format()
	if err != nil {
		return nil, err
	}
	// Calculate the edits to be made due to the change.
	diff := snapshot.Options().ComputeEdits(string(pm.Mapper.Content), string(newContent))
	return source.ToProtocolEdits(pm.Mapper, diff)
}

func (c *commandHandler) Test(ctx context.Context, uri protocol.DocumentURI, tests, benchmarks []string) error {
	return c.RunTests(ctx, command.RunTestsArgs{
		URI:        uri,
		Tests:      tests,
		Benchmarks: benchmarks,
	})
}

func (c *commandHandler) RunTests(ctx context.Context, args command.RunTestsArgs) error {
	return c.run(ctx, commandConfig{
		async:       true,
		progress:    "Running go test",
		requireSave: true,
		forURI:      args.URI,
	}, func(ctx context.Context, deps commandDeps) error {
		if err := c.runTests(ctx, deps.snapshot, deps.work, args.URI, args.Tests, args.Benchmarks); err != nil {
			return fmt.Errorf("running tests failed: %w", err)
		}
		return nil
	})
}

func (c *commandHandler) runTests(ctx context.Context, snapshot source.Snapshot, work *progress.WorkDone, uri protocol.DocumentURI, tests, benchmarks []string) error {
	// TODO: fix the error reporting when this runs async.
	meta, err := source.NarrowestMetadataForFile(ctx, snapshot, uri.SpanURI())
	if err != nil {
		return err
	}
	pkgPath := string(meta.ForTest)

	// create output
	buf := &bytes.Buffer{}
	ew := progress.NewEventWriter(ctx, "test")
	out := io.MultiWriter(ew, progress.NewWorkDoneWriter(ctx, work), buf)

	// Run `go test -run Func` on each test.
	var failedTests int
	for _, funcName := range tests {
		inv := &gocommand.Invocation{
			Verb:       "test",
			Args:       []string{pkgPath, "-v", "-count=1", "-run", fmt.Sprintf("^%s$", funcName)},
			WorkingDir: filepath.Dir(uri.SpanURI().Filename()),
		}
		if err := snapshot.RunGoCommandPiped(ctx, source.Normal, inv, out, out); err != nil {
			if errors.Is(err, context.Canceled) {
				return err
			}
			failedTests++
		}
	}

	// Run `go test -run=^$ -bench Func` on each test.
	var failedBenchmarks int
	for _, funcName := range benchmarks {
		inv := &gocommand.Invocation{
			Verb:       "test",
			Args:       []string{pkgPath, "-v", "-run=^$", "-bench", fmt.Sprintf("^%s$", funcName)},
			WorkingDir: filepath.Dir(uri.SpanURI().Filename()),
		}
		if err := snapshot.RunGoCommandPiped(ctx, source.Normal, inv, out, out); err != nil {
			if errors.Is(err, context.Canceled) {
				return err
			}
			failedBenchmarks++
		}
	}

	var title string
	if len(tests) > 0 && len(benchmarks) > 0 {
		title = "tests and benchmarks"
	} else if len(tests) > 0 {
		title = "tests"
	} else if len(benchmarks) > 0 {
		title = "benchmarks"
	} else {
		return errors.New("No functions were provided")
	}
	message := fmt.Sprintf("all %s passed", title)
	if failedTests > 0 && failedBenchmarks > 0 {
		message = fmt.Sprintf("%d / %d tests failed and %d / %d benchmarks failed", failedTests, len(tests), failedBenchmarks, len(benchmarks))
	} else if failedTests > 0 {
		message = fmt.Sprintf("%d / %d tests failed", failedTests, len(tests))
	} else if failedBenchmarks > 0 {
		message = fmt.Sprintf("%d / %d benchmarks failed", failedBenchmarks, len(benchmarks))
	}
	if failedTests > 0 || failedBenchmarks > 0 {
		message += "\n" + buf.String()
	}

	return c.s.client.ShowMessage(ctx, &protocol.ShowMessageParams{
		Type:    protocol.Info,
		Message: message,
	})
}

func (c *commandHandler) Generate(ctx context.Context, args command.GenerateArgs) error {
	title := "Running go generate ."
	if args.Recursive {
		title = "Running go generate ./..."
	}
	return c.run(ctx, commandConfig{
		requireSave: true,
		progress:    title,
		forURI:      args.Dir,
	}, func(ctx context.Context, deps commandDeps) error {
		er := progress.NewEventWriter(ctx, "generate")

		pattern := "."
		if args.Recursive {
			pattern = "./..."
		}
		inv := &gocommand.Invocation{
			Verb:       "generate",
			Args:       []string{"-x", pattern},
			WorkingDir: args.Dir.SpanURI().Filename(),
		}
		stderr := io.MultiWriter(er, progress.NewWorkDoneWriter(ctx, deps.work))
		if err := deps.snapshot.RunGoCommandPiped(ctx, source.Normal, inv, er, stderr); err != nil {
			return err
		}
		return nil
	})
}

func (c *commandHandler) GoGetPackage(ctx context.Context, args command.GoGetPackageArgs) error {
	return c.run(ctx, commandConfig{
		forURI:   args.URI,
		progress: "Running go get",
	}, func(ctx context.Context, deps commandDeps) error {
		// Run on a throwaway go.mod, otherwise it'll write to the real one.
		stdout, err := deps.snapshot.RunGoCommandDirect(ctx, source.WriteTemporaryModFile|source.AllowNetwork, &gocommand.Invocation{
			Verb:       "list",
			Args:       []string{"-f", "{{.Module.Path}}@{{.Module.Version}}", args.Pkg},
			WorkingDir: filepath.Dir(args.URI.SpanURI().Filename()),
		})
		if err != nil {
			return err
		}
		ver := strings.TrimSpace(stdout.String())
		return c.s.runGoModUpdateCommands(ctx, deps.snapshot, args.URI.SpanURI(), func(invoke func(...string) (*bytes.Buffer, error)) error {
			if args.AddRequire {
				if err := addModuleRequire(invoke, []string{ver}); err != nil {
					return err
				}
			}
			_, err := invoke(append([]string{"get", "-d"}, args.Pkg)...)
			return err
		})
	})
}

func (s *Server) runGoModUpdateCommands(ctx context.Context, snapshot source.Snapshot, uri span.URI, run func(invoke func(...string) (*bytes.Buffer, error)) error) error {
	tmpModfile, newModBytes, newSumBytes, err := snapshot.RunGoCommands(ctx, true, filepath.Dir(uri.Filename()), run)
	if err != nil {
		return err
	}
	if !tmpModfile {
		return nil
	}
	modURI := snapshot.GoModForFile(uri)
	sumURI := span.URIFromPath(strings.TrimSuffix(modURI.Filename(), ".mod") + ".sum")
	modEdits, err := collectFileEdits(ctx, snapshot, modURI, newModBytes)
	if err != nil {
		return err
	}
	sumEdits, err := collectFileEdits(ctx, snapshot, sumURI, newSumBytes)
	if err != nil {
		return err
	}
	return applyFileEdits(ctx, s.client, append(sumEdits, modEdits...))
}

// collectFileEdits collects any file edits required to transform the snapshot
// file specified by uri to the provided new content.
//
// If the file is not open, collectFileEdits simply writes the new content to
// disk.
//
// TODO(rfindley): fix this API asymmetry. It should be up to the caller to
// write the file or apply the edits.
func collectFileEdits(ctx context.Context, snapshot source.Snapshot, uri span.URI, newContent []byte) ([]protocol.TextDocumentEdit, error) {
	fh, err := snapshot.ReadFile(ctx, uri)
	if err != nil {
		return nil, err
	}
	oldContent, err := fh.Content()
	if err != nil && !os.IsNotExist(err) {
		return nil, err
	}

	if bytes.Equal(oldContent, newContent) {
		return nil, nil
	}

	// Sending a workspace edit to a closed file causes VS Code to open the
	// file and leave it unsaved. We would rather apply the changes directly,
	// especially to go.sum, which should be mostly invisible to the user.
	if !snapshot.IsOpen(uri) {
		err := os.WriteFile(uri.Filename(), newContent, 0666)
		return nil, err
	}

	m := protocol.NewMapper(fh.URI(), oldContent)
	diff := snapshot.Options().ComputeEdits(string(oldContent), string(newContent))
	edits, err := source.ToProtocolEdits(m, diff)
	if err != nil {
		return nil, err
	}
	return []protocol.TextDocumentEdit{{
		TextDocument: protocol.OptionalVersionedTextDocumentIdentifier{
			Version: fh.Version(),
			TextDocumentIdentifier: protocol.TextDocumentIdentifier{
				URI: protocol.URIFromSpanURI(uri),
			},
		},
		Edits: edits,
	}}, nil
}

func applyFileEdits(ctx context.Context, cli protocol.Client, edits []protocol.TextDocumentEdit) error {
	if len(edits) == 0 {
		return nil
	}
	documentChanges := []protocol.DocumentChanges{} // must be a slice
	for _, change := range edits {
		change := change
		documentChanges = append(documentChanges, protocol.DocumentChanges{
			TextDocumentEdit: &change,
		})
	}
	response, err := cli.ApplyEdit(ctx, &protocol.ApplyWorkspaceEditParams{
		Edit: protocol.WorkspaceEdit{
			DocumentChanges: documentChanges,
		},
	})
	if err != nil {
		return err
	}
	if !response.Applied {
		return fmt.Errorf("edits not applied because of %s", response.FailureReason)
	}
	return nil
}

func runGoGetModule(invoke func(...string) (*bytes.Buffer, error), addRequire bool, args []string) error {
	if addRequire {
		if err := addModuleRequire(invoke, args); err != nil {
			return err
		}
	}
	_, err := invoke(append([]string{"get", "-d"}, args...)...)
	return err
}

func addModuleRequire(invoke func(...string) (*bytes.Buffer, error), args []string) error {
	// Using go get to create a new dependency results in an
	// `// indirect` comment we may not want. The only way to avoid it
	// is to add the require as direct first. Then we can use go get to
	// update go.sum and tidy up.
	_, err := invoke(append([]string{"mod", "edit", "-require"}, args...)...)
	return err
}

func (s *Server) getUpgrades(ctx context.Context, snapshot source.Snapshot, uri span.URI, modules []string) (map[string]string, error) {
	stdout, err := snapshot.RunGoCommandDirect(ctx, source.Normal|source.AllowNetwork, &gocommand.Invocation{
		Verb:       "list",
		Args:       append([]string{"-m", "-u", "-json"}, modules...),
		WorkingDir: filepath.Dir(uri.Filename()),
		ModFlag:    "readonly",
	})
	if err != nil {
		return nil, err
	}

	upgrades := map[string]string{}
	for dec := json.NewDecoder(stdout); dec.More(); {
		mod := &gocommand.ModuleJSON{}
		if err := dec.Decode(mod); err != nil {
			return nil, err
		}
		if mod.Update == nil {
			continue
		}
		upgrades[mod.Path] = mod.Update.Version
	}
	return upgrades, nil
}

func (c *commandHandler) GCDetails(ctx context.Context, uri protocol.DocumentURI) error {
	return c.ToggleGCDetails(ctx, command.URIArg{URI: uri})
}

func (c *commandHandler) ToggleGCDetails(ctx context.Context, args command.URIArg) error {
	return c.run(ctx, commandConfig{
		requireSave: true,
		progress:    "Toggling GC Details",
		forURI:      args.URI,
	}, func(ctx context.Context, deps commandDeps) error {
		meta, err := source.NarrowestMetadataForFile(ctx, deps.snapshot, deps.fh.URI())
		if err != nil {
			return err
		}
		c.s.gcOptimizationDetailsMu.Lock()
		if _, ok := c.s.gcOptimizationDetails[meta.ID]; ok {
			delete(c.s.gcOptimizationDetails, meta.ID)
			c.s.clearDiagnosticSource(gcDetailsSource)
		} else {
			c.s.gcOptimizationDetails[meta.ID] = struct{}{}
		}
		c.s.gcOptimizationDetailsMu.Unlock()
		c.s.diagnoseSnapshot(deps.snapshot, nil, false, 0)
		return nil
	})
}

func (c *commandHandler) ListKnownPackages(ctx context.Context, args command.URIArg) (command.ListKnownPackagesResult, error) {
	var result command.ListKnownPackagesResult
	err := c.run(ctx, commandConfig{
		progress: "Listing packages",
		forURI:   args.URI,
	}, func(ctx context.Context, deps commandDeps) error {
		pkgs, err := source.KnownPackagePaths(ctx, deps.snapshot, deps.fh)
		for _, pkg := range pkgs {
			result.Packages = append(result.Packages, string(pkg))
		}
		return err
	})
	return result, err
}

func (c *commandHandler) ListImports(ctx context.Context, args command.URIArg) (command.ListImportsResult, error) {
	var result command.ListImportsResult
	err := c.run(ctx, commandConfig{
		forURI: args.URI,
	}, func(ctx context.Context, deps commandDeps) error {
		fh, err := deps.snapshot.ReadFile(ctx, args.URI.SpanURI())
		if err != nil {
			return err
		}
		pgf, err := deps.snapshot.ParseGo(ctx, fh, source.ParseHeader)
		if err != nil {
			return err
		}
		fset := tokeninternal.FileSetFor(pgf.Tok)
		for _, group := range astutil.Imports(fset, pgf.File) {
			for _, imp := range group {
				if imp.Path == nil {
					continue
				}
				var name string
				if imp.Name != nil {
					name = imp.Name.Name
				}
				result.Imports = append(result.Imports, command.FileImport{
					Path: string(source.UnquoteImportPath(imp)),
					Name: name,
				})
			}
		}
		meta, err := source.NarrowestMetadataForFile(ctx, deps.snapshot, args.URI.SpanURI())
		if err != nil {
			return err // e.g. cancelled
		}
		for pkgPath := range meta.DepsByPkgPath {
			result.PackageImports = append(result.PackageImports,
				command.PackageImport{Path: string(pkgPath)})
		}
		sort.Slice(result.PackageImports, func(i, j int) bool {
			return result.PackageImports[i].Path < result.PackageImports[j].Path
		})
		return nil
	})
	return result, err
}

func (c *commandHandler) AddImport(ctx context.Context, args command.AddImportArgs) error {
	return c.run(ctx, commandConfig{
		progress: "Adding import",
		forURI:   args.URI,
	}, func(ctx context.Context, deps commandDeps) error {
		edits, err := source.AddImport(ctx, deps.snapshot, deps.fh, args.ImportPath)
		if err != nil {
			return fmt.Errorf("could not add import: %v", err)
		}
		if _, err := c.s.client.ApplyEdit(ctx, &protocol.ApplyWorkspaceEditParams{
			Edit: protocol.WorkspaceEdit{
				DocumentChanges: documentChanges(deps.fh, edits),
			},
		}); err != nil {
			return fmt.Errorf("could not apply import edits: %v", err)
		}
		return nil
	})
}

func (c *commandHandler) StartDebugging(ctx context.Context, args command.DebuggingArgs) (result command.DebuggingResult, _ error) {
	addr := args.Addr
	if addr == "" {
		addr = "localhost:0"
	}
	di := debug.GetInstance(ctx)
	if di == nil {
		return result, errors.New("internal error: server has no debugging instance")
	}
	listenedAddr, err := di.Serve(ctx, addr)
	if err != nil {
		return result, fmt.Errorf("starting debug server: %w", err)
	}
	result.URLs = []string{"http://" + listenedAddr}
	openClientBrowser(ctx, c.s.client, result.URLs[0])
	return result, nil
}

func (c *commandHandler) StartProfile(ctx context.Context, args command.StartProfileArgs) (result command.StartProfileResult, _ error) {
	file, err := os.CreateTemp("", "gopls-profile-*")
	if err != nil {
		return result, fmt.Errorf("creating temp profile file: %v", err)
	}

	c.s.ongoingProfileMu.Lock()
	defer c.s.ongoingProfileMu.Unlock()

	if c.s.ongoingProfile != nil {
		file.Close() // ignore error
		return result, fmt.Errorf("profile already started (for %q)", c.s.ongoingProfile.Name())
	}

	if err := pprof.StartCPUProfile(file); err != nil {
		file.Close() // ignore error
		return result, fmt.Errorf("starting profile: %v", err)
	}

	c.s.ongoingProfile = file
	return result, nil
}

func (c *commandHandler) StopProfile(ctx context.Context, args command.StopProfileArgs) (result command.StopProfileResult, _ error) {
	c.s.ongoingProfileMu.Lock()
	defer c.s.ongoingProfileMu.Unlock()

	prof := c.s.ongoingProfile
	c.s.ongoingProfile = nil

	if prof == nil {
		return result, fmt.Errorf("no ongoing profile")
	}

	pprof.StopCPUProfile()
	if err := prof.Close(); err != nil {
		return result, fmt.Errorf("closing profile file: %v", err)
	}
	result.File = prof.Name()
	return result, nil
}

// Copy of pkgLoadConfig defined in internal/lsp/cmd/vulncheck.go
// TODO(hyangah): decide where to define this.
type pkgLoadConfig struct {
	// BuildFlags is a list of command-line flags to be passed through to
	// the build system's query tool.
	BuildFlags []string

	// If Tests is set, the loader includes related test packages.
	Tests bool
}

func (c *commandHandler) FetchVulncheckResult(ctx context.Context, arg command.URIArg) (map[protocol.DocumentURI]*vulncheck.Result, error) {
	ret := map[protocol.DocumentURI]*vulncheck.Result{}
	err := c.run(ctx, commandConfig{forURI: arg.URI}, func(ctx context.Context, deps commandDeps) error {
		if deps.snapshot.Options().Vulncheck == source.ModeVulncheckImports {
			for _, modfile := range deps.snapshot.ModFiles() {
				res, err := deps.snapshot.ModVuln(ctx, modfile)
				if err != nil {
					return err
				}
				ret[protocol.URIFromSpanURI(modfile)] = res
			}
		}
		// Overwrite if there is any govulncheck-based result.
		for modfile, result := range deps.snapshot.View().Vulnerabilities() {
			ret[protocol.URIFromSpanURI(modfile)] = result
		}
		return nil
	})
	return ret, err
}

func (c *commandHandler) RunGovulncheck(ctx context.Context, args command.VulncheckArgs) (command.RunVulncheckResult, error) {
	if args.URI == "" {
		return command.RunVulncheckResult{}, errors.New("VulncheckArgs is missing URI field")
	}

	// Return the workdone token so that clients can identify when this
	// vulncheck invocation is complete.
	//
	// Since the run function executes asynchronously, we use a channel to
	// synchronize the start of the run and return the token.
	tokenChan := make(chan protocol.ProgressToken, 1)
	err := c.run(ctx, commandConfig{
		async:       true, // need to be async to be cancellable
		progress:    "govulncheck",
		requireSave: true,
		forURI:      args.URI,
	}, func(ctx context.Context, deps commandDeps) error {
		tokenChan <- deps.work.Token()

		workDoneWriter := progress.NewWorkDoneWriter(ctx, deps.work)
		dir := filepath.Dir(args.URI.SpanURI().Filename())
		pattern := args.Pattern

		result, err := scan.RunGovulncheck(ctx, pattern, deps.snapshot, dir, workDoneWriter)
		if err != nil {
			return err
		}

		deps.snapshot.View().SetVulnerabilities(args.URI.SpanURI(), result)
		c.s.diagnoseSnapshot(deps.snapshot, nil, false, 0)

		affecting := make(map[string]bool, len(result.Entries))
		for _, finding := range result.Findings {
			if len(finding.Trace) > 1 { // at least 2 frames if callstack exists (vulnerability, entry)
				affecting[finding.OSV] = true
			}
		}
		if len(affecting) == 0 {
			return c.s.client.ShowMessage(ctx, &protocol.ShowMessageParams{
				Type:    protocol.Info,
				Message: "No vulnerabilities found",
			})
		}
		affectingOSVs := make([]string, 0, len(affecting))
		for id := range affecting {
			affectingOSVs = append(affectingOSVs, id)
		}
		sort.Strings(affectingOSVs)
		return c.s.client.ShowMessage(ctx, &protocol.ShowMessageParams{
			Type:    protocol.Warning,
			Message: fmt.Sprintf("Found %v", strings.Join(affectingOSVs, ", ")),
		})
	})
	if err != nil {
		return command.RunVulncheckResult{}, err
	}
	select {
	case <-ctx.Done():
		return command.RunVulncheckResult{}, ctx.Err()
	case token := <-tokenChan:
		return command.RunVulncheckResult{Token: token}, nil
	}
}

// MemStats implements the MemStats command. It returns an error as a
// future-proof API, but the resulting error is currently always nil.
func (c *commandHandler) MemStats(ctx context.Context) (command.MemStatsResult, error) {
	// GC a few times for stable results.
	runtime.GC()
	runtime.GC()
	runtime.GC()
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	return command.MemStatsResult{
		HeapAlloc:  m.HeapAlloc,
		HeapInUse:  m.HeapInuse,
		TotalAlloc: m.TotalAlloc,
	}, nil
}

// WorkspaceStats implements the WorkspaceStats command, reporting information
// about the current state of the loaded workspace for the current session.
func (c *commandHandler) WorkspaceStats(ctx context.Context) (command.WorkspaceStatsResult, error) {
	var res command.WorkspaceStatsResult
	res.Files.Total, res.Files.Largest, res.Files.Errs = c.s.session.Cache().FileStats()

	for _, view := range c.s.session.Views() {
		vs, err := collectViewStats(ctx, view)
		if err != nil {
			return res, err
		}
		res.Views = append(res.Views, vs)
	}
	return res, nil
}

func collectViewStats(ctx context.Context, view *cache.View) (command.ViewStats, error) {
	s, release, err := view.Snapshot()
	if err != nil {
		return command.ViewStats{}, err
	}
	defer release()

	allMD, err := s.AllMetadata(ctx)
	if err != nil {
		return command.ViewStats{}, err
	}
	allPackages := collectPackageStats(allMD)

	wsMD, err := s.WorkspaceMetadata(ctx)
	if err != nil {
		return command.ViewStats{}, err
	}
	workspacePackages := collectPackageStats(wsMD)

	var ids []source.PackageID
	for _, m := range wsMD {
		ids = append(ids, m.ID)
	}

	diags, err := s.PackageDiagnostics(ctx, ids...)
	if err != nil {
		return command.ViewStats{}, err
	}

	ndiags := 0
	for _, d := range diags {
		ndiags += len(d)
	}

	return command.ViewStats{
		GoCommandVersion:  view.GoVersionString(),
		AllPackages:       allPackages,
		WorkspacePackages: workspacePackages,
		Diagnostics:       ndiags,
	}, nil
}

func collectPackageStats(md []*source.Metadata) command.PackageStats {
	var stats command.PackageStats
	stats.Packages = len(md)
	modules := make(map[string]bool)

	for _, m := range md {
		n := len(m.CompiledGoFiles)
		stats.CompiledGoFiles += n
		if n > stats.LargestPackage {
			stats.LargestPackage = n
		}
		if m.Module != nil {
			modules[m.Module.Path] = true
		}
	}
	stats.Modules = len(modules)

	return stats
}

// RunGoWorkCommand invokes `go work <args>` with the provided arguments.
//
// args.InitFirst controls whether to first run `go work init`. This allows a
// single command to both create and recursively populate a go.work file -- as
// of writing there is no `go work init -r`.
//
// Some thought went into implementing this command. Unlike the go.mod commands
// above, this command simply invokes the go command and relies on the client
// to notify gopls of file changes via didChangeWatchedFile notifications.
// We could instead run these commands with GOWORK set to a temp file, but that
// poses the following problems:
//   - directory locations in the resulting temp go.work file will be computed
//     relative to the directory containing that go.work. If the go.work is in a
//     tempdir, the directories will need to be translated to/from that dir.
//   - it would be simpler to use a temp go.work file in the workspace
//     directory, or whichever directory contains the real go.work file, but
//     that sets a bad precedent of writing to a user-owned directory. We
//     shouldn't start doing that.
//   - Sending workspace edits to create a go.work file would require using
//     the CreateFile resource operation, which would need to be tested in every
//     client as we haven't used it before. We don't have time for that right
//     now.
//
// Therefore, we simply require that the current go.work file is saved (if it
// exists), and delegate to the go command.
func (c *commandHandler) RunGoWorkCommand(ctx context.Context, args command.RunGoWorkArgs) error {
	return c.run(ctx, commandConfig{
		progress: "Running go work command",
		forView:  args.ViewID,
	}, func(ctx context.Context, deps commandDeps) (runErr error) {
		snapshot := deps.snapshot
		view := snapshot.View().(*cache.View)
		viewDir := view.Folder().Filename()

		// If the user has explicitly set GOWORK=off, we should warn them
		// explicitly and avoid potentially misleading errors below.
		goworkURI, off := view.GOWORK()
		if off {
			return fmt.Errorf("cannot modify go.work files when GOWORK=off")
		}
		gowork := goworkURI.Filename()

		if goworkURI != "" {
			fh, err := snapshot.ReadFile(ctx, goworkURI)
			if err != nil {
				return fmt.Errorf("reading current go.work file: %v", err)
			}
			if !fh.SameContentsOnDisk() {
				return fmt.Errorf("must save workspace file %s before running go work commands", goworkURI)
			}
		} else {
			if !args.InitFirst {
				// If go.work does not exist, we should have detected that and asked
				// for InitFirst.
				return bug.Errorf("internal error: cannot run go work command: required go.work file not found")
			}
			gowork = filepath.Join(viewDir, "go.work")
			if err := c.invokeGoWork(ctx, viewDir, gowork, []string{"init"}); err != nil {
				return fmt.Errorf("running `go work init`: %v", err)
			}
		}

		return c.invokeGoWork(ctx, viewDir, gowork, args.Args)
	})
}

func (c *commandHandler) invokeGoWork(ctx context.Context, viewDir, gowork string, args []string) error {
	inv := gocommand.Invocation{
		Verb:       "work",
		Args:       args,
		WorkingDir: viewDir,
		Env:        append(os.Environ(), fmt.Sprintf("GOWORK=%s", gowork)),
	}
	if _, err := c.s.session.GoCommandRunner().Run(ctx, inv); err != nil {
		return fmt.Errorf("running go work command: %v", err)
	}
	return nil
}

// openClientBrowser causes the LSP client to open the specified URL
// in an external browser.
func openClientBrowser(ctx context.Context, cli protocol.Client, url protocol.URI) {
	showDocumentImpl(ctx, cli, url, nil)
}

// openClientEditor causes the LSP client to open the specified document
// and select the indicated range.
func openClientEditor(ctx context.Context, cli protocol.Client, loc protocol.Location) {
	showDocumentImpl(ctx, cli, protocol.URI(loc.URI), &loc.Range)
}

func showDocumentImpl(ctx context.Context, cli protocol.Client, url protocol.URI, rangeOpt *protocol.Range) {
	// In principle we shouldn't send a showDocument request to a
	// client that doesn't support it, as reported by
	// ShowDocumentClientCapabilities. But even clients that do
	// support it may defer the real work of opening the document
	// asynchronously, to avoid deadlocks due to rentrancy.
	//
	// For example: client sends request to server; server sends
	// showDocument to client; client opens editor; editor causes
	// new RPC to be sent to server, which is still busy with
	// previous request. (This happens in eglot.)
	//
	// So we can't rely on the success/failure information.
	// That's the reason this function doesn't return an error.

	// "External" means run the system-wide handler (e.g. open(1)
	// on macOS or xdg-open(1) on Linux) for this URL, ignoring
	// TakeFocus and Selection. Note that this may still end up
	// opening the same editor (e.g. VSCode) for a file: URL.
	res, err := cli.ShowDocument(ctx, &protocol.ShowDocumentParams{
		URI:       url,
		External:  rangeOpt == nil,
		TakeFocus: true,
		Selection: rangeOpt, // optional
	})
	if err != nil {
		event.Error(ctx, "client.showDocument: %v", err)
	} else if res != nil && !res.Success {
		event.Log(ctx, fmt.Sprintf("client declined to open document %v", url))
	}
}
