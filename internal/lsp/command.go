// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lsp

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"io/ioutil"
	"path/filepath"
	"strings"

	"golang.org/x/mod/modfile"
	"golang.org/x/tools/internal/event"
	"golang.org/x/tools/internal/gocommand"
	"golang.org/x/tools/internal/lsp/cache"
	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/span"
	"golang.org/x/tools/internal/xcontext"
	errors "golang.org/x/xerrors"
)

func (s *Server) executeCommand(ctx context.Context, params *protocol.ExecuteCommandParams) (interface{}, error) {
	var found bool
	for _, name := range s.session.Options().SupportedCommands {
		if name == params.Command {
			found = true
			break
		}
	}
	if !found {
		return nil, fmt.Errorf("%s is not a supported command", params.Command)
	}

	cmd := &commandHandler{
		ctx:    ctx,
		s:      s,
		params: params,
	}
	return cmd.dispatch()
}

type commandHandler struct {
	// ctx is temporarily held so that we may implement the command.Interface interface.
	ctx    context.Context
	s      *Server
	params *protocol.ExecuteCommandParams
}

// commandConfig configures common command set-up and execution.
type commandConfig struct {
	async       bool
	requireSave bool                 // whether all files must be saved for the command to work
	progress    string               // title to use for progress reporting. If empty, no progress will be reported.
	forURI      protocol.DocumentURI // URI to resolve to a snapshot. If unset, snapshot will be nil.
}

// commandDeps is evaluated from a commandConfig. Note that not all fields may
// be populated, depending on which configuration is set. See comments in-line
// for details.
type commandDeps struct {
	snapshot source.Snapshot            // present if cfg.forURI was set
	fh       source.VersionedFileHandle // present if cfg.forURI was set
	work     *workDone                  // present cfg.progress was set
}

type commandFunc func(context.Context, commandDeps) error

func (c *commandHandler) run(cfg commandConfig, run commandFunc) (err error) {
	if cfg.requireSave {
		for _, overlay := range c.s.session.Overlays() {
			if !overlay.Saved() {
				return errors.New("All files must be saved first")
			}
		}
	}
	var deps commandDeps
	if cfg.forURI != "" {
		var ok bool
		var release func()
		deps.snapshot, deps.fh, ok, release, err = c.s.beginFileRequest(c.ctx, cfg.forURI, source.UnknownKind)
		defer release()
		if !ok {
			return err
		}
	}
	ctx, cancel := context.WithCancel(xcontext.Detach(c.ctx))
	if cfg.progress != "" {
		deps.work = c.s.progress.start(ctx, cfg.progress, "Running...", c.params.WorkDoneToken, cancel)
	}
	runcmd := func() error {
		defer cancel()
		err := run(ctx, deps)
		switch {
		case errors.Is(err, context.Canceled):
			deps.work.end("canceled")
		case err != nil:
			event.Error(ctx, "command error", err)
			deps.work.end("failed")
		default:
			deps.work.end("completed")
		}
		return err
	}
	if cfg.async {
		go runcmd()
		return nil
	}
	return runcmd()
}

func (c *commandHandler) dispatch() (interface{}, error) {
	switch c.params.Command {
	case source.CommandFillStruct.ID(), source.CommandUndeclaredName.ID(),
		source.CommandExtractVariable.ID(), source.CommandExtractFunction.ID():
		var uri protocol.DocumentURI
		var rng protocol.Range
		if err := source.UnmarshalArgs(c.params.Arguments, &uri, &rng); err != nil {
			return nil, err
		}
		err := c.ApplyFix(uri, rng)
		return nil, err
	case source.CommandTest.ID():
		var uri protocol.DocumentURI
		var tests, benchmarks []string
		if err := source.UnmarshalArgs(c.params.Arguments, &uri, &tests, &benchmarks); err != nil {
			return nil, err
		}
		err := c.RunTests(uri, tests, benchmarks)
		return nil, err
	case source.CommandGenerate.ID():
		var uri protocol.DocumentURI
		var recursive bool
		if err := source.UnmarshalArgs(c.params.Arguments, &uri, &recursive); err != nil {
			return nil, err
		}
		err := c.Generate(uri, recursive)
		return nil, err
	case source.CommandRegenerateCgo.ID():
		var uri protocol.DocumentURI
		if err := source.UnmarshalArgs(c.params.Arguments, &uri); err != nil {
			return nil, err
		}
		return nil, c.RegenerateCgo(uri)
	case source.CommandTidy.ID():
		var uri protocol.DocumentURI
		if err := source.UnmarshalArgs(c.params.Arguments, &uri); err != nil {
			return nil, err
		}
		return nil, c.Tidy(uri)
	case source.CommandVendor.ID():
		var uri protocol.DocumentURI
		if err := source.UnmarshalArgs(c.params.Arguments, &uri); err != nil {
			return nil, err
		}
		return nil, c.Vendor(uri)
	case source.CommandUpdateGoSum.ID():
		var uri protocol.DocumentURI
		if err := source.UnmarshalArgs(c.params.Arguments, &uri); err != nil {
			return nil, err
		}
		return nil, c.UpdateGoSum(uri)
	case source.CommandCheckUpgrades.ID():
		var uri protocol.DocumentURI
		var modules []string
		if err := source.UnmarshalArgs(c.params.Arguments, &uri, &modules); err != nil {
			return nil, err
		}
		return nil, c.CheckUpgrades(uri, modules)
	case source.CommandAddDependency.ID(), source.CommandUpgradeDependency.ID():
		var uri protocol.DocumentURI
		var goCmdArgs []string
		var addRequire bool
		if err := source.UnmarshalArgs(c.params.Arguments, &uri, &addRequire, &goCmdArgs); err != nil {
			return nil, err
		}
		return nil, c.GoGetModule(uri, addRequire, goCmdArgs)
	case source.CommandRemoveDependency.ID():
		var uri protocol.DocumentURI
		var modulePath string
		var onlyDiagnostic bool
		if err := source.UnmarshalArgs(c.params.Arguments, &uri, &onlyDiagnostic, &modulePath); err != nil {
			return nil, err
		}
		return nil, c.RemoveDependency(modulePath, uri, onlyDiagnostic)
	case source.CommandGoGetPackage.ID():
		var uri protocol.DocumentURI
		var pkg string
		var addRequire bool
		if err := source.UnmarshalArgs(c.params.Arguments, &uri, &addRequire, &pkg); err != nil {
			return nil, err
		}
		return nil, c.GoGetPackage(uri, addRequire, pkg)
	case source.CommandToggleDetails.ID():
		var uri protocol.DocumentURI
		if err := source.UnmarshalArgs(c.params.Arguments, &uri); err != nil {
			return nil, err
		}
		return nil, c.GCDetails(uri)
	case source.CommandGenerateGoplsMod.ID():
		return nil, c.GenerateGoplsMod()
	}
	return nil, fmt.Errorf("unsupported command: %s", c.params.Command)
}

func (c *commandHandler) ApplyFix(uri protocol.DocumentURI, rng protocol.Range) error {
	return c.run(commandConfig{
		// Note: no progress here. Applying fixes should be quick.
		forURI: uri,
	}, func(ctx context.Context, deps commandDeps) error {
		edits, err := source.ApplyFix(ctx, c.params.Command, deps.snapshot, deps.fh, rng)
		if err != nil {
			return err
		}
		r, err := c.s.client.ApplyEdit(ctx, &protocol.ApplyWorkspaceEditParams{
			Edit: protocol.WorkspaceEdit{
				DocumentChanges: edits,
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

func (c *commandHandler) RegenerateCgo(uri protocol.DocumentURI) error {
	return c.run(commandConfig{
		progress: source.CommandRegenerateCgo.Title,
	}, func(ctx context.Context, deps commandDeps) error {
		mod := source.FileModification{
			URI:    uri.SpanURI(),
			Action: source.InvalidateMetadata,
		}
		return c.s.didModifyFiles(c.ctx, []source.FileModification{mod}, FromRegenerateCgo)
	})
}

func (c *commandHandler) CheckUpgrades(uri protocol.DocumentURI, modules []string) error {
	return c.run(commandConfig{
		forURI:   uri,
		progress: source.CommandCheckUpgrades.Title,
	}, func(ctx context.Context, deps commandDeps) error {
		upgrades, err := c.s.getUpgrades(ctx, deps.snapshot, uri.SpanURI(), modules)
		if err != nil {
			return err
		}
		deps.snapshot.View().RegisterModuleUpgrades(upgrades)
		// Re-diagnose the snapshot to publish the new module diagnostics.
		c.s.diagnoseSnapshot(deps.snapshot, nil, false)
		return nil
	})
}

func (c *commandHandler) GoGetModule(uri protocol.DocumentURI, addRequire bool, goCmdArgs []string) error {
	return c.run(commandConfig{
		requireSave: true,
		progress:    "Running go get",
		forURI:      uri,
	}, func(ctx context.Context, deps commandDeps) error {
		return runGoGetModule(ctx, deps.snapshot, uri.SpanURI(), addRequire, goCmdArgs)
	})
}

// TODO(rFindley): UpdateGoSum, Tidy, and Vendor could probably all be one command.

func (c *commandHandler) UpdateGoSum(uri protocol.DocumentURI) error {
	return c.run(commandConfig{
		requireSave: true,
		progress:    source.CommandUpdateGoSum.Title,
		forURI:      uri,
	}, func(ctx context.Context, deps commandDeps) error {
		return runSimpleGoCommand(ctx, deps.snapshot, source.UpdateUserModFile|source.AllowNetwork, uri.SpanURI(), "list", []string{"all"})
	})
}

func (c *commandHandler) Tidy(uri protocol.DocumentURI) error {
	return c.run(commandConfig{
		requireSave: true,
		progress:    source.CommandTidy.Title,
		forURI:      uri,
	}, func(ctx context.Context, deps commandDeps) error {
		return runSimpleGoCommand(ctx, deps.snapshot, source.UpdateUserModFile|source.AllowNetwork, uri.SpanURI(), "mod", []string{"tidy"})
	})
}

func (c *commandHandler) Vendor(uri protocol.DocumentURI) error {
	return c.run(commandConfig{
		requireSave: true,
		progress:    source.CommandVendor.Title,
		forURI:      uri,
	}, func(ctx context.Context, deps commandDeps) error {
		return runSimpleGoCommand(ctx, deps.snapshot, source.UpdateUserModFile|source.AllowNetwork, uri.SpanURI(), "mod", []string{"vendor"})
	})
}

func (c *commandHandler) RemoveDependency(modulePath string, uri protocol.DocumentURI, onlyDiagnostic bool) error {
	return c.run(commandConfig{
		requireSave: true,
		progress:    source.CommandRemoveDependency.Title,
		forURI:      uri,
	}, func(ctx context.Context, deps commandDeps) error {
		// If the module is tidied apart from the one unused diagnostic, we can
		// run `go get module@none`, and then run `go mod tidy`. Otherwise, we
		// must make textual edits.
		// TODO(rstambler): In Go 1.17+, we will be able to use the go command
		// without checking if the module is tidy.
		if onlyDiagnostic {
			if err := runGoGetModule(ctx, deps.snapshot, uri.SpanURI(), false, []string{modulePath + "@none"}); err != nil {
				return err
			}
			return runSimpleGoCommand(ctx, deps.snapshot, source.UpdateUserModFile|source.AllowNetwork, uri.SpanURI(), "mod", []string{"tidy"})
		}
		pm, err := deps.snapshot.ParseMod(ctx, deps.fh)
		if err != nil {
			return err
		}
		edits, err := dropDependency(deps.snapshot, pm, modulePath)
		if err != nil {
			return err
		}
		response, err := c.s.client.ApplyEdit(ctx, &protocol.ApplyWorkspaceEditParams{
			Edit: protocol.WorkspaceEdit{
				DocumentChanges: []protocol.TextDocumentEdit{{
					TextDocument: protocol.OptionalVersionedTextDocumentIdentifier{
						Version: deps.fh.Version(),
						TextDocumentIdentifier: protocol.TextDocumentIdentifier{
							URI: protocol.URIFromSpanURI(deps.fh.URI()),
						},
					},
					Edits: edits,
				}},
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
	diff, err := snapshot.View().Options().ComputeEdits(pm.URI, string(pm.Mapper.Content), string(newContent))
	if err != nil {
		return nil, err
	}
	return source.ToProtocolEdits(pm.Mapper, diff)
}

func (c *commandHandler) RunTests(uri protocol.DocumentURI, tests, benchmarks []string) error {
	return c.run(commandConfig{
		async:       true,
		progress:    source.CommandTest.Title,
		requireSave: true,
		forURI:      uri,
	}, func(ctx context.Context, deps commandDeps) error {
		if err := c.runTests(ctx, deps.snapshot, deps.work, uri, tests, benchmarks); err != nil {
			if err := c.s.client.ShowMessage(ctx, &protocol.ShowMessageParams{
				Type:    protocol.Error,
				Message: fmt.Sprintf("Running tests failed: %v", err),
			}); err != nil {
				event.Error(ctx, "running tests: failed to show message", err)
			}
		}
		// Since we're running asynchronously, any error returned here would be
		// ignored.
		return nil
	})
}

func (c *commandHandler) runTests(ctx context.Context, snapshot source.Snapshot, work *workDone, uri protocol.DocumentURI, tests, benchmarks []string) error {
	// TODO: fix the error reporting when this runs async.
	pkgs, err := snapshot.PackagesForFile(ctx, uri.SpanURI(), source.TypecheckWorkspace)
	if err != nil {
		return err
	}
	if len(pkgs) == 0 {
		return fmt.Errorf("package could not be found for file: %s", uri.SpanURI().Filename())
	}
	pkgPath := pkgs[0].ForTest()

	// create output
	buf := &bytes.Buffer{}
	ew := &eventWriter{ctx: ctx, operation: "test"}
	out := io.MultiWriter(ew, workDoneWriter{work}, buf)

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

func (c *commandHandler) Generate(uri protocol.DocumentURI, recursive bool) error {
	return c.run(commandConfig{
		requireSave: true,
		progress:    source.CommandGenerate.Title,
		forURI:      uri,
	}, func(ctx context.Context, deps commandDeps) error {
		er := &eventWriter{ctx: ctx, operation: "generate"}

		pattern := "."
		if recursive {
			pattern = "./..."
		}
		inv := &gocommand.Invocation{
			Verb:       "generate",
			Args:       []string{"-x", pattern},
			WorkingDir: uri.SpanURI().Filename(),
		}
		stderr := io.MultiWriter(er, workDoneWriter{deps.work})
		if err := deps.snapshot.RunGoCommandPiped(ctx, source.Normal, inv, er, stderr); err != nil {
			return err
		}
		return nil
	})
}

func (c *commandHandler) GoGetPackage(puri protocol.DocumentURI, addRequire bool, pkg string) error {
	return c.run(commandConfig{
		forURI:   puri,
		progress: source.CommandGoGetPackage.Title,
	}, func(ctx context.Context, deps commandDeps) error {
		uri := puri.SpanURI()
		stdout, err := deps.snapshot.RunGoCommandDirect(ctx, source.WriteTemporaryModFile|source.AllowNetwork, &gocommand.Invocation{
			Verb:       "list",
			Args:       []string{"-f", "{{.Module.Path}}@{{.Module.Version}}", pkg},
			WorkingDir: filepath.Dir(uri.Filename()),
		})
		if err != nil {
			return err
		}
		ver := strings.TrimSpace(stdout.String())
		return runGoGetModule(ctx, deps.snapshot, uri, addRequire, []string{ver})
	})
}

func runGoGetModule(ctx context.Context, snapshot source.Snapshot, uri span.URI, addRequire bool, args []string) error {
	if addRequire {
		// Using go get to create a new dependency results in an
		// `// indirect` comment we may not want. The only way to avoid it
		// is to add the require as direct first. Then we can use go get to
		// update go.sum and tidy up.
		if err := runSimpleGoCommand(ctx, snapshot, source.UpdateUserModFile, uri, "mod", append([]string{"edit", "-require"}, args...)); err != nil {
			return err
		}
	}
	return runSimpleGoCommand(ctx, snapshot, source.UpdateUserModFile|source.AllowNetwork, uri, "get", append([]string{"-d"}, args...))
}

func runSimpleGoCommand(ctx context.Context, snapshot source.Snapshot, mode source.InvocationFlags, uri span.URI, verb string, args []string) error {
	_, err := snapshot.RunGoCommandDirect(ctx, mode, &gocommand.Invocation{
		Verb:       verb,
		Args:       args,
		WorkingDir: filepath.Dir(uri.Filename()),
	})
	return err
}

func (s *Server) getUpgrades(ctx context.Context, snapshot source.Snapshot, uri span.URI, modules []string) (map[string]string, error) {
	stdout, err := snapshot.RunGoCommandDirect(ctx, source.Normal|source.AllowNetwork, &gocommand.Invocation{
		Verb:       "list",
		Args:       append([]string{"-m", "-u", "-json"}, modules...),
		WorkingDir: filepath.Dir(uri.Filename()),
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

func (c *commandHandler) GCDetails(uri protocol.DocumentURI) error {
	return c.run(commandConfig{
		requireSave: true,
		progress:    source.CommandToggleDetails.Title,
		forURI:      uri,
	}, func(ctx context.Context, deps commandDeps) error {
		pkgDir := span.URIFromPath(filepath.Dir(uri.SpanURI().Filename()))
		c.s.gcOptimizationDetailsMu.Lock()
		if _, ok := c.s.gcOptimizationDetails[pkgDir]; ok {
			delete(c.s.gcOptimizationDetails, pkgDir)
			c.s.clearDiagnosticSource(gcDetailsSource)
		} else {
			c.s.gcOptimizationDetails[pkgDir] = struct{}{}
		}
		c.s.gcOptimizationDetailsMu.Unlock()
		c.s.diagnoseSnapshot(deps.snapshot, nil, false)
		return nil
	})
}

func (c *commandHandler) GenerateGoplsMod() error {
	return c.run(commandConfig{
		requireSave: true,
		progress:    source.CommandGenerateGoplsMod.Title,
	}, func(ctx context.Context, deps commandDeps) error {
		views := c.s.session.Views()
		if len(views) != 1 {
			return fmt.Errorf("cannot resolve view: have %d views", len(views))
		}
		v := views[0]
		snapshot, release := v.Snapshot(ctx)
		defer release()
		modFile, err := cache.BuildGoplsMod(ctx, snapshot.View().Folder(), snapshot)
		if err != nil {
			return errors.Errorf("getting workspace mod file: %w", err)
		}
		content, err := modFile.Format()
		if err != nil {
			return errors.Errorf("formatting mod file: %w", err)
		}
		filename := filepath.Join(snapshot.View().Folder().Filename(), "gopls.mod")
		if err := ioutil.WriteFile(filename, content, 0644); err != nil {
			return errors.Errorf("writing mod file: %w", err)
		}
		return nil
	})
}
