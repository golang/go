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
	"os"
	"path/filepath"
	"strings"

	"golang.org/x/mod/modfile"
	"golang.org/x/tools/internal/event"
	"golang.org/x/tools/internal/gocommand"
	"golang.org/x/tools/internal/lsp/cache"
	"golang.org/x/tools/internal/lsp/command"
	"golang.org/x/tools/internal/lsp/debug"
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

// commandConfig configures common command set-up and execution.
type commandConfig struct {
	async       bool                 // whether to run the command asynchronously. Async commands can only return errors.
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

func (c *commandHandler) run(ctx context.Context, cfg commandConfig, run commandFunc) (err error) {
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
		deps.snapshot, deps.fh, ok, release, err = c.s.beginFileRequest(ctx, cfg.forURI, source.UnknownKind)
		defer release()
		if !ok {
			return err
		}
	}
	ctx, cancel := context.WithCancel(xcontext.Detach(ctx))
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
		deps.snapshot.View().RegisterModuleUpgrades(upgrades)
		// Re-diagnose the snapshot to publish the new module diagnostics.
		c.s.diagnoseSnapshot(deps.snapshot, nil, false)
		return nil
	})
}

func (c *commandHandler) AddDependency(ctx context.Context, args command.DependencyArgs) error {
	return c.GoGetModule(ctx, args)
}

func (c *commandHandler) UpgradeDependency(ctx context.Context, args command.DependencyArgs) error {
	return c.GoGetModule(ctx, args)
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
		_, err := deps.snapshot.RunGoCommandDirect(ctx, source.Normal|source.AllowNetwork, &gocommand.Invocation{
			Verb:       "mod",
			Args:       []string{"vendor"},
			WorkingDir: filepath.Dir(args.URI.SpanURI().Filename()),
		})
		return err
	})
}

func (c *commandHandler) RemoveDependency(ctx context.Context, args command.RemoveDependencyArgs) error {
	return c.run(ctx, commandConfig{
		progress: "Removing dependency",
		forURI:   args.URI,
	}, func(ctx context.Context, deps commandDeps) error {
		// If the module is tidied apart from the one unused diagnostic, we can
		// run `go get module@none`, and then run `go mod tidy`. Otherwise, we
		// must make textual edits.
		// TODO(rstambler): In Go 1.17+, we will be able to use the go command
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
			return errors.Errorf("running tests failed: %w", err)
		}
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
		er := &eventWriter{ctx: ctx, operation: "generate"}

		pattern := "."
		if args.Recursive {
			pattern = "./..."
		}
		inv := &gocommand.Invocation{
			Verb:       "generate",
			Args:       []string{"-x", pattern},
			WorkingDir: args.Dir.SpanURI().Filename(),
		}
		stderr := io.MultiWriter(er, workDoneWriter{deps.work})
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
	modEdits, err := applyFileEdits(ctx, snapshot, modURI, newModBytes)
	if err != nil {
		return err
	}
	sumEdits, err := applyFileEdits(ctx, snapshot, sumURI, newSumBytes)
	if err != nil {
		return err
	}
	changes := append(sumEdits, modEdits...)
	if len(changes) == 0 {
		return nil
	}
	response, err := s.client.ApplyEdit(ctx, &protocol.ApplyWorkspaceEditParams{
		Edit: protocol.WorkspaceEdit{
			DocumentChanges: changes,
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

func applyFileEdits(ctx context.Context, snapshot source.Snapshot, uri span.URI, newContent []byte) ([]protocol.TextDocumentEdit, error) {
	fh, err := snapshot.GetVersionedFile(ctx, uri)
	if err != nil {
		return nil, err
	}
	oldContent, err := fh.Read()
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
		err := ioutil.WriteFile(uri.Filename(), newContent, 0666)
		return nil, err
	}

	m := &protocol.ColumnMapper{
		URI:       fh.URI(),
		Converter: span.NewContentConverter(fh.URI().Filename(), oldContent),
		Content:   oldContent,
	}
	diff, err := snapshot.View().Options().ComputeEdits(uri, string(oldContent), string(newContent))
	if err != nil {
		return nil, err
	}
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
		pkg, err := deps.snapshot.PackageForFile(ctx, deps.fh.URI(), source.TypecheckWorkspace, source.NarrowestPackage)
		if err != nil {
			return err
		}
		c.s.gcOptimizationDetailsMu.Lock()
		if _, ok := c.s.gcOptimizationDetails[pkg.ID()]; ok {
			delete(c.s.gcOptimizationDetails, pkg.ID())
			c.s.clearDiagnosticSource(gcDetailsSource)
		} else {
			c.s.gcOptimizationDetails[pkg.ID()] = struct{}{}
		}
		c.s.gcOptimizationDetailsMu.Unlock()
		c.s.diagnoseSnapshot(deps.snapshot, nil, false)
		return nil
	})
}

func (c *commandHandler) GenerateGoplsMod(ctx context.Context, args command.URIArg) error {
	// TODO: go back to using URI
	return c.run(ctx, commandConfig{
		requireSave: true,
		progress:    "Generating gopls.mod",
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

func (c *commandHandler) ListKnownPackages(ctx context.Context, args command.URIArg) (command.ListKnownPackagesResult, error) {
	var result command.ListKnownPackagesResult
	err := c.run(ctx, commandConfig{
		progress: "Listing packages",
		forURI:   args.URI,
	}, func(ctx context.Context, deps commandDeps) error {
		var err error
		result.Packages, err = source.KnownPackages(ctx, deps.snapshot, deps.fh)
		return err
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

func (c *commandHandler) WorkspaceMetadata(ctx context.Context) (command.WorkspaceMetadataResult, error) {
	var result command.WorkspaceMetadataResult
	for _, view := range c.s.session.Views() {
		result.Workspaces = append(result.Workspaces, command.Workspace{
			Name:      view.Name(),
			ModuleDir: view.TempWorkspace().Filename(),
		})
	}
	return result, nil
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
		return result, errors.Errorf("starting debug server: %w", err)
	}
	result.URLs = []string{"http://" + listenedAddr}
	return result, nil
}
