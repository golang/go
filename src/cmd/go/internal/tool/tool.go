// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package tool implements the “go tool” command.
package tool

import (
	"cmd/internal/telemetry/counter"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"go/build"
	"internal/platform"
	"maps"
	"os"
	"os/exec"
	"os/signal"
	"path"
	"slices"
	"sort"
	"strings"

	"cmd/go/internal/base"
	"cmd/go/internal/cfg"
	"cmd/go/internal/load"
	"cmd/go/internal/modindex"
	"cmd/go/internal/modload"
	"cmd/go/internal/str"
	"cmd/go/internal/work"
)

var CmdTool = &base.Command{
	Run:       runTool,
	UsageLine: "go tool [-n] command [args...]",
	Short:     "run specified go tool",
	Long: `
Tool runs the go tool command identified by the arguments.

Go ships with a number of builtin tools, and additional tools
may be defined in the go.mod of the current module.

With no arguments it prints the list of known tools.

The -n flag causes tool to print the command that would be
executed but not execute it.

The -modfile=file.mod build flag causes tool to use an alternate file
instead of the go.mod in the module root directory.

Tool also provides the -C, -overlay, and -modcacherw build flags.

For more about build flags, see 'go help build'.

For more about each builtin tool command, see 'go doc cmd/<command>'.
`,
}

var toolN bool

// Return whether tool can be expected in the gccgo tool directory.
// Other binaries could be in the same directory so don't
// show those with the 'go tool' command.
func isGccgoTool(tool string) bool {
	switch tool {
	case "cgo", "fix", "cover", "godoc", "vet":
		return true
	}
	return false
}

func init() {
	base.AddChdirFlag(&CmdTool.Flag)
	base.AddModCommonFlags(&CmdTool.Flag)
	CmdTool.Flag.BoolVar(&toolN, "n", false, "")
}

func runTool(ctx context.Context, cmd *base.Command, args []string) {
	if len(args) == 0 {
		counter.Inc("go/subcommand:tool")
		listTools(modload.LoaderState, ctx)
		return
	}
	toolName := args[0]

	toolPath, err := base.ToolPath(toolName)
	if err != nil {
		if toolName == "dist" && len(args) > 1 && args[1] == "list" {
			// cmd/distpack removes the 'dist' tool from the toolchain to save space,
			// since it is normally only used for building the toolchain in the first
			// place. However, 'go tool dist list' is useful for listing all supported
			// platforms.
			//
			// If the dist tool does not exist, impersonate this command.
			if impersonateDistList(args[2:]) {
				// If it becomes necessary, we could increment an additional counter to indicate
				// that we're impersonating dist list if knowing that becomes important?
				counter.Inc("go/subcommand:tool-dist")
				return
			}
		}

		// See if tool can be a builtin tool. If so, try to build and run it.
		// buildAndRunBuiltinTool will fail if the install target of the loaded package is not
		// the tool directory.
		if tool := loadBuiltinTool(toolName); tool != "" {
			// Increment a counter for the tool subcommand with the tool name.
			counter.Inc("go/subcommand:tool-" + toolName)
			buildAndRunBuiltinTool(modload.LoaderState, ctx, toolName, tool, args[1:])
			return
		}

		// Try to build and run mod tool.
		tool := loadModTool(modload.LoaderState, ctx, toolName)
		if tool != "" {
			buildAndRunModtool(modload.LoaderState, ctx, toolName, tool, args[1:])
			return
		}

		counter.Inc("go/subcommand:tool-unknown")

		// Emit the usual error for the missing tool.
		_ = base.Tool(toolName)
	} else {
		// Increment a counter for the tool subcommand with the tool name.
		counter.Inc("go/subcommand:tool-" + toolName)
	}

	runBuiltTool(toolName, nil, append([]string{toolPath}, args[1:]...))
}

// listTools prints a list of the available tools in the tools directory.
func listTools(loaderstate *modload.State, ctx context.Context) {
	f, err := os.Open(build.ToolDir)
	if err != nil {
		fmt.Fprintf(os.Stderr, "go: no tool directory: %s\n", err)
		base.SetExitStatus(2)
		return
	}
	defer f.Close()
	names, err := f.Readdirnames(-1)
	if err != nil {
		fmt.Fprintf(os.Stderr, "go: can't read tool directory: %s\n", err)
		base.SetExitStatus(2)
		return
	}

	sort.Strings(names)
	for _, name := range names {
		// Unify presentation by going to lower case.
		// If it's windows, don't show the .exe suffix.
		name = strings.TrimSuffix(strings.ToLower(name), cfg.ToolExeSuffix())

		// The tool directory used by gccgo will have other binaries
		// in addition to go tools. Only display go tools here.
		if cfg.BuildToolchainName == "gccgo" && !isGccgoTool(name) {
			continue
		}
		fmt.Println(name)
	}

	modload.InitWorkfile(loaderstate)
	modload.LoadModFile(loaderstate, ctx)
	modTools := slices.Sorted(maps.Keys(loaderstate.MainModules.Tools()))
	for _, tool := range modTools {
		fmt.Println(tool)
	}
}

func impersonateDistList(args []string) (handled bool) {
	fs := flag.NewFlagSet("go tool dist list", flag.ContinueOnError)
	jsonFlag := fs.Bool("json", false, "produce JSON output")
	brokenFlag := fs.Bool("broken", false, "include broken ports")

	// The usage for 'go tool dist' claims that
	// “All commands take -v flags to emit extra information”,
	// but list -v appears not to have any effect.
	_ = fs.Bool("v", false, "emit extra information")

	if err := fs.Parse(args); err != nil || len(fs.Args()) > 0 {
		// Unrecognized flag or argument.
		// Force fallback to the real 'go tool dist'.
		return false
	}

	if !*jsonFlag {
		for _, p := range platform.List {
			if !*brokenFlag && platform.Broken(p.GOOS, p.GOARCH) {
				continue
			}
			fmt.Println(p)
		}
		return true
	}

	type jsonResult struct {
		GOOS         string
		GOARCH       string
		CgoSupported bool
		FirstClass   bool
		Broken       bool `json:",omitempty"`
	}

	var results []jsonResult
	for _, p := range platform.List {
		broken := platform.Broken(p.GOOS, p.GOARCH)
		if broken && !*brokenFlag {
			continue
		}
		if *jsonFlag {
			results = append(results, jsonResult{
				GOOS:         p.GOOS,
				GOARCH:       p.GOARCH,
				CgoSupported: platform.CgoSupported(p.GOOS, p.GOARCH),
				FirstClass:   platform.FirstClass(p.GOOS, p.GOARCH),
				Broken:       broken,
			})
		}
	}
	out, err := json.MarshalIndent(results, "", "\t")
	if err != nil {
		return false
	}

	os.Stdout.Write(out)
	return true
}

func defaultExecName(importPath string) string {
	var p load.Package
	p.ImportPath = importPath
	return p.DefaultExecName()
}

func loadBuiltinTool(toolName string) string {
	if !base.ValidToolName(toolName) {
		return ""
	}
	cmdTool := path.Join("cmd", toolName)
	if !modindex.IsStandardPackage(cfg.GOROOT, cfg.BuildContext.Compiler, cmdTool) {
		return ""
	}
	// Create a fake package and check to see if it would be installed to the tool directory.
	// If not, it's not a builtin tool.
	p := &load.Package{PackagePublic: load.PackagePublic{Name: "main", ImportPath: cmdTool, Goroot: true}}
	if load.InstallTargetDir(p) != load.ToTool {
		return ""
	}
	return cmdTool
}

func loadModTool(loaderstate *modload.State, ctx context.Context, name string) string {
	modload.InitWorkfile(loaderstate)
	modload.LoadModFile(loaderstate, ctx)

	matches := []string{}
	for tool := range loaderstate.MainModules.Tools() {
		if tool == name || defaultExecName(tool) == name {
			matches = append(matches, tool)
		}
	}

	if len(matches) == 1 {
		return matches[0]
	}

	if len(matches) > 1 {
		message := fmt.Sprintf("tool %q is ambiguous; choose one of:\n\t", name)
		for _, tool := range matches {
			message += tool + "\n\t"
		}
		base.Fatal(errors.New(message))
	}

	return ""
}

func builtTool(runAction *work.Action) string {
	linkAction := runAction.Deps[0]
	if toolN {
		// #72824: If -n is set, use the cached path if we can.
		// This is only necessary if the binary wasn't cached
		// before this invocation of the go command: if the binary
		// was cached, BuiltTarget() will be the cached executable.
		// It's only in the "first run", where we actually do the build
		// and save the result to the cache that BuiltTarget is not
		// the cached binary. Ideally, we would set BuiltTarget
		// to the cached path even in the first run, but if we
		// copy the binary to the cached path, and try to run it
		// in the same process, we'll run into the dreaded #22315
		// resulting in occasional ETXTBSYs. Instead of getting the
		// ETXTBSY and then retrying just don't use the cached path
		// on the first run if we're going to actually run the binary.
		if cached := linkAction.CachedExecutable(); cached != "" {
			return cached
		}
	}
	return linkAction.BuiltTarget()
}

func buildAndRunBuiltinTool(loaderstate *modload.State, ctx context.Context, toolName, tool string, args []string) {
	// Override GOOS and GOARCH for the build to build the tool using
	// the same GOOS and GOARCH as this go command.
	cfg.ForceHost()

	// Ignore go.mod and go.work: we don't need them, and we want to be able
	// to run the tool even if there's an issue with the module or workspace the
	// user happens to be in.
	loaderstate.RootMode = modload.NoRoot

	runFunc := func(b *work.Builder, ctx context.Context, a *work.Action) error {
		cmdline := str.StringList(builtTool(a), a.Args)
		return runBuiltTool(toolName, nil, cmdline)
	}

	buildAndRunTool(loaderstate, ctx, tool, args, runFunc)
}

func buildAndRunModtool(loaderstate *modload.State, ctx context.Context, toolName, tool string, args []string) {
	runFunc := func(b *work.Builder, ctx context.Context, a *work.Action) error {
		// Use the ExecCmd to run the binary, as go run does. ExecCmd allows users
		// to provide a runner to run the binary, for example a simulator for binaries
		// that are cross-compiled to a different platform.
		cmdline := str.StringList(work.FindExecCmd(), builtTool(a), a.Args)
		// Use same environment go run uses to start the executable:
		// the original environment with cfg.GOROOTbin added to the path.
		env := slices.Clip(cfg.OrigEnv)
		env = base.AppendPATH(env)

		return runBuiltTool(toolName, env, cmdline)
	}

	buildAndRunTool(loaderstate, ctx, tool, args, runFunc)
}

func buildAndRunTool(loaderstate *modload.State, ctx context.Context, tool string, args []string, runTool work.ActorFunc) {
	work.BuildInit(loaderstate)
	b := work.NewBuilder("", loaderstate.VendorDirOrEmpty)
	defer func() {
		if err := b.Close(); err != nil {
			base.Fatal(err)
		}
	}()

	pkgOpts := load.PackageOpts{MainOnly: true}
	p := load.PackagesAndErrors(loaderstate, ctx, pkgOpts, []string{tool})[0]
	p.Internal.OmitDebug = true
	p.Internal.ExeName = p.DefaultExecName()

	a1 := b.LinkAction(loaderstate, work.ModeBuild, work.ModeBuild, p)
	a1.CacheExecutable = true
	a := &work.Action{Mode: "go tool", Actor: runTool, Args: args, Deps: []*work.Action{a1}}
	b.Do(ctx, a)
}

func runBuiltTool(toolName string, env, cmdline []string) error {
	if toolN {
		fmt.Println(strings.Join(cmdline, " "))
		return nil
	}

	toolCmd := &exec.Cmd{
		Path:   cmdline[0],
		Args:   cmdline,
		Stdin:  os.Stdin,
		Stdout: os.Stdout,
		Stderr: os.Stderr,
		Env:    env,
	}
	err := toolCmd.Start()
	if err == nil {
		c := make(chan os.Signal, 100)
		signal.Notify(c)
		go func() {
			for sig := range c {
				toolCmd.Process.Signal(sig)
			}
		}()
		err = toolCmd.Wait()
		signal.Stop(c)
		close(c)
	}
	if err != nil {
		// Only print about the exit status if the command
		// didn't even run (not an ExitError) or if it didn't exit cleanly
		// or we're printing command lines too (-x mode).
		// Assume if command exited cleanly (even with non-zero status)
		// it printed any messages it wanted to print.
		e, ok := err.(*exec.ExitError)
		if !ok || !e.Exited() || cfg.BuildX {
			fmt.Fprintf(os.Stderr, "go tool %s: %s\n", toolName, err)
		}
		if ok {
			base.SetExitStatus(e.ExitCode())
		} else {
			base.SetExitStatus(1)
		}
	}

	return nil
}
