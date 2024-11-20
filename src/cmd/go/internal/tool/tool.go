// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package tool implements the “go tool” command.
package tool

import (
	"cmd/internal/telemetry/counter"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"go/build"
	"internal/platform"
	"os"
	"os/exec"
	"os/signal"
	"sort"
	"strings"

	"cmd/go/internal/base"
	"cmd/go/internal/cfg"
)

var CmdTool = &base.Command{
	Run:       runTool,
	UsageLine: "go tool [-n] command [args...]",
	Short:     "run specified go tool",
	Long: `
Tool runs the go tool command identified by the arguments.
With no arguments it prints the list of known tools.

The -n flag causes tool to print the command that would be
executed but not execute it.

For more about each tool command, see 'go doc cmd/<command>'.
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
	CmdTool.Flag.BoolVar(&toolN, "n", false, "")
}

func runTool(ctx context.Context, cmd *base.Command, args []string) {
	if len(args) == 0 {
		counter.Inc("go/subcommand:tool")
		listTools()
		return
	}
	toolName := args[0]
	// The tool name must be lower-case letters, numbers or underscores.
	for _, c := range toolName {
		switch {
		case 'a' <= c && c <= 'z', '0' <= c && c <= '9', c == '_':
		default:
			fmt.Fprintf(os.Stderr, "go: bad tool name %q\n", toolName)
			base.SetExitStatus(2)
			return
		}
	}

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

		counter.Inc("go/subcommand:tool-unknown")
		// Emit the usual error for the missing tool.
		_ = base.Tool(toolName)
	} else {
		// Increment a counter for the tool subcommand with the tool name.
		counter.Inc("go/subcommand:tool-" + toolName)
	}

	if toolN {
		cmd := toolPath
		if len(args) > 1 {
			cmd += " " + strings.Join(args[1:], " ")
		}
		fmt.Printf("%s\n", cmd)
		return
	}
	args[0] = toolPath // in case the tool wants to re-exec itself, e.g. cmd/dist
	toolCmd := &exec.Cmd{
		Path:   toolPath,
		Args:   args,
		Stdin:  os.Stdin,
		Stdout: os.Stdout,
		Stderr: os.Stderr,
	}
	err = toolCmd.Start()
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
		// didn't even run (not an ExitError) or it didn't exit cleanly
		// or we're printing command lines too (-x mode).
		// Assume if command exited cleanly (even with non-zero status)
		// it printed any messages it wanted to print.
		if e, ok := err.(*exec.ExitError); !ok || !e.Exited() || cfg.BuildX {
			fmt.Fprintf(os.Stderr, "go tool %s: %s\n", toolName, err)
		}
		base.SetExitStatus(1)
		return
	}
}

// listTools prints a list of the available tools in the tools directory.
func listTools() {
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
