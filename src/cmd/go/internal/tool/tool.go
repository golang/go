// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package tool implements the ``go tool'' command.
package tool

import (
	"fmt"
	"os"
	"os/exec"
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
	CmdTool.Flag.BoolVar(&toolN, "n", false, "")
}

func runTool(cmd *base.Command, args []string) {
	if len(args) == 0 {
		listTools()
		return
	}
	toolName := args[0]
	// The tool name must be lower-case letters, numbers or underscores.
	for _, c := range toolName {
		switch {
		case 'a' <= c && c <= 'z', '0' <= c && c <= '9', c == '_':
		default:
			fmt.Fprintf(os.Stderr, "go tool: bad tool name %q\n", toolName)
			base.SetExitStatus(2)
			return
		}
	}
	toolPath := base.Tool(toolName)
	if toolPath == "" {
		return
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
	err := toolCmd.Run()
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
	f, err := os.Open(base.ToolDir)
	if err != nil {
		fmt.Fprintf(os.Stderr, "go tool: no tool directory: %s\n", err)
		base.SetExitStatus(2)
		return
	}
	defer f.Close()
	names, err := f.Readdirnames(-1)
	if err != nil {
		fmt.Fprintf(os.Stderr, "go tool: can't read directory: %s\n", err)
		base.SetExitStatus(2)
		return
	}

	sort.Strings(names)
	for _, name := range names {
		// Unify presentation by going to lower case.
		name = strings.ToLower(name)
		// If it's windows, don't show the .exe suffix.
		if base.ToolIsWindows && strings.HasSuffix(name, base.ToolWindowsExtension) {
			name = name[:len(name)-len(base.ToolWindowsExtension)]
		}
		// The tool directory used by gccgo will have other binaries
		// in addition to go tools. Only display go tools here.
		if cfg.BuildToolchainName == "gccgo" && !isGccgoTool(name) {
			continue
		}
		fmt.Println(name)
	}
}
