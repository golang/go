// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"go/build"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
)

var cmdTool = &Command{
	Run:       runTool,
	UsageLine: "tool [-n] command [args...]",
	Short:     "run specified go tool",
	Long: `
Tool runs the go tool command identified by the arguments.
With no arguments it prints the list of known tools.

The -n flag causes tool to print the command that would be
executed but not execute it.

For more about each tool command, see 'go tool command -h'.
`,
}

var (
	toolGOOS      = runtime.GOOS
	toolGOARCH    = runtime.GOARCH
	toolIsWindows = toolGOOS == "windows"
	toolDir       = build.ToolDir

	toolN bool
)

func init() {
	cmdTool.Flag.BoolVar(&toolN, "n", false, "")
}

const toolWindowsExtension = ".exe"

func tool(name string) string {
	p := filepath.Join(toolDir, name)
	if toolIsWindows {
		p += toolWindowsExtension
	}
	return p
}

func runTool(cmd *Command, args []string) {
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
			setExitStatus(2)
			return
		}
	}
	toolPath := tool(toolName)
	// Give a nice message if there is no tool with that name.
	if _, err := os.Stat(toolPath); err != nil {
		fmt.Fprintf(os.Stderr, "go tool: no such tool %q\n", toolName)
		setExitStatus(3)
		return
	}

	if toolN {
		fmt.Printf("%s %s\n", toolPath, strings.Join(args[1:], " "))
		return
	}
	toolCmd := &exec.Cmd{
		Path:   toolPath,
		Args:   args,
		Stdin:  os.Stdin,
		Stdout: os.Stdout,
		Stderr: os.Stderr,
	}
	err := toolCmd.Run()
	if err != nil {
		fmt.Fprintf(os.Stderr, "go tool %s: %s\n", toolName, err)
		setExitStatus(1)
		return
	}
}

// listTools prints a list of the available tools in the tools directory.
func listTools() {
	f, err := os.Open(toolDir)
	if err != nil {
		fmt.Fprintf(os.Stderr, "go tool: no tool directory: %s\n", err)
		setExitStatus(2)
		return
	}
	defer f.Close()
	names, err := f.Readdirnames(-1)
	if err != nil {
		fmt.Fprintf(os.Stderr, "go tool: can't read directory: %s\n", err)
		setExitStatus(2)
		return
	}

	sort.Strings(names)
	for _, name := range names {
		// Unify presentation by going to lower case.
		name = strings.ToLower(name)
		// If it's windows, don't show the .exe suffix.
		if toolIsWindows && strings.HasSuffix(name, toolWindowsExtension) {
			name = name[:len(name)-len(toolWindowsExtension)]
		}
		fmt.Println(name)
	}
}
