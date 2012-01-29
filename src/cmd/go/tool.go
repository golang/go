// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"go/build"
	"os"
	"os/exec"
	"sort"
	"strings"
)

var cmdTool = &Command{
	Run:       runTool,
	UsageLine: "tool command [args...]",
	Short:     "run specified go tool",
	Long: `
Tool runs the go tool command identified by the arguments.
With no arguments it prints the list of known tools.

For more about each tool command, see 'go tool command -h'.
`,
}

var (
	toolGoos       = build.DefaultContext.GOOS
	toolIsWindows  = toolGoos == "windows"
	toolBinToolDir = build.Path[0].BinDir() + "/go-tool"
)

const toolWindowsExtension = ".exe"

func runTool(cmd *Command, args []string) {
	if len(args) == 0 {
		listTools()
		return
	}
	tool := args[0]
	// The tool name must be lower-case letters and numbers.
	for _, c := range tool {
		switch {
		case 'a' <= c && c <= 'z', '0' <= c && c <= '9':
		default:
			fmt.Fprintf(os.Stderr, "go tool: bad tool name %q\n", tool)
			exitStatus = 2
			return
		}
	}
	toolPath := toolBinToolDir + "/" + tool
	if toolIsWindows {
		toolPath += toolWindowsExtension
	}
	// Give a nice message if there is no tool with that name.
	if _, err := os.Stat(toolPath); err != nil {
		fmt.Fprintf(os.Stderr, "go tool: no such tool %q\n", tool)
		exitStatus = 3
		return
	}
	toolCmd := &exec.Cmd{
		Path:   toolPath,
		Args:   args,
		Stdout: os.Stdout,
		Stderr: os.Stderr,
	}
	err := toolCmd.Run()
	if err != nil {
		fmt.Fprintf(os.Stderr, "go tool %s failed: %s\n", tool, err)
		exitStatus = 1
		return
	}
}

// listTools prints a list of the available tools in the go-tools directory.
func listTools() {
	toolDir, err := os.Open(toolBinToolDir)
	if err != nil {
		fmt.Fprintf(os.Stderr, "go tool: no tool directory: %s\n", err)
		exitStatus = 2
		return
	}
	names, err := toolDir.Readdirnames(-1)
	if err != nil {
		fmt.Fprintf(os.Stderr, "go tool: can't read directory: %s\n", err)
		exitStatus = 2
		return
	}
	sort.StringSlice(names).Sort()
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
