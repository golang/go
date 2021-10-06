// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package base

import (
	"fmt"
	"go/build"
	"os"
	"path/filepath"
	"runtime"

	"cmd/go/internal/cfg"
)

// Configuration for finding tool binaries.
var (
	ToolGOOS      = runtime.GOOS
	ToolGOARCH    = runtime.GOARCH
	ToolIsWindows = ToolGOOS == "windows"
	ToolDir       = build.ToolDir
)

const ToolWindowsExtension = ".exe"

// Tool returns the path to the named tool (for example, "vet").
// If the tool cannot be found, Tool exits the process.
func Tool(toolName string) string {
	toolPath := filepath.Join(ToolDir, toolName)
	if ToolIsWindows {
		toolPath += ToolWindowsExtension
	}
	if len(cfg.BuildToolexec) > 0 {
		return toolPath
	}
	// Give a nice message if there is no tool with that name.
	if _, err := os.Stat(toolPath); err != nil {
		fmt.Fprintf(os.Stderr, "go: no such tool %q\n", toolName)
		SetExitStatus(2)
		Exit()
	}
	return toolPath
}
