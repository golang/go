// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package base

import (
	"fmt"
	"go/build"
	"os"
	"path/filepath"

	"cmd/go/internal/cfg"
	"cmd/go/internal/par"
)

// Tool returns the path to the named tool (for example, "vet").
// If the tool cannot be found, Tool exits the process.
func Tool(toolName string) string {
	toolPath, err := ToolPath(toolName)
	if err != nil && len(cfg.BuildToolexec) == 0 {
		// Give a nice message if there is no tool with that name.
		fmt.Fprintf(os.Stderr, "go: no such tool %q\n", toolName)
		SetExitStatus(2)
		Exit()
	}
	return toolPath
}

// ToolPath returns the path at which we expect to find the named tool
// (for example, "vet"), and the error (if any) from statting that path.
func ToolPath(toolName string) (string, error) {
	toolPath := filepath.Join(build.ToolDir, toolName) + cfg.ToolExeSuffix()
	err := toolStatCache.Do(toolPath, func() error {
		_, err := os.Stat(toolPath)
		return err
	})
	return toolPath, err
}

var toolStatCache par.Cache[string, error]
