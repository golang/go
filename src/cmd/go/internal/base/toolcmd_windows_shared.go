//go:build windows && sharedtool

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package base

import (
	"fmt"
	"os/exec"
	"path/filepath"
	"strings"
)

// ToolSuffix reports the suffix used by the shared Windows tool package.
func ToolSuffix() string { return ".py" }

// ToolCommandLine runs a shared tool script through the installed Python 3
// launcher or interpreter. Windows cannot execute a .py file directly with
// CreateProcess, which is why the Go command must make this translation.
func ToolCommandLine(cmdline []string) ([]string, error) {
	if len(cmdline) == 0 || !strings.EqualFold(filepath.Ext(cmdline[0]), ".py") {
		return cmdline, nil
	}

	for _, candidate := range []struct {
		name string
		args []string
	}{
		{name: "py.exe", args: []string{"-3"}},
		{name: "py", args: []string{"-3"}},
		{name: "python.exe"},
		{name: "python"},
	} {
		python, err := exec.LookPath(candidate.name)
		if err != nil {
			continue
		}
		result := make([]string, 0, len(cmdline)+1+len(candidate.args))
		result = append(result, python)
		result = append(result, candidate.args...)
		result = append(result, cmdline...)
		return result, nil
	}
	return nil, fmt.Errorf("Python 3 is required to run Go tool script %s; install Python or add py.exe/python.exe to PATH", cmdline[0])
}
