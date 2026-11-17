// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
)

func main() {
	args := os.Args[1:]
	if len(args) == 0 {
		os.Exit(2)
	}
	spec := os.Getenv("GOPARITY_MUTANT_TOOL")
	if spec != "" {
		name, path, ok := strings.Cut(spec, "=")
		if !ok || path == "" {
			fmt.Fprintf(os.Stderr, "goparity toolexec: bad GOPARITY_MUTANT_TOOL %q\n", spec)
			os.Exit(2)
		}
		if filepath.Base(args[0]) == name {
			args[0] = path
		}
	}
	cmd := exec.Command(args[0], args[1:]...)
	cmd.Stdin = os.Stdin
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	err := cmd.Run()
	if err == nil {
		os.Exit(0)
	}
	if ee, ok := errors.AsType[*exec.ExitError](err); ok {
		os.Exit(ee.ExitCode())
	}
	fmt.Fprintln(os.Stderr, "goparity toolexec:", err)
	os.Exit(1)
}
