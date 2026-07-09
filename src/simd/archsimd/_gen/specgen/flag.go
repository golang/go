// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package specgen

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
)

// FindSpecDir returns the path to the standard spec package
// $GOROOT/simd/internal/spec.
func FindSpecDir() (string, error) {
	goroot, err := goEnvGoroot()
	if err != nil {
		return "", fmt.Errorf("could not find GOROOT: %w", err)
	}
	path := filepath.Join(goroot, "src/simd/internal/spec")
	if _, err := os.Stat(path); err != nil {
		return "", fmt.Errorf("could not find spec package: %w (this tool requires a complete Go checkout)", err)
	}
	return path, nil
}

func MustFindSpecDir() string {
	path, err := FindSpecDir()
	if err != nil {
		fmt.Fprintln(os.Stderr, err.Error())
		os.Exit(1)
	}
	return path
}

func goEnvGoroot() (string, error) {
	out, err := exec.Command("go", "env", "GOROOT").Output()
	if err != nil {
		return "", err
	}
	return strings.TrimSpace(string(out)), nil
}
