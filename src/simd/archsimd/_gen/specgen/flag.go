// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package specgen

import (
	"fmt"
	"os"
	"path/filepath"
	"simd/archsimd/_gen/gentools"
)

// FindSpecDir returns the path to the standard spec package
// GOROOT/simd/internal/spec. If GOROOT is "", it uses gentools.DefaultGOROOT().
func FindSpecDir(goroot string) (string, error) {
	if goroot == "" {
		goroot = gentools.DefaultGOROOT()
		if goroot == "" {
			return "", fmt.Errorf("could not find GOROOT")
		}
	}
	path := filepath.Join(goroot, "src/simd/internal/spec")
	if _, err := os.Stat(path); err != nil {
		return "", fmt.Errorf("could not find spec package: %w (this tool requires a complete Go checkout)", err)
	}
	return path, nil
}

func MustFindSpecDir(goroot string) string {
	path, err := FindSpecDir(goroot)
	if err != nil {
		fmt.Fprintln(os.Stderr, err.Error())
		os.Exit(1)
	}
	return path
}
