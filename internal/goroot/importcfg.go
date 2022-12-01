// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package goroot is a copy of package internal/goroot
// in the main GO repot. It provides a utility to produce
// an importcfg and import path to package file map mapping
// standard library packages to the locations of their export
// data files.
package goroot

import (
	"bytes"
	"fmt"
	"os/exec"
	"strings"
	"sync"
)

// Importcfg returns an importcfg file to be passed to the
// Go compiler that contains the cached paths for the .a files for the
// standard library.
func Importcfg() (string, error) {
	var icfg bytes.Buffer

	m, err := PkgfileMap()
	if err != nil {
		return "", err
	}
	fmt.Fprintf(&icfg, "# import config")
	for importPath, export := range m {
		fmt.Fprintf(&icfg, "\npackagefile %s=%s", importPath, export)
	}
	s := icfg.String()
	return s, nil
}

var (
	stdlibPkgfileMap map[string]string
	stdlibPkgfileErr error
	once             sync.Once
)

// PkgfileMap returns a map of package paths to the location on disk
// of the .a file for the package.
// The caller must not modify the map.
func PkgfileMap() (map[string]string, error) {
	once.Do(func() {
		m := make(map[string]string)
		output, err := exec.Command("go", "list", "-export", "-e", "-f", "{{.ImportPath}} {{.Export}}", "std", "cmd").Output()
		if err != nil {
			stdlibPkgfileErr = err
		}
		for _, line := range strings.Split(string(output), "\n") {
			if line == "" {
				continue
			}
			sp := strings.SplitN(line, " ", 2)
			if len(sp) != 2 {
				err = fmt.Errorf("determining pkgfile map: invalid line in go list output: %q", line)
				return
			}
			importPath, export := sp[0], sp[1]
			if export != "" {
				m[importPath] = export
			}
		}
		stdlibPkgfileMap = m
	})
	return stdlibPkgfileMap, stdlibPkgfileErr
}
