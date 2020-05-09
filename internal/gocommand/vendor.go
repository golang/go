// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gocommand

import (
	"bytes"
	"context"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"

	"golang.org/x/mod/semver"
)

// ModuleJSON holds information about a module.
type ModuleJSON struct {
	Path      string      // module path
	Replace   *ModuleJSON // replaced by this module
	Main      bool        // is this the main module?
	Indirect  bool        // is this module only an indirect dependency of main module?
	Dir       string      // directory holding files for this module, if any
	GoMod     string      // path to go.mod file for this module, if any
	GoVersion string      // go version used in module
}

var modFlagRegexp = regexp.MustCompile(`-mod[ =](\w+)`)

// VendorEnabled reports whether vendoring is enabled. It takes a *Runner to execute Go commands
// with the supplied context.Context and Invocation. The Invocation can contain pre-defined fields,
// of which only Verb and Args are modified to run the appropriate Go command.
// Inspired by setDefaultBuildMod in modload/init.go
func VendorEnabled(ctx context.Context, inv Invocation, r *Runner) (*ModuleJSON, bool, error) {
	mainMod, go114, err := getMainModuleAnd114(ctx, inv, r)
	if err != nil {
		return nil, false, err
	}

	// We check the GOFLAGS to see if there is anything overridden or not.
	inv.Verb = "env"
	inv.Args = []string{"GOFLAGS"}
	stdout, err := r.Run(ctx, inv)
	if err != nil {
		return nil, false, err
	}
	goflags := string(bytes.TrimSpace(stdout.Bytes()))
	matches := modFlagRegexp.FindStringSubmatch(goflags)
	var modFlag string
	if len(matches) != 0 {
		modFlag = matches[1]
	}
	if modFlag != "" {
		// Don't override an explicit '-mod=' argument.
		return mainMod, modFlag == "vendor", nil
	}
	if mainMod == nil || !go114 {
		return mainMod, false, nil
	}
	// Check 1.14's automatic vendor mode.
	if fi, err := os.Stat(filepath.Join(mainMod.Dir, "vendor")); err == nil && fi.IsDir() {
		if mainMod.GoVersion != "" && semver.Compare("v"+mainMod.GoVersion, "v1.14") >= 0 {
			// The Go version is at least 1.14, and a vendor directory exists.
			// Set -mod=vendor by default.
			return mainMod, true, nil
		}
	}
	return mainMod, false, nil
}

// getMainModuleAnd114 gets the main module's information and whether the
// go command in use is 1.14+. This is the information needed to figure out
// if vendoring should be enabled.
func getMainModuleAnd114(ctx context.Context, inv Invocation, r *Runner) (*ModuleJSON, bool, error) {
	const format = `{{.Path}}
{{.Dir}}
{{.GoMod}}
{{.GoVersion}}
{{range context.ReleaseTags}}{{if eq . "go1.14"}}{{.}}{{end}}{{end}}
`
	inv.Verb = "list"
	inv.Args = []string{"-m", "-f", format}
	stdout, err := r.Run(ctx, inv)
	if err != nil {
		return nil, false, err
	}

	lines := strings.Split(stdout.String(), "\n")
	if len(lines) < 5 {
		return nil, false, fmt.Errorf("unexpected stdout: %q", stdout.String())
	}
	mod := &ModuleJSON{
		Path:      lines[0],
		Dir:       lines[1],
		GoMod:     lines[2],
		GoVersion: lines[3],
		Main:      true,
	}
	return mod, lines[4] == "go1.14", nil
}
