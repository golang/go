// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cache

import (
	"context"
	"io"
	"io/ioutil"
	"os"
	"path/filepath"
	"strings"

	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/lsp/telemetry"
	"golang.org/x/tools/internal/span"
	"golang.org/x/tools/internal/telemetry/log"
	errors "golang.org/x/xerrors"
)

func (v *view) modFiles(ctx context.Context) (span.URI, span.URI, error) {
	// Don't return errors if the view is not a module.
	if v.mod == nil {
		return "", "", nil
	}
	// Copy the real go.mod file content into the temp go.mod file.
	origFile, err := os.Open(v.mod.realMod.Filename())
	if err != nil {
		return "", "", err
	}
	defer origFile.Close()

	tempFile, err := os.OpenFile(v.mod.tempMod.Filename(), os.O_WRONLY, os.ModePerm)
	if err != nil {
		return "", "", err
	}
	defer tempFile.Close()

	if _, err := io.Copy(tempFile, origFile); err != nil {
		return "", "", err
	}
	return v.mod.realMod, v.mod.tempMod, nil
}

// This function will return the main go.mod file for this folder if it exists and whether the -modfile
// flag exists for this version of go.
func (v *view) modfileFlagExists(ctx context.Context, env []string) (string, bool, error) {
	// Check the go version by running "go list" with modules off.
	// Borrowed from internal/imports/mod.go:620.
	const format = `{{range context.ReleaseTags}}{{if eq . "go1.14"}}{{.}}{{end}}{{end}}`
	folder := v.folder.Filename()
	stdout, err := source.InvokeGo(ctx, folder, append(env, "GO111MODULE=off"), "list", "-e", "-f", format)
	if err != nil {
		return "", false, err
	}
	// If the output is not go1.14 or an empty string, then it could be an error.
	lines := strings.Split(stdout.String(), "\n")
	if len(lines) < 2 && stdout.String() != "" {
		log.Error(ctx, "unexpected stdout when checking for go1.14", errors.Errorf("%q", stdout), telemetry.Directory.Of(folder))
		return "", false, nil
	}
	modfile := strings.TrimSpace(v.gomod)
	if modfile == os.DevNull {
		return "", false, errors.Errorf("unable to detect a go.mod file in %s", v.folder)
	}
	return modfile, lines[0] == "go1.14", nil
}

func (v *view) setModuleInformation(ctx context.Context, enabled bool) error {
	// The user has disabled the use of the -modfile flag.
	if !enabled {
		log.Print(ctx, "using the -modfile flag is disabled", telemetry.Directory.Of(v.folder))
		return nil
	}
	modFile, flagExists, err := v.modfileFlagExists(ctx, v.Options().Env)
	if err != nil {
		return err
	}
	// The user's version of Go does not support the -modfile flag.
	if !flagExists {
		return nil
	}
	if modFile == "" || modFile == os.DevNull {
		return errors.Errorf("unable to detect a go.mod file in %s", v.folder)
	}
	// Copy the current go.mod file into the temporary go.mod file.
	// The file's name will be of the format go.1234.mod.
	// It's temporary go.sum file should have the corresponding format of go.1234.sum.
	tempModFile, err := ioutil.TempFile("", "go.*.mod")
	if err != nil {
		return err
	}
	defer tempModFile.Close()

	origFile, err := os.Open(modFile)
	if err != nil {
		return err
	}
	defer origFile.Close()

	if _, err := io.Copy(tempModFile, origFile); err != nil {
		return err
	}
	v.mod = &moduleInformation{
		realMod: span.FileURI(modFile),
		tempMod: span.FileURI(tempModFile.Name()),
	}
	// Copy go.sum file as well (if there is one).
	sumFile := filepath.Join(filepath.Dir(modFile), "go.sum")
	stat, err := os.Stat(sumFile)
	if err != nil || !stat.Mode().IsRegular() {
		return nil
	}
	contents, err := ioutil.ReadFile(sumFile)
	if err != nil {
		return err
	}
	if err := ioutil.WriteFile(v.mod.tempSumFile(), contents, stat.Mode()); err != nil {
		return err
	}
	return nil
}

// tempSumFile returns the path to the copied temporary go.sum file.
// It simply replaces the extension of the temporary go.mod file with "sum".
func (mod *moduleInformation) tempSumFile() string {
	tmp := mod.tempMod.Filename()
	return tmp[:len(tmp)-len("mod")] + "sum"
}
