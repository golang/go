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

// This function will return the main go.mod file for this folder if it exists and whether the -modfile
// flag exists for this version of go.
func (v *view) modfileFlagExists(ctx context.Context, env []string) (bool, error) {
	// Check the go version by running "go list" with modules off.
	// Borrowed from internal/imports/mod.go:620.
	const format = `{{range context.ReleaseTags}}{{if eq . "go1.14"}}{{.}}{{end}}{{end}}`
	folder := v.folder.Filename()
	stdout, err := source.InvokeGo(ctx, folder, append(env, "GO111MODULE=off"), "list", "-e", "-f", format)
	if err != nil {
		return false, err
	}
	// If the output is not go1.14 or an empty string, then it could be an error.
	lines := strings.Split(stdout.String(), "\n")
	if len(lines) < 2 && stdout.String() != "" {
		log.Error(ctx, "unexpected stdout when checking for go1.14", errors.Errorf("%q", stdout), telemetry.Directory.Of(folder))
		return false, nil
	}
	return lines[0] == "go1.14", nil
}

func (v *view) setModuleInformation(ctx context.Context, gomod string, modfileFlagEnabled bool) error {
	modFile := strings.TrimSpace(gomod)
	if modFile == os.DevNull {
		return nil
	}
	v.mod = moduleInformation{
		realMod: span.FileURI(modFile),
	}
	// The user has disabled the use of the -modfile flag.
	if !modfileFlagEnabled {
		return nil
	}
	if modfileFlag, err := v.modfileFlagExists(ctx, v.Options().Env); err != nil {
		return err
	} else if !modfileFlag {
		return nil
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
	v.mod.tempMod = span.FileURI(tempModFile.Name())

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
	if err := ioutil.WriteFile(tempSumFile(tempModFile.Name()), contents, stat.Mode()); err != nil {
		return err
	}
	return nil
}

// tempSumFile returns the path to the copied temporary go.sum file.
// It simply replaces the extension of the temporary go.mod file with "sum".
func tempSumFile(filename string) string {
	if filename == "" {
		return ""
	}
	return filename[:len(filename)-len("mod")] + "sum"
}
