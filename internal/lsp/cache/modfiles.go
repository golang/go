// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cache

import (
	"context"
	"io"
	"io/ioutil"
	"os"
	"strings"

	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/lsp/telemetry"
	"golang.org/x/tools/internal/telemetry/log"
	errors "golang.org/x/xerrors"
)

// This function will return the main go.mod file for this folder if it exists and whether the -modfile
// flag exists for this version of go.
func modfileFlagExists(ctx context.Context, folder string, env []string) (string, bool, error) {
	// Borrowed from (internal/imports/mod.go:620)
	const format = `{{range context.ReleaseTags}}{{if eq . "go1.14"}}{{.}}{{end}}{{end}}`
	// Check the go version by running "go list" with modules off.
	stdout, err := source.InvokeGo(ctx, folder, append(env, "GO111MODULE=off"), "list", "-f", format)
	if err != nil {
		return "", false, err
	}
	lines := strings.Split(stdout.String(), "\n")
	if len(lines) < 2 {
		return "", false, errors.Errorf("unexpected stdout: %q", stdout)
	}
	// Get the go.mod file associated with this module.
	b, err := source.InvokeGo(ctx, folder, env, "env", "GOMOD")
	if err != nil {
		return "", false, err
	}
	modfile := strings.TrimSpace(b.String())
	if modfile == os.DevNull {
		return "", false, errors.Errorf("go env GOMOD did not detect a go.mod file in this folder")
	}
	return modfile, lines[0] == "go1.14", nil
}

// The function getModfiles will return the go.mod files associated with the directory that is passed in.
func getModfiles(ctx context.Context, folder string, options source.Options) (*modfiles, error) {
	if !options.TempModfile {
		log.Print(ctx, "using the -modfile flag is disabled", telemetry.Directory.Of(folder))
		return nil, nil
	}
	modfile, flagExists, err := modfileFlagExists(ctx, folder, options.Env)
	if err != nil {
		return nil, err
	}
	if !flagExists {
		return nil, nil
	}
	if modfile == "" || modfile == os.DevNull {
		return nil, errors.Errorf("go env GOMOD cannot detect a go.mod file in this folder")
	}
	// Copy the current go.mod file into the temporary go.mod file.
	tempFile, err := ioutil.TempFile("", "go.*.mod")
	if err != nil {
		return nil, err
	}
	defer tempFile.Close()
	origFile, err := os.Open(modfile)
	if err != nil {
		return nil, err
	}
	defer origFile.Close()
	if _, err := io.Copy(tempFile, origFile); err != nil {
		return nil, err
	}
	copySumFile(modfile, tempFile.Name())
	return &modfiles{real: modfile, temp: tempFile.Name()}, nil
}

func copySumFile(realFile, tempFile string) {
	realSum := realFile[0:len(realFile)-3] + "sum"
	tempSum := tempFile[0:len(tempFile)-3] + "sum"
	stat, err := os.Stat(realSum)
	if err != nil || !stat.Mode().IsRegular() {
		return
	}
	contents, err := ioutil.ReadFile(realSum)
	if err != nil {
		return
	}
	ioutil.WriteFile(tempSum, contents, stat.Mode())
}
