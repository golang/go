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

// Borrowed from (internal/imports/mod.go:620)
// This function will return the main go.mod file for this folder if it exists and whether the -modfile
// flag exists for this version of go.
func modfileFlagExists(ctx context.Context, folder string, env []string) (string, bool, error) {
	const format = `{{.GoMod}}
{{range context.ReleaseTags}}{{if eq . "go1.14"}}{{.}}{{end}}{{end}}
`
	stdout, err := source.InvokeGo(ctx, folder, env, "list", "-m", "-f", format)
	if err != nil {
		return "", false, err
	}
	lines := strings.Split(stdout.String(), "\n")
	if len(lines) < 2 {
		return "", false, errors.Errorf("unexpected stdout: %q", stdout)
	}
	return lines[0], lines[1] == "go1.14", nil
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
	f, err := ioutil.TempFile("", "go.*.mod")
	if err != nil {
		return nil, err
	}
	defer f.Close()
	// Copy the current go.mod file into the temporary go.mod file.
	origFile, err := os.Open(modfile)
	if err != nil {
		return nil, err
	}
	defer origFile.Close()
	if _, err := io.Copy(f, origFile); err != nil {
		return nil, err
	}
	if err := f.Close(); err != nil {
		return nil, err
	}
	return &modfiles{real: modfile, temp: f.Name()}, nil
}
