// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.18
// +build go1.18

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
	"os/exec"

	"github.com/google/go-cmp/cmp"
	"golang.org/x/tools/gopls/internal/lsp/source"
)

const usage = `api-diff <previous version> [<current version>]

Compare the API of two gopls versions. If the second argument is provided, it
will be used as the new version to compare against. Otherwise, compare against
the current API.
`

func main() {
	flag.Parse()

	if flag.NArg() < 1 || flag.NArg() > 2 {
		fmt.Fprint(os.Stderr, usage)
		os.Exit(2)
	}

	oldVer := flag.Arg(0)
	newVer := ""
	if flag.NArg() == 2 {
		newVer = flag.Arg(1)
	}

	apiDiff, err := diffAPI(oldVer, newVer)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println("\n" + apiDiff)
}

func diffAPI(oldVer, newVer string) (string, error) {
	ctx := context.Background()
	previousAPI, err := loadAPI(ctx, oldVer)
	if err != nil {
		return "", fmt.Errorf("loading %s: %v", oldVer, err)
	}
	var currentAPI *source.APIJSON
	if newVer == "" {
		currentAPI = source.GeneratedAPIJSON
	} else {
		var err error
		currentAPI, err = loadAPI(ctx, newVer)
		if err != nil {
			return "", fmt.Errorf("loading %s: %v", newVer, err)
		}
	}

	return cmp.Diff(previousAPI, currentAPI), nil
}

func loadAPI(ctx context.Context, version string) (*source.APIJSON, error) {
	ver := fmt.Sprintf("golang.org/x/tools/gopls@%s", version)
	cmd := exec.Command("go", "run", ver, "api-json")

	stdout := &bytes.Buffer{}
	stderr := &bytes.Buffer{}
	cmd.Stdout = stdout
	cmd.Stderr = stderr

	if err := cmd.Run(); err != nil {
		return nil, fmt.Errorf("go run failed: %v; stderr:\n%s", err, stderr)
	}
	apiJson := &source.APIJSON{}
	if err := json.Unmarshal(stdout.Bytes(), apiJson); err != nil {
		return nil, fmt.Errorf("unmarshal: %v", err)
	}
	return apiJson, nil
}
