// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build ignore

// The run program is invoked via the dist tool.
// To invoke manually: go tool dist test -run api --no-rebuild
package main

import (
	"errors"
	"fmt"
	exec "internal/execabs"
	"internal/goversion"
	"io/fs"
	"log"
	"os"
	"path/filepath"
	"runtime"
	"strings"
)

func goCmd() string {
	var exeSuffix string
	if runtime.GOOS == "windows" {
		exeSuffix = ".exe"
	}
	path := filepath.Join(runtime.GOROOT(), "bin", "go"+exeSuffix)
	if _, err := os.Stat(path); err == nil {
		return path
	}
	return "go"
}

var goroot string

func main() {
	log.SetFlags(0)
	goroot = os.Getenv("GOROOT") // should be set by run.{bash,bat}
	if goroot == "" {
		log.Fatal("No $GOROOT set.")
	}

	apiDir := filepath.Join(goroot, "api")
	out, err := exec.Command(goCmd(), "tool", "api",
		"-c", findAPIDirFiles(apiDir),
		allowNew(apiDir),
		"-next", filepath.Join(apiDir, "next.txt"),
		"-except", filepath.Join(apiDir, "except.txt")).CombinedOutput()
	if err != nil {
		log.Fatalf("Error running API checker: %v\n%s", err, out)
	}
	fmt.Print(string(out))
}

// findAPIDirFiles returns a comma-separated list of Go API files
// (go1.txt, go1.1.txt, etc.) located in apiDir.
func findAPIDirFiles(apiDir string) string {
	dir, err := os.Open(apiDir)
	if err != nil {
		log.Fatal(err)
	}
	defer dir.Close()
	fs, err := dir.Readdirnames(-1)
	if err != nil {
		log.Fatal(err)
	}
	var apiFiles []string
	for _, fn := range fs {
		if strings.HasPrefix(fn, "go1") {
			apiFiles = append(apiFiles, filepath.Join(apiDir, fn))
		}
	}
	return strings.Join(apiFiles, ",")
}

// allowNew returns the -allow_new flag to use for the 'go tool api' invocation.
func allowNew(apiDir string) string {
	// Verify that the api/go1.n.txt for previous Go version exists.
	// It definitely should, otherwise it's a signal that the logic below may be outdated.
	if _, err := os.Stat(filepath.Join(apiDir, fmt.Sprintf("go1.%d.txt", goversion.Version-1))); err != nil {
		log.Fatalln("Problem with api file for previous release:", err)
	}

	// See whether the api/go1.n.txt for this Go version has been created.
	// (As of April 2021, it gets created during the release of the first Beta.)
	_, err := os.Stat(filepath.Join(apiDir, fmt.Sprintf("go1.%d.txt", goversion.Version)))
	if errors.Is(err, fs.ErrNotExist) {
		// It doesn't exist, so we're in development or before Beta 1.
		// At this stage, unmentioned API additions are deemed okay.
		// (They will be quietly shown in API check output, but the test won't fail).
		return "-allow_new=true"
	} else if err == nil {
		// The api/go1.n.txt for this Go version has been created,
		// so we're definitely past Beta 1 in the release cycle.
		//
		// From this point, enforce that api/go1.n.txt is an accurate and complete
		// representation of what's going into the release by failing API check if
		// there are API additions (a month into the freeze, there shouldn't be many).
		//
		// See golang.org/issue/43956.
		return "-allow_new=false"
	} else {
		log.Fatal(err)
	}
	panic("unreachable")
}
