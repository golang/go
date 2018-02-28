// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build ignore

// The run program is invoked via the dist tool.
// To invoke manually: go tool dist test -run api --no-rebuild
package main

import (
	"fmt"
	"log"
	"os"
	"os/exec"
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
