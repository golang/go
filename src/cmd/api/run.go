// +build from_src_run

// Copyright 2013 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// The run program is invoked via "go run" from src/run.bash or
// src/run.bat conditionally builds and runs the cmd/api tool.
//
// TODO(bradfitz): the "conditional" condition is always true.
// We should only do this if the user has the hg codereview extension
// enabled and verifies that the go.tools subrepo is checked out with
// a suitably recently version. In prep for the cmd/api rewrite.
package main

import (
	"fmt"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
)

// goToolsVersion is the hg revision of the go.tools subrepo we need
// to build cmd/api.  This only needs to be updated whenever a go/types
// bug fix is needed by the cmd/api tool.
const goToolsVersion = "6698ca2900e2"

var goroot string

func main() {
	log.SetFlags(0)
	goroot = os.Getenv("GOROOT") // should be set by run.{bash,bat}
	if goroot == "" {
		log.Fatal("No $GOROOT set.")
	}
	isGoDeveloper := exec.Command("hg", "pq").Run() == nil
	if !isGoDeveloper && !forceAPICheck() {
		fmt.Println("Skipping cmd/api checks; hg codereview extension not available and GO_FORCE_API_CHECK not set")
		return
	}

	gopath := prepGoPath()

	cmd := exec.Command("go", "install", "--tags=api_tool", "cmd/api")
	cmd.Env = append([]string{"GOPATH=" + gopath}, os.Environ()...)
	out, err := cmd.CombinedOutput()
	if err != nil {
		log.Fatalf("Error installing cmd/api: %v\n%s", err, out)
	}

	out, err = exec.Command("go", "tool", "api",
		"-c", file("go1", "go1.1"),
		"-next", file("next"),
		"-except", file("except")).CombinedOutput()
	if err != nil {
		log.Fatalf("Error running API checker: %v\n%s", err, out)
	}
}

// file expands s to $GOROOT/api/s.txt.
// If there are more than 1, they're comma-separated.
func file(s ...string) string {
	if len(s) > 1 {
		return file(s[0]) + "," + file(s[1:]...)
	}
	return filepath.Join(goroot, "api", s[0]+".txt")
}

// GO_FORCE_API_CHECK is set by builders.
func forceAPICheck() bool {
	v, _ := strconv.ParseBool(os.Getenv("GO_FORCE_API_CHECK"))
	return v
}

// prepGoPath returns a GOPATH for the "go" tool to compile the API tool with.
// It tries to re-use a go.tools checkout from a previous run if possible,
// else it hg clones it.
func prepGoPath() string {
	const tempBase = "go.tools.TMP"

	// The GOPATH we'll return
	gopath := filepath.Join(os.TempDir(), "gopath-api", goToolsVersion)

	// cloneDir is where we run "hg clone".
	cloneDir := filepath.Join(gopath, "src", "code.google.com", "p")

	// The dir we clone into. We only atomically rename it to finalDir on
	// clone success.
	tmpDir := filepath.Join(cloneDir, tempBase)

	// finalDir is where the checkout will live once it's complete.
	// If this exists already, we're done.
	finalDir := filepath.Join(cloneDir, "go.tools")

	if fi, err := os.Stat(finalDir); err == nil && fi.IsDir() {
		return gopath
	}

	if err := os.MkdirAll(cloneDir, 0700); err != nil {
		log.Fatal(err)
	}
	cmd := exec.Command("hg",
		"clone", "--rev="+goToolsVersion,
		"https://code.google.com/p/go.tools",
		tempBase)
	cmd.Dir = cloneDir
	out, err := cmd.CombinedOutput()
	if err != nil {
		log.Fatalf("Error running hg clone on go.tools: %v\n%s", err, out)
	}
	if err := os.Rename(tmpDir, finalDir); err != nil {
		log.Fatal(err)
	}
	return gopath
}
