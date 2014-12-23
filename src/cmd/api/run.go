// Copyright 2013 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build ignore

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
	"net/http"
	"os"
	"os/exec"
	"os/user"
	"path/filepath"
	"runtime"
	"strings"
)

// goToolsVersion is the git revision of the x/tools subrepo we need
// to build cmd/api.  This only needs to be updated whenever a go/types
// bug fix is needed by the cmd/api tool.
const goToolsVersion = "875ff2496f865e" // aka hg 6698ca2900e2

var goroot string

func main() {
	log.SetFlags(0)
	goroot = os.Getenv("GOROOT") // should be set by run.{bash,bat}
	if goroot == "" {
		log.Fatal("No $GOROOT set.")
	}
	_, err := exec.LookPath("git")
	if err != nil {
		fmt.Println("Skipping cmd/api checks; git not available")
		return
	}

	gopath := prepGoPath()

	cmd := exec.Command("go", "install", "--tags=api_tool", "cmd/api")
	cmd.Env = append(filterOut(os.Environ(), "GOARCH", "GOPATH"), "GOPATH="+gopath)
	out, err := cmd.CombinedOutput()
	if err != nil {
		log.Fatalf("Error installing cmd/api: %v\n%s", err, out)
	}

	out, err = exec.Command("go", "tool", "api",
		"-c", file("go1", "go1.1", "go1.2", "go1.3", "go1.4"),
		"-next", file("next"),
		"-except", file("except")).CombinedOutput()
	if err != nil {
		log.Fatalf("Error running API checker: %v\n%s", err, out)
	}
	fmt.Print(string(out))
}

// filterOut returns a copy of the src environment without environment
// variables from remove.
// TODO: delete when issue 6201 is fixed.
func filterOut(src []string, remove ...string) (out []string) {
S:
	for _, s := range src {
		for _, r := range remove {
			if strings.HasPrefix(s, r) && strings.HasPrefix(s, r+"=") {
				continue S
			}
		}
		out = append(out, s)
	}
	return
}

// file expands s to $GOROOT/api/s.txt.
// If there are more than 1, they're comma-separated.
func file(s ...string) string {
	if len(s) > 1 {
		return file(s[0]) + "," + file(s[1:]...)
	}
	return filepath.Join(goroot, "api", s[0]+".txt")
}

// prepGoPath returns a GOPATH for the "go" tool to compile the API tool with.
// It tries to re-use a go.tools checkout from a previous run if possible,
// else it hg clones it.
func prepGoPath() string {
	// Use a builder-specific temp directory name, so builders running
	// two copies don't trample on each other: https://golang.org/issue/9407
	// We don't use io.TempDir or a PID or timestamp here because we do
	// want this to be stable between runs, to minimize "git clone" calls
	// in the common case.
	var tempBase = fmt.Sprintf("go.tools.TMP.%s.%s", runtime.GOOS, runtime.GOARCH)

	username := ""
	u, err := user.Current()
	if err == nil {
		username = u.Username
	} else {
		username = os.Getenv("USER")
		if username == "" {
			username = "nobody"
		}
	}

	// The GOPATH we'll return
	gopath := filepath.Join(os.TempDir(), "gopath-api-"+cleanUsername(username)+"-"+cleanUsername(strings.Fields(runtime.Version())[0]), goToolsVersion)

	// cloneDir is where we run "git clone".
	cloneDir := filepath.Join(gopath, "src", "code.google.com", "p")

	// The dir we clone into. We only atomically rename it to finalDir on
	// clone success.
	tmpDir := filepath.Join(cloneDir, tempBase)

	// finalDir is where the checkout will live once it's complete.
	finalDir := filepath.Join(cloneDir, "go.tools")

	if goToolsCheckoutGood(finalDir) {
		return gopath
	}
	os.RemoveAll(finalDir) // in case it's there but corrupt
	os.RemoveAll(tmpDir)   // in case of aborted hg clone before

	if err := os.MkdirAll(cloneDir, 0700); err != nil {
		log.Fatal(err)
	}
	cmd := exec.Command("git", "clone", "https://go.googlesource.com/tools", tempBase)
	cmd.Dir = cloneDir
	out, err := cmd.CombinedOutput()
	if err != nil {
		if _, err := http.Head("http://ip.appspot.com/"); err != nil {
			log.Printf("# Skipping API check; network appears to be unavailable")
			os.Exit(0)
		}
		log.Fatalf("Error running git clone on x/tools: %v\n%s", err, out)
	}
	cmd = exec.Command("git", "reset", "--hard", goToolsVersion)
	cmd.Dir = tmpDir
	out, err = cmd.CombinedOutput()
	if err != nil {
		log.Fatalf("Error updating x/tools in %v to %v: %v, %s", tmpDir, goToolsVersion, err, out)
	}

	if err := os.Rename(tmpDir, finalDir); err != nil {
		if os.IsExist(err) {
			// A different builder beat us into putting this repo into
			// its final place. But that's fine; if it's there, it's
			// the right version and we can use it.
			//
			// This happens on the Go project's Windows builders
			// especially, where we have two builders (386 and amd64)
			// running at the same time, trying to compete for moving
			// it into place.
			os.RemoveAll(tmpDir)
		} else {
			log.Fatal(err)
		}
	}
	return gopath
}

func cleanUsername(n string) string {
	b := make([]rune, len(n))
	for i, r := range n {
		if r == '\\' || r == '/' || r == ':' {
			b[i] = '_'
		} else {
			b[i] = r
		}
	}
	return string(b)
}

func goToolsCheckoutGood(dir string) bool {
	if _, err := os.Stat(dir); err != nil {
		return false
	}

	cmd := exec.Command("git", "rev-parse", "HEAD")
	cmd.Dir = dir
	out, err := cmd.Output()
	if err != nil {
		return false
	}
	id := strings.TrimSpace(string(out))
	if !strings.HasPrefix(id, goToolsVersion) {
		return false
	}

	cmd = exec.Command("git", "status", "--porcelain")
	cmd.Dir = dir
	out, err = cmd.Output()
	if err != nil || strings.TrimSpace(string(out)) != "" {
		return false
	}
	return true
}
