// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// The go-contrib-init command helps new Go contributors get their development
// environment set up for the Go contribution process.
//
// It aims to be a complement or alternative to https://golang.org/doc/contribute.html.
package main

import (
	"bytes"
	"flag"
	"fmt"
	"io/ioutil"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
)

var repo = flag.String("repo", "go", "Which go repo you want to contribute to. Use \"go\" for the core, or e.g. \"net\" for golang.org/x/net/*")

func main() {
	log.SetFlags(0)
	flag.Parse()

	checkCLA()
	checkGoroot()
	checkWorkingDir()
	checkGitOrigin()
}

func checkCLA() {
	slurp, err := ioutil.ReadFile(cookiesFile())
	if err != nil && !os.IsNotExist(err) {
		log.Fatal(err)
	}
	if bytes.Contains(slurp, []byte("go.googlesource.com")) &&
		bytes.Contains(slurp, []byte("go-review.googlesource.com")) {
		// Probably good.
		return
	}
	log.Fatal("Your .gitcookies file isn't configured.\n" +
		"Next steps:\n" +
		"  * Submit a CLA (https://golang.org/doc/contribute.html#cla) if not done\n" +
		"  * Go to https://go.googlesource.com/ and click \"Generate Password\" at the top,\n" +
		"    then follow instructions.\n" +
		"  * Run go-contrib-init again.\n")
}

func cookiesFile() string {
	if runtime.GOOS == "windows" {
		return filepath.Join(os.Getenv("USERPROFILE"), ".gitcookies")
	}
	return filepath.Join(os.Getenv("HOME"), ".gitcookies")
}

func checkGoroot() {
	v := os.Getenv("GOROOT")
	if v == "" {
		return
	}
	if *repo == "go" {
		if strings.HasPrefix(v, "/usr/") {
			log.Fatalf("Your GOROOT environment variable is set to %q\n"+
				"This is almost certainly not what you want. Either unset\n"+
				"your GOROOT or set it to the path of your development version\n"+
				"of Go.", v)
		}
		slurp, err := ioutil.ReadFile(filepath.Join(v, "VERSION"))
		if err == nil {
			slurp = bytes.TrimSpace(slurp)
			log.Fatalf("Your GOROOT environment variable is set to %q\n"+
				"But that path is to a binary release of Go, with VERSION file %q.\n"+
				"You should hack on Go in a fresh checkout of Go. Fix or unset your GOROOT.\n",
				v, slurp)
		}
	}
}

func checkWorkingDir() {
	// TODO
}

// mostly check that they didn't clone from github
func checkGitOrigin() {
	if _, err := exec.LookPath("git"); err != nil {
		log.Fatalf("You don't appear to have git installed. Do that.")
	}
	if _, err := os.Stat(".git"); err != nil {
		log.Fatalf("You are not currently in a git checkout of https://go.googlesource.com/%s", *repo)
	}
	remotes, err := exec.Command("git", "remote", "-v").Output()
	if err != nil {
		log.Fatalf("Error running git remote -v: %v", cmdErr(err))
	}
	matches := 0
	wantRemote := "https://go.googlesource.com/" + *repo
	for _, line := range strings.Split(string(remotes), "\n") {
		line = strings.TrimSpace(line)
		if !strings.HasPrefix(line, "origin") {
			continue
		}
		if !strings.Contains(line, wantRemote) {
			curRemote := strings.Fields(strings.TrimPrefix(line, "origin"))[0]
			log.Fatalf("Current directory's git was cloned from %q; origin should be %q", curRemote, wantRemote)
		}
		matches++
	}
	if matches == 0 {
		log.Fatalf("git remote -v output didn't contain expected %q. Got:\n%s", wantRemote, remotes)
	}
}

func cmdErr(err error) string {
	if ee, ok := err.(*exec.ExitError); ok && len(ee.Stderr) > 0 {
		return fmt.Sprintf("%s: %s", err, ee.Stderr)
	}
	return fmt.Sprint(err)
}
