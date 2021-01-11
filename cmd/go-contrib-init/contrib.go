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
	"go/build"
	exec "golang.org/x/sys/execabs"
	"io/ioutil"
	"log"
	"os"
	"path/filepath"
	"regexp"
	"runtime"
	"strings"
)

var (
	repo = flag.String("repo", detectrepo(), "Which go repo you want to contribute to. Use \"go\" for the core, or e.g. \"net\" for golang.org/x/net/*")
	dry  = flag.Bool("dry-run", false, "Fail with problems instead of trying to fix things.")
)

func main() {
	log.SetFlags(0)
	flag.Parse()

	checkCLA()
	checkGoroot()
	checkWorkingDir()
	checkGitOrigin()
	checkGitCodeReview()
	fmt.Print("All good. Happy hacking!\n" +
		"Remember to squash your revised commits and preserve the magic Change-Id lines.\n" +
		"Next steps: https://golang.org/doc/contribute.html#commit_changes\n")
}

func detectrepo() string {
	wd, err := os.Getwd()
	if err != nil {
		return "go"
	}

	for _, path := range filepath.SplitList(build.Default.GOPATH) {
		rightdir := filepath.Join(path, "src", "golang.org", "x") + string(os.PathSeparator)
		if strings.HasPrefix(wd, rightdir) {
			tail := wd[len(rightdir):]
			end := strings.Index(tail, string(os.PathSeparator))
			if end > 0 {
				repo := tail[:end]
				return repo
			}
		}
	}

	return "go"
}

var googleSourceRx = regexp.MustCompile(`(?m)^(go|go-review)?\.googlesource.com\b`)

func checkCLA() {
	slurp, err := ioutil.ReadFile(cookiesFile())
	if err != nil && !os.IsNotExist(err) {
		log.Fatal(err)
	}
	if googleSourceRx.Match(slurp) {
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

func expandUser(s string) string {
	env := "HOME"
	if runtime.GOOS == "windows" {
		env = "USERPROFILE"
	} else if runtime.GOOS == "plan9" {
		env = "home"
	}
	home := os.Getenv(env)
	if home == "" {
		return s
	}

	if len(s) >= 2 && s[0] == '~' && os.IsPathSeparator(s[1]) {
		if runtime.GOOS == "windows" {
			s = filepath.ToSlash(filepath.Join(home, s[2:]))
		} else {
			s = filepath.Join(home, s[2:])
		}
	}
	return os.Expand(s, func(env string) string {
		if env == "HOME" {
			return home
		}
		return os.Getenv(env)
	})
}

func cookiesFile() string {
	out, _ := exec.Command("git", "config", "http.cookiefile").Output()
	if s := strings.TrimSpace(string(out)); s != "" {
		if strings.HasPrefix(s, "~") {
			s = expandUser(s)
		}
		return s
	}
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
	wd, err := os.Getwd()
	if err != nil {
		log.Fatal(err)
	}
	if *repo == "go" {
		if inGoPath(wd) {
			log.Fatalf(`You can't work on Go from within your GOPATH. Please checkout Go outside of your GOPATH

Current directory: %s
GOPATH: %s
`, wd, os.Getenv("GOPATH"))
		}
		return
	}

	gopath := firstGoPath()
	if gopath == "" {
		log.Fatal("Your GOPATH is not set, please set it")
	}

	rightdir := filepath.Join(gopath, "src", "golang.org", "x", *repo)
	if !strings.HasPrefix(wd, rightdir) {
		dirExists, err := exists(rightdir)
		if err != nil {
			log.Fatal(err)
		}
		if !dirExists {
			log.Fatalf("The repo you want to work on is currently not on your system.\n"+
				"Run %q to obtain this repo\n"+
				"then go to the directory %q\n",
				"go get -d golang.org/x/"+*repo, rightdir)
		}
		log.Fatalf("Your current directory is:%q\n"+
			"Working on golang/x/%v requires you be in %q\n",
			wd, *repo, rightdir)
	}
}

func firstGoPath() string {
	list := filepath.SplitList(build.Default.GOPATH)
	if len(list) < 1 {
		return ""
	}
	return list[0]
}

func exists(path string) (bool, error) {
	_, err := os.Stat(path)
	if os.IsNotExist(err) {
		return false, nil
	}
	return true, err
}

func inGoPath(wd string) bool {
	if os.Getenv("GOPATH") == "" {
		return false
	}

	for _, path := range filepath.SplitList(os.Getenv("GOPATH")) {
		if strings.HasPrefix(wd, filepath.Join(path, "src")) {
			return true
		}
	}

	return false
}

// mostly check that they didn't clone from github
func checkGitOrigin() {
	if _, err := exec.LookPath("git"); err != nil {
		log.Fatalf("You don't appear to have git installed. Do that.")
	}
	wantRemote := "https://go.googlesource.com/" + *repo
	remotes, err := exec.Command("git", "remote", "-v").Output()
	if err != nil {
		msg := cmdErr(err)
		if strings.Contains(msg, "Not a git repository") {
			log.Fatalf("Your current directory is not in a git checkout of %s", wantRemote)
		}
		log.Fatalf("Error running git remote -v: %v", msg)
	}
	matches := 0
	for _, line := range strings.Split(string(remotes), "\n") {
		line = strings.TrimSpace(line)
		if !strings.HasPrefix(line, "origin") {
			continue
		}
		if !strings.Contains(line, wantRemote) {
			curRemote := strings.Fields(strings.TrimPrefix(line, "origin"))[0]
			// TODO: if not in dryRun mode, just fix it?
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

func checkGitCodeReview() {
	if _, err := exec.LookPath("git-codereview"); err != nil {
		if *dry {
			log.Fatalf("You don't appear to have git-codereview tool. While this is technically optional,\n" +
				"almost all Go contributors use it. Our documentation and this tool assume it is used.\n" +
				"To install it, run:\n\n\t$ go get golang.org/x/review/git-codereview\n\n(Then run go-contrib-init again)")
		}
		err := exec.Command("go", "get", "golang.org/x/review/git-codereview").Run()
		if err != nil {
			log.Fatalf("Error running go get golang.org/x/review/git-codereview: %v", cmdErr(err))
		}
		log.Printf("Installed git-codereview (ran `go get golang.org/x/review/git-codereview`)")
	}
	missing := false
	for _, cmd := range []string{"change", "gofmt", "mail", "pending", "submit", "sync"} {
		v, _ := exec.Command("git", "config", "alias."+cmd).Output()
		if strings.Contains(string(v), "codereview") {
			continue
		}
		if *dry {
			log.Printf("Missing alias. Run:\n\t$ git config alias.%s \"codereview %s\"", cmd, cmd)
			missing = true
		} else {
			err := exec.Command("git", "config", "alias."+cmd, "codereview "+cmd).Run()
			if err != nil {
				log.Fatalf("Error setting alias.%s: %v", cmd, cmdErr(err))
			}
		}
	}
	if missing {
		log.Fatalf("Missing aliases. (While optional, this tool assumes you use them.)")
	}
}
