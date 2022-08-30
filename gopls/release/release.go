// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
// Package release checks that the a given version of gopls is ready for
// release. It can also tag and publish the release.
//
// To run:
//
// $ cd $GOPATH/src/golang.org/x/tools/gopls
// $ go run release/release.go -version=<version>
package main

import (
	"flag"
	"fmt"
	"go/types"
	exec "golang.org/x/sys/execabs"
	"io/ioutil"
	"log"
	"os"
	"os/user"
	"path/filepath"
	"strconv"
	"strings"

	"golang.org/x/mod/modfile"
	"golang.org/x/mod/semver"
	"golang.org/x/tools/go/packages"
)

var (
	versionFlag = flag.String("version", "", "version to tag")
	remoteFlag  = flag.String("remote", "", "remote to which to push the tag")
	releaseFlag = flag.Bool("release", false, "release is true if you intend to tag and push a release")
)

func main() {
	flag.Parse()

	if *versionFlag == "" {
		log.Fatalf("must provide -version flag")
	}
	if !semver.IsValid(*versionFlag) {
		log.Fatalf("invalid version %s", *versionFlag)
	}
	if semver.Major(*versionFlag) != "v0" {
		log.Fatalf("expected major version v0, got %s", semver.Major(*versionFlag))
	}
	if semver.Build(*versionFlag) != "" {
		log.Fatalf("unexpected build suffix: %s", *versionFlag)
	}
	if *releaseFlag && *remoteFlag == "" {
		log.Fatalf("must provide -remote flag if releasing")
	}
	user, err := user.Current()
	if err != nil {
		log.Fatal(err)
	}
	// Validate that the user is running the program from the gopls module.
	wd, err := os.Getwd()
	if err != nil {
		log.Fatal(err)
	}
	if filepath.Base(wd) != "gopls" {
		log.Fatalf("must run from the gopls module")
	}
	// Confirm that they are running on a branch with a name following the
	// format of "gopls-release-branch.<major>.<minor>".
	if err := validateBranchName(*versionFlag); err != nil {
		log.Fatal(err)
	}
	// Confirm that they have updated the hardcoded version.
	if err := validateHardcodedVersion(wd, *versionFlag); err != nil {
		log.Fatal(err)
	}
	// Confirm that the versions in the go.mod file are correct.
	if err := validateGoModFile(wd); err != nil {
		log.Fatal(err)
	}
	earlyExitMsg := "Validated that the release is ready. Exiting without tagging and publishing."
	if !*releaseFlag {
		fmt.Println(earlyExitMsg)
		os.Exit(0)
	}
	fmt.Println(`Proceeding to tagging and publishing the release...
Please enter Y if you wish to proceed or anything else if you wish to exit.`)
	// Accept and process user input.
	var input string
	fmt.Scanln(&input)
	switch input {
	case "Y":
		fmt.Println("Proceeding to tagging and publishing the release.")
	default:
		fmt.Println(earlyExitMsg)
		os.Exit(0)
	}
	// To tag the release:
	// $ git -c user.email=username@google.com tag -a -m “<message>” gopls/v<major>.<minor>.<patch>-<pre-release>
	goplsVersion := fmt.Sprintf("gopls/%s", *versionFlag)
	cmd := exec.Command("git", "-c", fmt.Sprintf("user.email=%s@google.com", user.Username), "tag", "-a", "-m", fmt.Sprintf("%q", goplsVersion), goplsVersion)
	if err := cmd.Run(); err != nil {
		log.Fatal(err)
	}
	// Push the tag to the remote:
	// $ git push <remote> gopls/v<major>.<minor>.<patch>-pre.1
	cmd = exec.Command("git", "push", *remoteFlag, goplsVersion)
	if err := cmd.Run(); err != nil {
		log.Fatal(err)
	}
}

// validateBranchName reports whether the user's current branch name is of the
// form "gopls-release-branch.<major>.<minor>". It reports an error if not.
func validateBranchName(version string) error {
	cmd := exec.Command("git", "branch", "--show-current")
	stdout, err := cmd.Output()
	if err != nil {
		return err
	}
	branch := strings.TrimSpace(string(stdout))
	expectedBranch := fmt.Sprintf("gopls-release-branch.%s", strings.TrimPrefix(semver.MajorMinor(version), "v"))
	if branch != expectedBranch {
		return fmt.Errorf("expected release branch %s, got %s", expectedBranch, branch)
	}
	return nil
}

// validateHardcodedVersion reports whether the version hardcoded in the gopls
// binary is equivalent to the version being published. It reports an error if
// not.
func validateHardcodedVersion(wd string, version string) error {
	pkgs, err := packages.Load(&packages.Config{
		Dir: filepath.Dir(wd),
		Mode: packages.NeedName | packages.NeedFiles |
			packages.NeedCompiledGoFiles | packages.NeedImports |
			packages.NeedTypes | packages.NeedTypesSizes,
	}, "golang.org/x/tools/gopls/internal/lsp/debug")
	if err != nil {
		return err
	}
	if len(pkgs) != 1 {
		return fmt.Errorf("expected 1 package, got %v", len(pkgs))
	}
	pkg := pkgs[0]
	obj := pkg.Types.Scope().Lookup("Version")
	c, ok := obj.(*types.Const)
	if !ok {
		return fmt.Errorf("no constant named Version")
	}
	hardcodedVersion, err := strconv.Unquote(c.Val().ExactString())
	if err != nil {
		return err
	}
	if semver.Prerelease(hardcodedVersion) != "" {
		return fmt.Errorf("unexpected pre-release for hardcoded version: %s", hardcodedVersion)
	}
	// Don't worry about pre-release tags and expect that there is no build
	// suffix.
	version = strings.TrimSuffix(version, semver.Prerelease(version))
	if hardcodedVersion != version {
		return fmt.Errorf("expected version to be %s, got %s", *versionFlag, hardcodedVersion)
	}
	return nil
}

func validateGoModFile(wd string) error {
	filename := filepath.Join(wd, "go.mod")
	data, err := ioutil.ReadFile(filename)
	if err != nil {
		return err
	}
	gomod, err := modfile.Parse(filename, data, nil)
	if err != nil {
		return err
	}
	// Confirm that there is no replace directive in the go.mod file.
	if len(gomod.Replace) > 0 {
		return fmt.Errorf("expected no replace directives, got %v", len(gomod.Replace))
	}
	// Confirm that the version of x/tools in the gopls/go.mod file points to
	// the second-to-last commit. (The last commit will be the one to update the
	// go.mod file.)
	cmd := exec.Command("git", "rev-parse", "@~")
	stdout, err := cmd.Output()
	if err != nil {
		return err
	}
	hash := string(stdout)
	// Find the golang.org/x/tools require line and compare the versions.
	var version string
	for _, req := range gomod.Require {
		if req.Mod.Path == "golang.org/x/tools" {
			version = req.Mod.Version
			break
		}
	}
	if version == "" {
		return fmt.Errorf("no require for golang.org/x/tools")
	}
	split := strings.Split(version, "-")
	if len(split) != 3 {
		return fmt.Errorf("unexpected pseudoversion format %s", version)
	}
	last := split[len(split)-1]
	if last == "" {
		return fmt.Errorf("unexpected pseudoversion format %s", version)
	}
	if !strings.HasPrefix(hash, last) {
		return fmt.Errorf("golang.org/x/tools pseudoversion should be at commit %s, instead got %s", hash, last)
	}
	return nil
}
