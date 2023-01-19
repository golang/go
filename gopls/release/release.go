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
	"io/ioutil"
	"log"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	exec "golang.org/x/sys/execabs"

	"golang.org/x/mod/modfile"
	"golang.org/x/mod/semver"
	"golang.org/x/tools/go/packages"
)

var versionFlag = flag.String("version", "", "version to tag")

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
	// Validate that the user is running the program from the gopls module.
	wd, err := os.Getwd()
	if err != nil {
		log.Fatal(err)
	}
	if filepath.Base(wd) != "gopls" {
		log.Fatalf("must run from the gopls module")
	}
	// Confirm that they have updated the hardcoded version.
	if err := validateHardcodedVersion(*versionFlag); err != nil {
		log.Fatal(err)
	}
	// Confirm that the versions in the go.mod file are correct.
	if err := validateGoModFile(wd); err != nil {
		log.Fatal(err)
	}
	fmt.Println("Validated that the release is ready.")
	os.Exit(0)
}

// validateHardcodedVersion reports whether the version hardcoded in the gopls
// binary is equivalent to the version being published. It reports an error if
// not.
func validateHardcodedVersion(version string) error {
	const debugPkg = "golang.org/x/tools/gopls/internal/lsp/debug"
	pkgs, err := packages.Load(&packages.Config{
		Mode: packages.NeedName | packages.NeedFiles |
			packages.NeedCompiledGoFiles | packages.NeedImports |
			packages.NeedTypes | packages.NeedTypesSizes,
	}, debugPkg)
	if err != nil {
		return err
	}
	if len(pkgs) != 1 {
		return fmt.Errorf("expected 1 package, got %v", len(pkgs))
	}
	pkg := pkgs[0]
	if len(pkg.Errors) > 0 {
		return fmt.Errorf("failed to load %q: first error: %w", debugPkg, pkg.Errors[0])
	}
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

func validateGoModFile(goplsDir string) error {
	filename := filepath.Join(goplsDir, "go.mod")
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
