// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"path/filepath"
	"runtime"
	"strings"

	"code.google.com/p/go.tools/go/vcs"
)

// These variables are copied from the gobuilder's environment
// to the envv of its subprocesses.
var extraEnv = []string{
	"GOARM",

	// For Unix derivatives.
	"CC",
	"PATH",
	"TMPDIR",
	"USER",

	// For Plan 9.
	"objtype",
	"cputype",
	"path",
}

// builderEnv represents the environment that a Builder will run tests in.
type builderEnv interface {
	// setup sets up the builder environment and returns the directory to run the buildCmd in.
	setup(repo *Repo, workpath, hash string, envv []string) (string, error)
}

// goEnv represents the builderEnv for the main Go repo.
type goEnv struct {
	goos, goarch string
}

func (b *Builder) envv() []string {
	if runtime.GOOS == "windows" {
		return b.envvWindows()
	}

	var e []string
	if *buildTool == "go" {
		e = []string{
			"GOOS=" + b.goos,
			"GOHOSTOS=" + b.goos,
			"GOARCH=" + b.goarch,
			"GOHOSTARCH=" + b.goarch,
			"GOROOT_FINAL=/usr/local/go",
		}
	}

	for _, k := range extraEnv {
		if s, ok := getenvOk(k); ok {
			e = append(e, k+"="+s)
		}
	}
	return e
}

func (b *Builder) envvWindows() []string {
	var start map[string]string
	if *buildTool == "go" {
		start = map[string]string{
			"GOOS":         b.goos,
			"GOHOSTOS":     b.goos,
			"GOARCH":       b.goarch,
			"GOHOSTARCH":   b.goarch,
			"GOROOT_FINAL": `c:\go`,
			"GOBUILDEXIT":  "1", // exit all.bat with completion status.
		}
	}

	for _, name := range extraEnv {
		if s, ok := getenvOk(name); ok {
			start[name] = s
		}
	}
	skip := map[string]bool{
		"GOBIN":   true,
		"GOPATH":  true,
		"GOROOT":  true,
		"INCLUDE": true,
		"LIB":     true,
	}
	var e []string
	for name, v := range start {
		e = append(e, name+"="+v)
		skip[name] = true
	}
	for _, kv := range os.Environ() {
		s := strings.SplitN(kv, "=", 2)
		name := strings.ToUpper(s[0])
		switch {
		case name == "":
			// variables, like "=C:=C:\", just copy them
			e = append(e, kv)
		case !skip[name]:
			e = append(e, kv)
			skip[name] = true
		}
	}
	return e
}

// setup for a goEnv clones the main go repo to workpath/go at the provided hash
// and returns the path workpath/go/src, the location of all go build scripts.
func (env *goEnv) setup(repo *Repo, workpath, hash string, envv []string) (string, error) {
	goworkpath := filepath.Join(workpath, "go")
	if _, err := repo.Clone(goworkpath, hash); err != nil {
		return "", fmt.Errorf("error cloning repository: %s", err)
	}
	return filepath.Join(goworkpath, "src"), nil
}

// gccgoEnv represents the builderEnv for the gccgo compiler.
type gccgoEnv struct{}

// setup for a gccgoEnv clones the gofrontend repo to workpath/go at the hash
// and clones the latest GCC branch to repo.Path/gcc. The gccgo sources are
// replaced with the updated sources in the gofrontend repo and gcc gets
// gets configured and built in workpath/gcc-objdir. The path to
// workpath/gcc-objdir is returned.
func (env *gccgoEnv) setup(repo *Repo, workpath, hash string, envv []string) (string, error) {
	gofrontendpath := filepath.Join(workpath, "gofrontend")
	gccpath := filepath.Join(repo.Path, "gcc")
	gccgopath := filepath.Join(gccpath, "gcc", "go", "gofrontend")
	gcclibgopath := filepath.Join(gccpath, "libgo")

	// get a handle to SVN vcs.Cmd for pulling down GCC.
	svn := vcs.ByCmd("svn")

	// only pull down gcc if we don't have a local copy.
	if _, err := os.Stat(gccpath); err != nil {
		if err := timeout(*cmdTimeout, func() error {
			// pull down a working copy of GCC.
			return svn.Create(gccpath, *gccPath)
		}); err != nil {
			return "", err
		}
	} else {
		// make sure to remove gccgopath and gcclibgopath before
		// updating the repo to avoid file clobbering.
		if err := os.RemoveAll(gccgopath); err != nil {
			return "", err
		}
		if err := os.RemoveAll(gcclibgopath); err != nil {
			return "", err
		}
	}
	if err := svn.Download(gccpath); err != nil {
		return "", err
	}

	// clone gofrontend repo at specified revision
	if _, err := repo.Clone(gofrontendpath, hash); err != nil {
		return "", err
	}

	// remove gccgopath and gcclibgopath before copying over gofrontend.
	if err := os.RemoveAll(gccgopath); err != nil {
		return "", err
	}
	if err := os.RemoveAll(gcclibgopath); err != nil {
		return "", err
	}

	// copy gofrontend and libgo to appropriate locations
	if err := copyDir(filepath.Join(gofrontendpath, "go"), gccgopath); err != nil {
		return "", fmt.Errorf("Failed to copy gofrontend/go to gcc/go/gofrontend: %s\n", err)
	}
	if err := copyDir(filepath.Join(gofrontendpath, "libgo"), gcclibgopath); err != nil {
		return "", fmt.Errorf("Failed to copy gofrontend/libgo to gcc/libgo: %s\n", err)
	}

	// make objdir to work in
	gccobjdir := filepath.Join(workpath, "gcc-objdir")
	if err := os.Mkdir(gccobjdir, mkdirPerm); err != nil {
		return "", err
	}

	// configure GCC with substituted gofrontend and libgo
	gccConfigCmd := []string{filepath.Join(gccpath, "configure"), "--enable-languages=c,c++,go", "--disable-bootstrap"}
	if _, err := runOutput(*cmdTimeout, envv, ioutil.Discard, gccobjdir, gccConfigCmd...); err != nil {
		return "", fmt.Errorf("Failed to configure GCC: %s", err)
	}

	// build gcc
	if _, err := runOutput(*buildTimeout, envv, ioutil.Discard, gccobjdir, "make"); err != nil {
		return "", fmt.Errorf("Failed to build GCC: %s", err)
	}

	return gccobjdir, nil
}

// copyDir copies the src directory into the dst
func copyDir(src, dst string) error {
	return filepath.Walk(src, func(path string, f os.FileInfo, err error) error {
		dstPath := strings.Replace(path, src, dst, 1)
		if f.IsDir() {
			return os.Mkdir(dstPath, mkdirPerm)
		}

		srcFile, err := os.Open(path)
		if err != nil {
			return err
		}
		defer srcFile.Close()

		dstFile, err := os.Create(dstPath)
		if err != nil {
			return err
		}

		if _, err := io.Copy(dstFile, srcFile); err != nil {
			return err
		}
		return dstFile.Close()
	})
}

func getenvOk(k string) (v string, ok bool) {
	v = os.Getenv(k)
	if v != "" {
		return v, true
	}
	keq := k + "="
	for _, kv := range os.Environ() {
		if kv == keq {
			return "", true
		}
	}
	return "", false
}
