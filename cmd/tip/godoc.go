// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by the Apache 2.0
// license that can be found in the LICENSE file.

package main

import (
	"bytes"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
)

type godocBuilder struct {
}

func (b godocBuilder) Signature(heads map[string]string) string {
	return heads["go"] + "-" + heads["tools"]
}

func (b godocBuilder) Init(dir, hostport string, heads map[string]string) (*exec.Cmd, error) {
	goDir := filepath.Join(dir, "go")
	toolsDir := filepath.Join(dir, "gopath/src/golang.org/x/tools")
	if err := checkout(repoURL+"go", heads["go"], goDir); err != nil {
		return nil, err
	}
	if err := checkout(repoURL+"tools", heads["tools"], toolsDir); err != nil {
		return nil, err
	}

	make := exec.Command(filepath.Join(goDir, "src/make.bash"))
	make.Dir = filepath.Join(goDir, "src")
	if err := runErr(make); err != nil {
		return nil, err
	}
	goBin := filepath.Join(goDir, "bin/go")
	goPath := filepath.Join(dir, "gopath")
	install := exec.Command(goBin, "install", "golang.org/x/tools/cmd/godoc")
	install.Env = []string{
		"GOROOT=" + goDir,
		"GOPATH=" + goPath,
		"GOROOT_BOOTSTRAP=" + os.Getenv("GOROOT_BOOTSTRAP"),
	}
	if err := runErr(install); err != nil {
		return nil, err
	}

	godocBin := filepath.Join(goPath, "bin/godoc")
	godoc := exec.Command(godocBin, "-http="+hostport, "-index", "-index_interval=-1s")
	godoc.Env = []string{"GOROOT=" + goDir}
	// TODO(adg): log this somewhere useful
	godoc.Stdout = os.Stdout
	godoc.Stderr = os.Stderr
	if err := godoc.Start(); err != nil {
		return nil, err
	}
	return godoc, nil
}

var indexingMsg = []byte("Indexing in progress: result may be inaccurate")

func (b godocBuilder) HealthCheck(hostport string) error {
	body, err := getOK(fmt.Sprintf("http://%v/search?q=FALLTHROUGH", hostport))
	if err != nil {
		return err
	}
	if bytes.Contains(body, indexingMsg) {
		return errors.New("still indexing")
	}
	return nil
}
