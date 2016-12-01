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
	"runtime"
)

type talksBuilder struct {
}

func (b talksBuilder) Signature(heads map[string]string) string {
	return heads["talks"]
}

const talksToolsRev = "e04df2157ae7263e17159baabadc99fb03fc7514"

func (b talksBuilder) Init(dir, hostport string, heads map[string]string) (*exec.Cmd, error) {
	toolsDir := filepath.Join(dir, "gopath/src/golang.org/x/tools")
	if err := checkout(repoURL+"tools", talksToolsRev, toolsDir); err != nil {
		return nil, err
	}
	talksDir := filepath.Join(dir, "gopath/src/golang.org/x/talks")
	if err := checkout(repoURL+"talks", heads["talks"], talksDir); err != nil {
		return nil, err
	}

	goDir := os.Getenv("GOROOT_BOOTSTRAP")
	if goDir == "" {
		goDir = runtime.GOROOT()
	}
	goBin := filepath.Join(goDir, "bin/go")
	goPath := filepath.Join(dir, "gopath")
	presentPath := "golang.org/x/tools/cmd/present"
	install := exec.Command(goBin, "install", "-tags=appenginevm", presentPath)
	install.Env = []string{"GOROOT=" + goDir, "GOPATH=" + goPath}
	if err := runErr(install); err != nil {
		return nil, err
	}

	talksBin := filepath.Join(goPath, "bin/present")
	presentSrc := filepath.Join(goPath, "src", presentPath)
	present := exec.Command(talksBin, "-http="+hostport, "-base="+presentSrc)
	present.Dir = talksDir
	// TODO(adg): log this somewhere useful
	present.Stdout = os.Stdout
	present.Stderr = os.Stderr
	if err := present.Start(); err != nil {
		return nil, err
	}
	return present, nil
}

var talksMsg = []byte("Talks - The Go Programming Language")

func (b talksBuilder) HealthCheck(hostport string) error {
	body, err := getOK(fmt.Sprintf("http://%v/", hostport))
	if err != nil {
		return err
	}
	if !bytes.Contains(body, talksMsg) {
		return errors.New("couldn't match string")
	}
	return nil
}
