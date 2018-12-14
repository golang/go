// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by the Apache 2.0
// license that can be found in the LICENSE file.

package main

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"log"
	"os"
	"os/exec"
	"path/filepath"
)

type godocBuilder struct{}

func prefix8(s string) string {
	if len(s) < 8 {
		return s
	}
	return s[:8]
}

func (b godocBuilder) Signature(heads map[string]string) string {
	return fmt.Sprintf("go=%v/tools=%v", prefix8(heads["go"]), prefix8(heads["tools"]))
}

func (b godocBuilder) Init(logger *log.Logger, dir, hostport string, heads map[string]string) (*exec.Cmd, error) {

	goDir := filepath.Join(dir, "go")
	toolsDir := filepath.Join(dir, "gopath/src/golang.org/x/tools")
	logger.Printf("checking out go repo ...")
	if err := checkout(repoURL+"go", heads["go"], goDir); err != nil {
		return nil, fmt.Errorf("checkout of go: %v", err)
	}
	logger.Printf("checking out tools repo ...")
	if err := checkout(repoURL+"tools", heads["tools"], toolsDir); err != nil {
		return nil, fmt.Errorf("checkout of tools: %v", err)
	}

	var logWriter io.Writer = toLoggerWriter{logger}

	make := exec.Command(filepath.Join(goDir, "src/make.bash"))
	make.Dir = filepath.Join(goDir, "src")
	make.Stdout = logWriter
	make.Stderr = logWriter
	logger.Printf("running make.bash in %s ...", make.Dir)
	if err := make.Run(); err != nil {
		return nil, fmt.Errorf("running make.bash: %v", err)
	}

	logger.Printf("installing godoc ...")
	goBin := filepath.Join(goDir, "bin/go")
	goPath := filepath.Join(dir, "gopath")
	install := exec.Command(goBin, "install", "golang.org/x/tools/cmd/godoc")
	install.Stdout = logWriter
	install.Stderr = logWriter
	install.Env = append(os.Environ(),
		"GOROOT="+goDir,
		"GOPATH="+goPath,
		"GOROOT_BOOTSTRAP="+os.Getenv("GOROOT_BOOTSTRAP"),
	)
	if err := install.Run(); err != nil {
		return nil, fmt.Errorf("go install golang.org/x/tools/cmd/godoc: %v", err)
	}

	logger.Printf("starting godoc ...")
	godocBin := filepath.Join(goPath, "bin/godoc")
	godoc := exec.Command(godocBin, "-http="+hostport, "-index", "-index_interval=-1s", "-play")
	godoc.Env = append(os.Environ(), "GOROOT="+goDir)
	godoc.Stdout = logWriter
	godoc.Stderr = logWriter
	if err := godoc.Start(); err != nil {
		return nil, fmt.Errorf("starting godoc: %v", err)
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
