// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by the Apache 2.0
// license that can be found in the LICENSE file.

package main

import (
	"bytes"
	"fmt"
	"io/ioutil"
	"log"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"time"
)

type godocBuilder struct {
}

func (b godocBuilder) Signature(heads map[string]string) string {
	return heads["go"] + "-" + heads["tools"]
}

var indexingMsg = []byte("Indexing in progress: result may be inaccurate")

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
	install := exec.Command(goBin, "install", "golang.org/x/tools/cmd/godoc")
	install.Env = []string{
		"GOROOT=" + goDir,
		"GOPATH=" + filepath.Join(dir, "gopath"),
		"GOROOT_BOOTSTRAP=" + os.Getenv("GOROOT_BOOTSTRAP"),
	}
	if err := runErr(install); err != nil {
		return nil, err
	}

	godocBin := filepath.Join(goDir, "bin/godoc")
	godoc := exec.Command(godocBin, "-http="+hostport, "-index", "-index_interval=-1s")
	godoc.Env = []string{"GOROOT=" + goDir}
	// TODO(adg): log this somewhere useful
	godoc.Stdout = os.Stdout
	godoc.Stderr = os.Stderr
	if err := godoc.Start(); err != nil {
		return nil, err
	}
	go func() {
		// TODO(bradfitz): tell the proxy that this side is dead
		if err := godoc.Wait(); err != nil {
			log.Printf("process in %v exited: %v", dir, err)
		}
	}()

	var err error
	deadline := time.Now().Add(startTimeout)
	for time.Now().Before(deadline) {
		time.Sleep(time.Second)
		var res *http.Response
		res, err = http.Get(fmt.Sprintf("http://%v/search?q=FALLTHROUGH", hostport))
		if err != nil {
			continue
		}
		rbody, err := ioutil.ReadAll(res.Body)
		res.Body.Close()
		if err == nil && res.StatusCode == http.StatusOK &&
			!bytes.Contains(rbody, indexingMsg) {
			return godoc, nil
		}
	}
	godoc.Process.Kill()
	return nil, fmt.Errorf("timed out waiting for process in %v at %v (%v)", dir, hostport, err)
}
