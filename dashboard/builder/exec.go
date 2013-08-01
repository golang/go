// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bytes"
	"fmt"
	"io"
	"log"
	"os"
	"os/exec"
	"time"
)

// run is a simple wrapper for exec.Run/Close
func run(timeout time.Duration, envv []string, dir string, argv ...string) error {
	if *verbose {
		log.Println("run", argv)
	}
	cmd := exec.Command(argv[0], argv[1:]...)
	cmd.Dir = dir
	cmd.Env = envv
	cmd.Stderr = os.Stderr
	if err := cmd.Start(); err != nil {
		return err
	}
	return waitWithTimeout(timeout, cmd)
}

// runLog runs a process and returns the combined stdout/stderr. It returns
// process combined stdout and stderr output, exit status and error. The
// error returned is nil, if process is started successfully, even if exit
// status is not successful.
func runLog(timeout time.Duration, envv []string, dir string, argv ...string) (string, bool, error) {
	var b bytes.Buffer
	ok, err := runOutput(timeout, envv, &b, dir, argv...)
	return b.String(), ok, err
}

// runOutput runs a process and directs any output to the supplied writer.
// It returns exit status and error. The error returned is nil, if process
// is started successfully, even if exit status is not successful.
func runOutput(timeout time.Duration, envv []string, out io.Writer, dir string, argv ...string) (bool, error) {
	if *verbose {
		log.Println("runOutput", argv)
	}

	cmd := exec.Command(argv[0], argv[1:]...)
	cmd.Dir = dir
	cmd.Env = envv
	cmd.Stdout = out
	cmd.Stderr = out

	startErr := cmd.Start()
	if startErr != nil {
		return false, startErr
	}
	if err := waitWithTimeout(timeout, cmd); err != nil {
		return false, err
	}
	return true, nil
}

func waitWithTimeout(timeout time.Duration, cmd *exec.Cmd) error {
	errc := make(chan error, 1)
	go func() {
		errc <- cmd.Wait()
	}()
	var err error
	select {
	case <-time.After(timeout):
		cmd.Process.Kill()
		err = fmt.Errorf("timed out after %v", timeout)
	case err = <-errc:
	}
	return err
}
