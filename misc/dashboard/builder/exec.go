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

// runLog runs a process and returns the combined stdout/stderr, 
// as well as writing it to logfile (if specified). It returns
// process combined stdout and stderr output, exit status and error.
// The error returned is nil, if process is started successfully,
// even if exit status is not successful.
func runLog(timeout time.Duration, envv []string, logfile, dir string, argv ...string) (string, int, error) {
	if *verbose {
		log.Println("runLog", argv)
	}

	b := new(bytes.Buffer)
	var w io.Writer = b
	if logfile != "" {
		f, err := os.OpenFile(logfile, os.O_WRONLY|os.O_CREATE|os.O_APPEND, 0666)
		if err != nil {
			return "", 0, err
		}
		defer f.Close()
		w = io.MultiWriter(f, b)
	}

	cmd := exec.Command(argv[0], argv[1:]...)
	cmd.Dir = dir
	cmd.Env = envv
	cmd.Stdout = w
	cmd.Stderr = w

	startErr := cmd.Start()
	if startErr != nil {
		return "", 1, startErr
	}
	exitStatus := 0
	if err := waitWithTimeout(timeout, cmd); err != nil {
		exitStatus = 1 // TODO(bradfitz): this is fake. no callers care, so just return a bool instead.
	}
	return b.String(), exitStatus, nil
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
