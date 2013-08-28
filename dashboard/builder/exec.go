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
func run(d time.Duration, envv []string, dir string, argv ...string) error {
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
	return timeout(d, func() error {
		if err := cmd.Wait(); err != nil {
			if _, ok := err.(TimeoutErr); ok {
				cmd.Process.Kill()
			}
			return err
		}
		return nil
	})
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
func runOutput(d time.Duration, envv []string, out io.Writer, dir string, argv ...string) (bool, error) {
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

	if err := timeout(d, func() error {
		if err := cmd.Wait(); err != nil {
			if _, ok := err.(TimeoutErr); ok {
				cmd.Process.Kill()
			}
			return err
		}
		return nil
	}); err != nil {
		return false, err
	}
	return true, nil
}

// timeout runs f and returns its error value, or if the function does not
// complete before the provided duration it returns a timeout error.
func timeout(d time.Duration, f func() error) error {
	errc := make(chan error)
	go func() {
		errc <- f()
	}()
	t := time.NewTimer(d)
	defer t.Stop()
	select {
	case <-t.C:
		return fmt.Errorf("timed out after %v", d)
	case err := <-errc:
		return err
	}
}

type TimeoutErr time.Duration

func (e TimeoutErr) Error() string {
	return fmt.Sprintf("timed out after %v", time.Duration(e))
}
