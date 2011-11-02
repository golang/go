// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bytes"
	"exec"
	"io"
	"log"
	"os"
	"strings"
)

// run is a simple wrapper for exec.Run/Close
func run(envv []string, dir string, argv ...string) error {
	if *verbose {
		log.Println("run", argv)
	}
	argv = useBash(argv)
	cmd := exec.Command(argv[0], argv[1:]...)
	cmd.Dir = dir
	cmd.Env = envv
	cmd.Stderr = os.Stderr
	return cmd.Run()
}

// runLog runs a process and returns the combined stdout/stderr, 
// as well as writing it to logfile (if specified). It returns
// process combined stdout and stderr output, exit status and error.
// The error returned is nil, if process is started successfully,
// even if exit status is not 0.
func runLog(envv []string, logfile, dir string, argv ...string) (string, int, error) {
	if *verbose {
		log.Println("runLog", argv)
	}
	argv = useBash(argv)

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

	err := cmd.Run()
	if err != nil {
		if ws, ok := err.(*exec.ExitError); ok {
			return b.String(), ws.ExitStatus(), nil
		}
	}
	return b.String(), 0, nil
}

// useBash prefixes a list of args with 'bash' if the first argument
// is a bash script.
func useBash(argv []string) []string {
	// TODO(brainman): choose a more reliable heuristic here.
	if strings.HasSuffix(argv[0], ".bash") {
		argv = append([]string{"bash"}, argv...)
	}
	return argv
}
