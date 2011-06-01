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
func run(envv []string, dir string, argv ...string) os.Error {
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
// as well as writing it to logfile (if specified).
func runLog(envv []string, logfile, dir string, argv ...string) (output string, exitStatus int, err os.Error) {
	if *verbose {
		log.Println("runLog", argv)
	}
	argv = useBash(argv)

	b := new(bytes.Buffer)
	var w io.Writer = b
	if logfile != "" {
		f, err := os.OpenFile(logfile, os.O_WRONLY|os.O_CREATE|os.O_APPEND, 0666)
		if err != nil {
			return
		}
		defer f.Close()
		w = io.MultiWriter(f, b)
	}

	cmd := exec.Command(argv[0], argv[1:]...)
	cmd.Dir = dir
	cmd.Env = envv
	cmd.Stdout = w
	cmd.Stderr = w

	err = cmd.Run()
	output = b.String()
	if err != nil {
		if ws, ok := err.(*os.Waitmsg); ok {
			exitStatus = ws.ExitStatus()
		}
		return
	}
	return
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
