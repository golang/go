// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package exec runs external commands. It wraps os.StartProcess to make it
// easier to remap stdin and stdout, connect I/O with pipes, and do other
// adjustments.
package exec

// BUG(r): This package should be made even easier to use or merged into os.

import (
	"os"
	"strconv"
)

// Arguments to Run.
const (
	DevNull = iota
	PassThrough
	Pipe
	MergeWithStdout
)

// A Cmd represents a running command.
// Stdin, Stdout, and Stderr are Files representing pipes
// connected to the running command's standard input, output, and error,
// or else nil, depending on the arguments to Run.
// Process represents the underlying operating system process.
type Cmd struct {
	Stdin   *os.File
	Stdout  *os.File
	Stderr  *os.File
	Process *os.Process
}

// PathError records the name of a binary that was not
// found on the current $PATH.
type PathError struct {
	Name string
}

func (e *PathError) String() string {
	return "command " + strconv.Quote(e.Name) + " not found in $PATH"
}

// Given mode (DevNull, etc), return file for child
// and file to record in Cmd structure.
func modeToFiles(mode, fd int) (*os.File, *os.File, os.Error) {
	switch mode {
	case DevNull:
		rw := os.O_WRONLY
		if fd == 0 {
			rw = os.O_RDONLY
		}
		f, err := os.OpenFile(os.DevNull, rw, 0)
		return f, nil, err
	case PassThrough:
		switch fd {
		case 0:
			return os.Stdin, nil, nil
		case 1:
			return os.Stdout, nil, nil
		case 2:
			return os.Stderr, nil, nil
		}
	case Pipe:
		r, w, err := os.Pipe()
		if err != nil {
			return nil, nil, err
		}
		if fd == 0 {
			return r, w, nil
		}
		return w, r, nil
	}
	return nil, nil, os.EINVAL
}

// Run starts the named binary running with
// arguments argv and environment envv.
// If the dir argument is not empty, the child changes
// into the directory before executing the binary.
// It returns a pointer to a new Cmd representing
// the command or an error.
//
// The arguments stdin, stdout, and stderr
// specify how to handle standard input, output, and error.
// The choices are DevNull (connect to /dev/null),
// PassThrough (connect to the current process's standard stream),
// Pipe (connect to an operating system pipe), and
// MergeWithStdout (only for standard error; use the same
// file descriptor as was used for standard output).
// If an argument is Pipe, then the corresponding field (Stdin, Stdout, Stderr)
// of the returned Cmd is the other end of the pipe.
// Otherwise the field in Cmd is nil.
func Run(name string, argv, envv []string, dir string, stdin, stdout, stderr int) (c *Cmd, err os.Error) {
	c = new(Cmd)
	var fd [3]*os.File

	if fd[0], c.Stdin, err = modeToFiles(stdin, 0); err != nil {
		goto Error
	}
	if fd[1], c.Stdout, err = modeToFiles(stdout, 1); err != nil {
		goto Error
	}
	if stderr == MergeWithStdout {
		fd[2] = fd[1]
	} else if fd[2], c.Stderr, err = modeToFiles(stderr, 2); err != nil {
		goto Error
	}

	// Run command.
	c.Process, err = os.StartProcess(name, argv, &os.ProcAttr{Dir: dir, Files: fd[:], Env: envv})
	if err != nil {
		goto Error
	}
	if fd[0] != os.Stdin {
		fd[0].Close()
	}
	if fd[1] != os.Stdout {
		fd[1].Close()
	}
	if fd[2] != os.Stderr && fd[2] != fd[1] {
		fd[2].Close()
	}
	return c, nil

Error:
	if fd[0] != os.Stdin && fd[0] != nil {
		fd[0].Close()
	}
	if fd[1] != os.Stdout && fd[1] != nil {
		fd[1].Close()
	}
	if fd[2] != os.Stderr && fd[2] != nil && fd[2] != fd[1] {
		fd[2].Close()
	}
	if c.Stdin != nil {
		c.Stdin.Close()
	}
	if c.Stdout != nil {
		c.Stdout.Close()
	}
	if c.Stderr != nil {
		c.Stderr.Close()
	}
	if c.Process != nil {
		c.Process.Release()
	}
	return nil, err
}

// Wait waits for the running command c,
// returning the Waitmsg returned when the process exits.
// The options are passed to the process's Wait method.
// Setting options to 0 waits for c to exit;
// other options cause Wait to return for other
// process events; see package os for details.
func (c *Cmd) Wait(options int) (*os.Waitmsg, os.Error) {
	if c.Process == nil {
		return nil, os.ErrorString("exec: invalid use of Cmd.Wait")
	}
	w, err := c.Process.Wait(options)
	if w != nil && (w.Exited() || w.Signaled()) {
		c.Process.Release()
		c.Process = nil
	}
	return w, err
}

// Close waits for the running command c to exit,
// if it hasn't already, and then closes the non-nil file descriptors
// c.Stdin, c.Stdout, and c.Stderr.
func (c *Cmd) Close() os.Error {
	if c.Process != nil {
		// Loop on interrupt, but
		// ignore other errors -- maybe
		// caller has already waited for pid.
		_, err := c.Wait(0)
		for err == os.EINTR {
			_, err = c.Wait(0)
		}
	}

	// Close the FDs that are still open.
	var err os.Error
	if c.Stdin != nil && c.Stdin.Fd() >= 0 {
		if err1 := c.Stdin.Close(); err1 != nil {
			err = err1
		}
	}
	if c.Stdout != nil && c.Stdout.Fd() >= 0 {
		if err1 := c.Stdout.Close(); err1 != nil && err != nil {
			err = err1
		}
	}
	if c.Stderr != nil && c.Stderr != c.Stdout && c.Stderr.Fd() >= 0 {
		if err1 := c.Stderr.Close(); err1 != nil && err != nil {
			err = err1
		}
	}
	return err
}
