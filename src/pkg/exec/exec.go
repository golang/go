// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package exec runs external commands. It wraps os.StartProcess to make it
// easier to remap stdin and stdout, connect I/O with pipes, and do other
// adjustments.
package exec

import (
	"bytes"
	"io"
	"os"
	"strconv"
	"syscall"
)

// Error records the name of a binary that failed to be be executed
// and the reason it failed.
type Error struct {
	Name  string
	Error os.Error
}

func (e *Error) String() string {
	return "exec: " + strconv.Quote(e.Name) + ": " + e.Error.String()
}

// Cmd represents an external command being prepared or run.
type Cmd struct {
	// Path is the path of the command to run.
	//
	// This is the only field that must be set to a non-zero
	// value.
	Path string

	// Args holds command line arguments, including the command as Args[0].
	// If the Args field is empty or nil, Run uses {Path}.
	// 
	// In typical use, both Path and Args are set by calling Command.
	Args []string

	// Env specifies the environment of the process.
	// If Env is nil, Run uses the current process's environment.
	Env []string

	// Dir specifies the working directory of the command.
	// If Dir is the empty string, Run runs the command in the
	// calling process's current directory.
	Dir string

	// Stdin specifies the process's standard input.
	// If Stdin is nil, the process reads from DevNull.
	Stdin io.Reader

	// Stdout and Stderr specify the process's standard output and error.
	//
	// If either is nil, Run connects the
	// corresponding file descriptor to /dev/null.
	//
	// If Stdout and Stderr are are the same writer, at most one
	// goroutine at a time will call Write.
	Stdout io.Writer
	Stderr io.Writer

	// ExtraFiles specifies additional open files to be inherited by the
	// new process. It does not include standard input, standard output, or
	// standard error. If non-nil, entry i becomes file descriptor 3+i.
	ExtraFiles []*os.File

	// SysProcAttr holds optional, operating system-specific attributes.
	// Run passes it to os.StartProcess as the os.ProcAttr's Sys field.
	SysProcAttr *syscall.SysProcAttr

	// Process is the underlying process, once started.
	Process *os.Process

	err             os.Error // last error (from LookPath, stdin, stdout, stderr)
	finished        bool     // when Wait was called
	childFiles      []*os.File
	closeAfterStart []io.Closer
	closeAfterWait  []io.Closer
	goroutine       []func() os.Error
	errch           chan os.Error // one send per goroutine
}

// Command returns the Cmd struct to execute the named program with
// the given arguments.
//
// It sets Path and Args in the returned structure and zeroes the
// other fields.
//
// If name contains no path separators, Command uses LookPath to
// resolve the path to a complete name if possible. Otherwise it uses
// name directly.
//
// The returned Cmd's Args field is constructed from the command name
// followed by the elements of arg, so arg should not include the
// command name itself. For example, Command("echo", "hello")
func Command(name string, arg ...string) *Cmd {
	aname, err := LookPath(name)
	if err != nil {
		aname = name
	}
	return &Cmd{
		Path: aname,
		Args: append([]string{name}, arg...),
		err:  err,
	}
}

// interfaceEqual protects against panics from doing equality tests on
// two interfaces with non-comparable underlying types
func interfaceEqual(a, b interface{}) bool {
	defer func() {
		recover()
	}()
	return a == b
}

func (c *Cmd) envv() []string {
	if c.Env != nil {
		return c.Env
	}
	return os.Environ()
}

func (c *Cmd) argv() []string {
	if len(c.Args) > 0 {
		return c.Args
	}
	return []string{c.Path}
}

func (c *Cmd) stdin() (f *os.File, err os.Error) {
	if c.Stdin == nil {
		f, err = os.Open(os.DevNull)
		c.closeAfterStart = append(c.closeAfterStart, f)
		return
	}

	if f, ok := c.Stdin.(*os.File); ok {
		return f, nil
	}

	pr, pw, err := os.Pipe()
	if err != nil {
		return
	}

	c.closeAfterStart = append(c.closeAfterStart, pr)
	c.closeAfterWait = append(c.closeAfterWait, pw)
	c.goroutine = append(c.goroutine, func() os.Error {
		_, err := io.Copy(pw, c.Stdin)
		if err1 := pw.Close(); err == nil {
			err = err1
		}
		return err
	})
	return pr, nil
}

func (c *Cmd) stdout() (f *os.File, err os.Error) {
	return c.writerDescriptor(c.Stdout)
}

func (c *Cmd) stderr() (f *os.File, err os.Error) {
	if c.Stderr != nil && interfaceEqual(c.Stderr, c.Stdout) {
		return c.childFiles[1], nil
	}
	return c.writerDescriptor(c.Stderr)
}

func (c *Cmd) writerDescriptor(w io.Writer) (f *os.File, err os.Error) {
	if w == nil {
		f, err = os.OpenFile(os.DevNull, os.O_WRONLY, 0)
		c.closeAfterStart = append(c.closeAfterStart, f)
		return
	}

	if f, ok := w.(*os.File); ok {
		return f, nil
	}

	pr, pw, err := os.Pipe()
	if err != nil {
		return
	}

	c.closeAfterStart = append(c.closeAfterStart, pw)
	c.closeAfterWait = append(c.closeAfterWait, pr)
	c.goroutine = append(c.goroutine, func() os.Error {
		_, err := io.Copy(w, pr)
		return err
	})
	return pw, nil
}

// Run starts the specified command and waits for it to complete.
//
// The returned error is nil if the command runs, has no problems
// copying stdin, stdout, and stderr, and exits with a zero exit
// status.
//
// If the command fails to run or doesn't complete successfully, the
// error is of type *os.Waitmsg. Other error types may be
// returned for I/O problems.
func (c *Cmd) Run() os.Error {
	if err := c.Start(); err != nil {
		return err
	}
	return c.Wait()
}

// Start starts the specified command but does not wait for it to complete.
func (c *Cmd) Start() os.Error {
	if c.err != nil {
		return c.err
	}
	if c.Process != nil {
		return os.NewError("exec: already started")
	}

	type F func(*Cmd) (*os.File, os.Error)
	for _, setupFd := range []F{(*Cmd).stdin, (*Cmd).stdout, (*Cmd).stderr} {
		fd, err := setupFd(c)
		if err != nil {
			return err
		}
		c.childFiles = append(c.childFiles, fd)
	}
	c.childFiles = append(c.childFiles, c.ExtraFiles...)

	var err os.Error
	c.Process, err = os.StartProcess(c.Path, c.argv(), &os.ProcAttr{
		Dir:   c.Dir,
		Files: c.childFiles,
		Env:   c.envv(),
		Sys:   c.SysProcAttr,
	})
	if err != nil {
		return err
	}

	for _, fd := range c.closeAfterStart {
		fd.Close()
	}

	c.errch = make(chan os.Error, len(c.goroutine))
	for _, fn := range c.goroutine {
		go func(fn func() os.Error) {
			c.errch <- fn()
		}(fn)
	}

	return nil
}

// Wait waits for the command to exit.
// It must have been started by Start.
//
// The returned error is nil if the command runs, has no problems
// copying stdin, stdout, and stderr, and exits with a zero exit
// status.
//
// If the command fails to run or doesn't complete successfully, the
// error is of type *os.Waitmsg. Other error types may be
// returned for I/O problems.
func (c *Cmd) Wait() os.Error {
	if c.Process == nil {
		return os.NewError("exec: not started")
	}
	if c.finished {
		return os.NewError("exec: Wait was already called")
	}
	c.finished = true
	msg, err := c.Process.Wait(0)

	var copyError os.Error
	for _ = range c.goroutine {
		if err := <-c.errch; err != nil && copyError == nil {
			copyError = err
		}
	}

	for _, fd := range c.closeAfterWait {
		fd.Close()
	}

	if err != nil {
		return err
	} else if !msg.Exited() || msg.ExitStatus() != 0 {
		return msg
	}

	return copyError
}

// Output runs the command and returns its standard output.
func (c *Cmd) Output() ([]byte, os.Error) {
	if c.Stdout != nil {
		return nil, os.NewError("exec: Stdout already set")
	}
	var b bytes.Buffer
	c.Stdout = &b
	err := c.Run()
	return b.Bytes(), err
}

// CombinedOutput runs the command and returns its combined standard
// output and standard error.
func (c *Cmd) CombinedOutput() ([]byte, os.Error) {
	if c.Stdout != nil {
		return nil, os.NewError("exec: Stdout already set")
	}
	if c.Stderr != nil {
		return nil, os.NewError("exec: Stderr already set")
	}
	var b bytes.Buffer
	c.Stdout = &b
	c.Stderr = &b
	err := c.Run()
	return b.Bytes(), err
}

// StdinPipe returns a pipe that will be connected to the command's
// standard input when the command starts.
func (c *Cmd) StdinPipe() (io.WriteCloser, os.Error) {
	if c.Stdin != nil {
		return nil, os.NewError("exec: Stdin already set")
	}
	if c.Process != nil {
		return nil, os.NewError("exec: StdinPipe after process started")
	}
	pr, pw, err := os.Pipe()
	if err != nil {
		return nil, err
	}
	c.Stdin = pr
	c.closeAfterStart = append(c.closeAfterStart, pr)
	c.closeAfterWait = append(c.closeAfterWait, pw)
	return pw, nil
}

// StdoutPipe returns a pipe that will be connected to the command's
// standard output when the command starts.
// The pipe will be closed automatically after Wait sees the command exit.
func (c *Cmd) StdoutPipe() (io.ReadCloser, os.Error) {
	if c.Stdout != nil {
		return nil, os.NewError("exec: Stdout already set")
	}
	if c.Process != nil {
		return nil, os.NewError("exec: StdoutPipe after process started")
	}
	pr, pw, err := os.Pipe()
	if err != nil {
		return nil, err
	}
	c.Stdout = pw
	c.closeAfterStart = append(c.closeAfterStart, pw)
	c.closeAfterWait = append(c.closeAfterWait, pr)
	return pr, nil
}

// StderrPipe returns a pipe that will be connected to the command's
// standard error when the command starts.
// The pipe will be closed automatically after Wait sees the command exit.
func (c *Cmd) StderrPipe() (io.ReadCloser, os.Error) {
	if c.Stderr != nil {
		return nil, os.NewError("exec: Stderr already set")
	}
	if c.Process != nil {
		return nil, os.NewError("exec: StderrPipe after process started")
	}
	pr, pw, err := os.Pipe()
	if err != nil {
		return nil, err
	}
	c.Stderr = pw
	c.closeAfterStart = append(c.closeAfterStart, pw)
	c.closeAfterWait = append(c.closeAfterWait, pr)
	return pr, nil
}
