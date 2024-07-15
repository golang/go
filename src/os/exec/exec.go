// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package exec runs external commands. It wraps os.StartProcess to make it
// easier to remap stdin and stdout, connect I/O with pipes, and do other
// adjustments.
//
// Unlike the "system" library call from C and other languages, the
// os/exec package intentionally does not invoke the system shell and
// does not expand any glob patterns or handle other expansions,
// pipelines, or redirections typically done by shells. The package
// behaves more like C's "exec" family of functions. To expand glob
// patterns, either call the shell directly, taking care to escape any
// dangerous input, or use the [path/filepath] package's Glob function.
// To expand environment variables, use package os's ExpandEnv.
//
// Note that the examples in this package assume a Unix system.
// They may not run on Windows, and they do not run in the Go Playground
// used by golang.org and godoc.org.
//
// # Executables in the current directory
//
// The functions [Command] and [LookPath] look for a program
// in the directories listed in the current path, following the
// conventions of the host operating system.
// Operating systems have for decades included the current
// directory in this search, sometimes implicitly and sometimes
// configured explicitly that way by default.
// Modern practice is that including the current directory
// is usually unexpected and often leads to security problems.
//
// To avoid those security problems, as of Go 1.19, this package will not resolve a program
// using an implicit or explicit path entry relative to the current directory.
// That is, if you run [LookPath]("go"), it will not successfully return
// ./go on Unix nor .\go.exe on Windows, no matter how the path is configured.
// Instead, if the usual path algorithms would result in that answer,
// these functions return an error err satisfying [errors.Is](err, [ErrDot]).
//
// For example, consider these two program snippets:
//
//	path, err := exec.LookPath("prog")
//	if err != nil {
//		log.Fatal(err)
//	}
//	use(path)
//
// and
//
//	cmd := exec.Command("prog")
//	if err := cmd.Run(); err != nil {
//		log.Fatal(err)
//	}
//
// These will not find and run ./prog or .\prog.exe,
// no matter how the current path is configured.
//
// Code that always wants to run a program from the current directory
// can be rewritten to say "./prog" instead of "prog".
//
// Code that insists on including results from relative path entries
// can instead override the error using an errors.Is check:
//
//	path, err := exec.LookPath("prog")
//	if errors.Is(err, exec.ErrDot) {
//		err = nil
//	}
//	if err != nil {
//		log.Fatal(err)
//	}
//	use(path)
//
// and
//
//	cmd := exec.Command("prog")
//	if errors.Is(cmd.Err, exec.ErrDot) {
//		cmd.Err = nil
//	}
//	if err := cmd.Run(); err != nil {
//		log.Fatal(err)
//	}
//
// Setting the environment variable GODEBUG=execerrdot=0
// disables generation of ErrDot entirely, temporarily restoring the pre-Go 1.19
// behavior for programs that are unable to apply more targeted fixes.
// A future version of Go may remove support for this variable.
//
// Before adding such overrides, make sure you understand the
// security implications of doing so.
// See https://go.dev/blog/path-security for more information.
package exec

import (
	"bytes"
	"context"
	"errors"
	"internal/godebug"
	"internal/syscall/execenv"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"syscall"
	"time"
)

// Error is returned by [LookPath] when it fails to classify a file as an
// executable.
type Error struct {
	// Name is the file name for which the error occurred.
	Name string
	// Err is the underlying error.
	Err error
}

func (e *Error) Error() string {
	return "exec: " + strconv.Quote(e.Name) + ": " + e.Err.Error()
}

func (e *Error) Unwrap() error { return e.Err }

// ErrWaitDelay is returned by [Cmd.Wait] if the process exits with a
// successful status code but its output pipes are not closed before the
// command's WaitDelay expires.
var ErrWaitDelay = errors.New("exec: WaitDelay expired before I/O complete")

// wrappedError wraps an error without relying on fmt.Errorf.
type wrappedError struct {
	prefix string
	err    error
}

func (w wrappedError) Error() string {
	return w.prefix + ": " + w.err.Error()
}

func (w wrappedError) Unwrap() error {
	return w.err
}

// Cmd represents an external command being prepared or run.
//
// A Cmd cannot be reused after calling its [Cmd.Run], [Cmd.Output] or [Cmd.CombinedOutput]
// methods.
type Cmd struct {
	// Path is the path of the command to run.
	//
	// This is the only field that must be set to a non-zero
	// value. If Path is relative, it is evaluated relative
	// to Dir.
	Path string

	// Args holds command line arguments, including the command as Args[0].
	// If the Args field is empty or nil, Run uses {Path}.
	//
	// In typical use, both Path and Args are set by calling Command.
	Args []string

	// Env specifies the environment of the process.
	// Each entry is of the form "key=value".
	// If Env is nil, the new process uses the current process's
	// environment.
	// If Env contains duplicate environment keys, only the last
	// value in the slice for each duplicate key is used.
	// As a special case on Windows, SYSTEMROOT is always added if
	// missing and not explicitly set to the empty string.
	Env []string

	// Dir specifies the working directory of the command.
	// If Dir is the empty string, Run runs the command in the
	// calling process's current directory.
	Dir string

	// Stdin specifies the process's standard input.
	//
	// If Stdin is nil, the process reads from the null device (os.DevNull).
	//
	// If Stdin is an *os.File, the process's standard input is connected
	// directly to that file.
	//
	// Otherwise, during the execution of the command a separate
	// goroutine reads from Stdin and delivers that data to the command
	// over a pipe. In this case, Wait does not complete until the goroutine
	// stops copying, either because it has reached the end of Stdin
	// (EOF or a read error), or because writing to the pipe returned an error,
	// or because a nonzero WaitDelay was set and expired.
	Stdin io.Reader

	// Stdout and Stderr specify the process's standard output and error.
	//
	// If either is nil, Run connects the corresponding file descriptor
	// to the null device (os.DevNull).
	//
	// If either is an *os.File, the corresponding output from the process
	// is connected directly to that file.
	//
	// Otherwise, during the execution of the command a separate goroutine
	// reads from the process over a pipe and delivers that data to the
	// corresponding Writer. In this case, Wait does not complete until the
	// goroutine reaches EOF or encounters an error or a nonzero WaitDelay
	// expires.
	//
	// If Stdout and Stderr are the same writer, and have a type that can
	// be compared with ==, at most one goroutine at a time will call Write.
	Stdout io.Writer
	Stderr io.Writer

	// ExtraFiles specifies additional open files to be inherited by the
	// new process. It does not include standard input, standard output, or
	// standard error. If non-nil, entry i becomes file descriptor 3+i.
	//
	// ExtraFiles is not supported on Windows.
	ExtraFiles []*os.File

	// SysProcAttr holds optional, operating system-specific attributes.
	// Run passes it to os.StartProcess as the os.ProcAttr's Sys field.
	SysProcAttr *syscall.SysProcAttr

	// Process is the underlying process, once started.
	Process *os.Process

	// ProcessState contains information about an exited process.
	// If the process was started successfully, Wait or Run will
	// populate its ProcessState when the command completes.
	ProcessState *os.ProcessState

	// ctx is the context passed to CommandContext, if any.
	ctx context.Context

	Err error // LookPath error, if any.

	// If Cancel is non-nil, the command must have been created with
	// CommandContext and Cancel will be called when the command's
	// Context is done. By default, CommandContext sets Cancel to
	// call the Kill method on the command's Process.
	//
	// Typically a custom Cancel will send a signal to the command's
	// Process, but it may instead take other actions to initiate cancellation,
	// such as closing a stdin or stdout pipe or sending a shutdown request on a
	// network socket.
	//
	// If the command exits with a success status after Cancel is
	// called, and Cancel does not return an error equivalent to
	// os.ErrProcessDone, then Wait and similar methods will return a non-nil
	// error: either an error wrapping the one returned by Cancel,
	// or the error from the Context.
	// (If the command exits with a non-success status, or Cancel
	// returns an error that wraps os.ErrProcessDone, Wait and similar methods
	// continue to return the command's usual exit status.)
	//
	// If Cancel is set to nil, nothing will happen immediately when the command's
	// Context is done, but a nonzero WaitDelay will still take effect. That may
	// be useful, for example, to work around deadlocks in commands that do not
	// support shutdown signals but are expected to always finish quickly.
	//
	// Cancel will not be called if Start returns a non-nil error.
	Cancel func() error

	// If WaitDelay is non-zero, it bounds the time spent waiting on two sources
	// of unexpected delay in Wait: a child process that fails to exit after the
	// associated Context is canceled, and a child process that exits but leaves
	// its I/O pipes unclosed.
	//
	// The WaitDelay timer starts when either the associated Context is done or a
	// call to Wait observes that the child process has exited, whichever occurs
	// first. When the delay has elapsed, the command shuts down the child process
	// and/or its I/O pipes.
	//
	// If the child process has failed to exit — perhaps because it ignored or
	// failed to receive a shutdown signal from a Cancel function, or because no
	// Cancel function was set — then it will be terminated using os.Process.Kill.
	//
	// Then, if the I/O pipes communicating with the child process are still open,
	// those pipes are closed in order to unblock any goroutines currently blocked
	// on Read or Write calls.
	//
	// If pipes are closed due to WaitDelay, no Cancel call has occurred,
	// and the command has otherwise exited with a successful status, Wait and
	// similar methods will return ErrWaitDelay instead of nil.
	//
	// If WaitDelay is zero (the default), I/O pipes will be read until EOF,
	// which might not occur until orphaned subprocesses of the command have
	// also closed their descriptors for the pipes.
	WaitDelay time.Duration

	// childIOFiles holds closers for any of the child process's
	// stdin, stdout, and/or stderr files that were opened by the Cmd itself
	// (not supplied by the caller). These should be closed as soon as they
	// are inherited by the child process.
	childIOFiles []io.Closer

	// parentIOPipes holds closers for the parent's end of any pipes
	// connected to the child's stdin, stdout, and/or stderr streams
	// that were opened by the Cmd itself (not supplied by the caller).
	// These should be closed after Wait sees the command and copying
	// goroutines exit, or after WaitDelay has expired.
	parentIOPipes []io.Closer

	// goroutine holds a set of closures to execute to copy data
	// to and/or from the command's I/O pipes.
	goroutine []func() error

	// If goroutineErr is non-nil, it receives the first error from a copying
	// goroutine once all such goroutines have completed.
	// goroutineErr is set to nil once its error has been received.
	goroutineErr <-chan error

	// If ctxResult is non-nil, it receives the result of watchCtx exactly once.
	ctxResult <-chan ctxResult

	// The stack saved when the Command was created, if GODEBUG contains
	// execwait=2. Used for debugging leaks.
	createdByStack []byte

	// For a security release long ago, we created x/sys/execabs,
	// which manipulated the unexported lookPathErr error field
	// in this struct. For Go 1.19 we exported the field as Err error,
	// above, but we have to keep lookPathErr around for use by
	// old programs building against new toolchains.
	// The String and Start methods look for an error in lookPathErr
	// in preference to Err, to preserve the errors that execabs sets.
	//
	// In general we don't guarantee misuse of reflect like this,
	// but the misuse of reflect was by us, the best of various bad
	// options to fix the security problem, and people depend on
	// those old copies of execabs continuing to work.
	// The result is that we have to leave this variable around for the
	// rest of time, a compatibility scar.
	//
	// See https://go.dev/blog/path-security
	// and https://go.dev/issue/43724 for more context.
	lookPathErr error

	// cachedLookExtensions caches the result of calling lookExtensions.
	// It is set when Command is called with an absolute path, letting it do
	// the work of resolving the extension, so Start doesn't need to do it again.
	// This is only used on Windows.
	cachedLookExtensions struct{ in, out string }
}

// A ctxResult reports the result of watching the Context associated with a
// running command (and sending corresponding signals if needed).
type ctxResult struct {
	err error

	// If timer is non-nil, it expires after WaitDelay has elapsed after
	// the Context is done.
	//
	// (If timer is nil, that means that the Context was not done before the
	// command completed, or no WaitDelay was set, or the WaitDelay already
	// expired and its effect was already applied.)
	timer *time.Timer
}

var execwait = godebug.New("#execwait")
var execerrdot = godebug.New("execerrdot")

// Command returns the [Cmd] struct to execute the named program with
// the given arguments.
//
// It sets only the Path and Args in the returned structure.
//
// If name contains no path separators, Command uses [LookPath] to
// resolve name to a complete path if possible. Otherwise it uses name
// directly as Path.
//
// The returned Cmd's Args field is constructed from the command name
// followed by the elements of arg, so arg should not include the
// command name itself. For example, Command("echo", "hello").
// Args[0] is always name, not the possibly resolved Path.
//
// On Windows, processes receive the whole command line as a single string
// and do their own parsing. Command combines and quotes Args into a command
// line string with an algorithm compatible with applications using
// CommandLineToArgvW (which is the most common way). Notable exceptions are
// msiexec.exe and cmd.exe (and thus, all batch files), which have a different
// unquoting algorithm. In these or other similar cases, you can do the
// quoting yourself and provide the full command line in SysProcAttr.CmdLine,
// leaving Args empty.
func Command(name string, arg ...string) *Cmd {
	cmd := &Cmd{
		Path: name,
		Args: append([]string{name}, arg...),
	}

	if v := execwait.Value(); v != "" {
		if v == "2" {
			// Obtain the caller stack. (This is equivalent to runtime/debug.Stack,
			// copied to avoid importing the whole package.)
			stack := make([]byte, 1024)
			for {
				n := runtime.Stack(stack, false)
				if n < len(stack) {
					stack = stack[:n]
					break
				}
				stack = make([]byte, 2*len(stack))
			}

			if i := bytes.Index(stack, []byte("\nos/exec.Command(")); i >= 0 {
				stack = stack[i+1:]
			}
			cmd.createdByStack = stack
		}

		runtime.SetFinalizer(cmd, func(c *Cmd) {
			if c.Process != nil && c.ProcessState == nil {
				debugHint := ""
				if c.createdByStack == nil {
					debugHint = " (set GODEBUG=execwait=2 to capture stacks for debugging)"
				} else {
					os.Stderr.WriteString("GODEBUG=execwait=2 detected a leaked exec.Cmd created by:\n")
					os.Stderr.Write(c.createdByStack)
					os.Stderr.WriteString("\n")
					debugHint = ""
				}
				panic("exec: Cmd started a Process but leaked without a call to Wait" + debugHint)
			}
		})
	}

	if filepath.Base(name) == name {
		lp, err := LookPath(name)
		if lp != "" {
			// Update cmd.Path even if err is non-nil.
			// If err is ErrDot (especially on Windows), lp may include a resolved
			// extension (like .exe or .bat) that should be preserved.
			cmd.Path = lp
		}
		if err != nil {
			cmd.Err = err
		}
	} else if runtime.GOOS == "windows" && filepath.IsAbs(name) {
		// We may need to add a filename extension from PATHEXT
		// or verify an extension that is already present.
		// Since the path is absolute, its extension should be unambiguous
		// and independent of cmd.Dir, and we can go ahead and cache the lookup now.
		//
		// Note that we don't cache anything here for relative paths, because
		// cmd.Dir may be set after we return from this function and that may
		// cause the command to resolve to a different extension.
		if lp, err := lookExtensions(name, ""); err == nil {
			cmd.cachedLookExtensions.in, cmd.cachedLookExtensions.out = name, lp
		} else {
			cmd.Err = err
		}
	}
	return cmd
}

// CommandContext is like [Command] but includes a context.
//
// The provided context is used to interrupt the process
// (by calling cmd.Cancel or [os.Process.Kill])
// if the context becomes done before the command completes on its own.
//
// CommandContext sets the command's Cancel function to invoke the Kill method
// on its Process, and leaves its WaitDelay unset. The caller may change the
// cancellation behavior by modifying those fields before starting the command.
func CommandContext(ctx context.Context, name string, arg ...string) *Cmd {
	if ctx == nil {
		panic("nil Context")
	}
	cmd := Command(name, arg...)
	cmd.ctx = ctx
	cmd.Cancel = func() error {
		return cmd.Process.Kill()
	}
	return cmd
}

// String returns a human-readable description of c.
// It is intended only for debugging.
// In particular, it is not suitable for use as input to a shell.
// The output of String may vary across Go releases.
func (c *Cmd) String() string {
	if c.Err != nil || c.lookPathErr != nil {
		// failed to resolve path; report the original requested path (plus args)
		return strings.Join(c.Args, " ")
	}
	// report the exact executable path (plus args)
	b := new(strings.Builder)
	b.WriteString(c.Path)
	for _, a := range c.Args[1:] {
		b.WriteByte(' ')
		b.WriteString(a)
	}
	return b.String()
}

// interfaceEqual protects against panics from doing equality tests on
// two interfaces with non-comparable underlying types.
func interfaceEqual(a, b any) bool {
	defer func() {
		recover()
	}()
	return a == b
}

func (c *Cmd) argv() []string {
	if len(c.Args) > 0 {
		return c.Args
	}
	return []string{c.Path}
}

func (c *Cmd) childStdin() (*os.File, error) {
	if c.Stdin == nil {
		f, err := os.Open(os.DevNull)
		if err != nil {
			return nil, err
		}
		c.childIOFiles = append(c.childIOFiles, f)
		return f, nil
	}

	if f, ok := c.Stdin.(*os.File); ok {
		return f, nil
	}

	pr, pw, err := os.Pipe()
	if err != nil {
		return nil, err
	}

	c.childIOFiles = append(c.childIOFiles, pr)
	c.parentIOPipes = append(c.parentIOPipes, pw)
	c.goroutine = append(c.goroutine, func() error {
		_, err := io.Copy(pw, c.Stdin)
		if skipStdinCopyError(err) {
			err = nil
		}
		if err1 := pw.Close(); err == nil {
			err = err1
		}
		return err
	})
	return pr, nil
}

func (c *Cmd) childStdout() (*os.File, error) {
	return c.writerDescriptor(c.Stdout)
}

func (c *Cmd) childStderr(childStdout *os.File) (*os.File, error) {
	if c.Stderr != nil && interfaceEqual(c.Stderr, c.Stdout) {
		return childStdout, nil
	}
	return c.writerDescriptor(c.Stderr)
}

// writerDescriptor returns an os.File to which the child process
// can write to send data to w.
//
// If w is nil, writerDescriptor returns a File that writes to os.DevNull.
func (c *Cmd) writerDescriptor(w io.Writer) (*os.File, error) {
	if w == nil {
		f, err := os.OpenFile(os.DevNull, os.O_WRONLY, 0)
		if err != nil {
			return nil, err
		}
		c.childIOFiles = append(c.childIOFiles, f)
		return f, nil
	}

	if f, ok := w.(*os.File); ok {
		return f, nil
	}

	pr, pw, err := os.Pipe()
	if err != nil {
		return nil, err
	}

	c.childIOFiles = append(c.childIOFiles, pw)
	c.parentIOPipes = append(c.parentIOPipes, pr)
	c.goroutine = append(c.goroutine, func() error {
		_, err := io.Copy(w, pr)
		pr.Close() // in case io.Copy stopped due to write error
		return err
	})
	return pw, nil
}

func closeDescriptors(closers []io.Closer) {
	for _, fd := range closers {
		fd.Close()
	}
}

// Run starts the specified command and waits for it to complete.
//
// The returned error is nil if the command runs, has no problems
// copying stdin, stdout, and stderr, and exits with a zero exit
// status.
//
// If the command starts but does not complete successfully, the error is of
// type [*ExitError]. Other error types may be returned for other situations.
//
// If the calling goroutine has locked the operating system thread
// with [runtime.LockOSThread] and modified any inheritable OS-level
// thread state (for example, Linux or Plan 9 name spaces), the new
// process will inherit the caller's thread state.
func (c *Cmd) Run() error {
	if err := c.Start(); err != nil {
		return err
	}
	return c.Wait()
}

// Start starts the specified command but does not wait for it to complete.
//
// If Start returns successfully, the c.Process field will be set.
//
// After a successful call to Start the [Cmd.Wait] method must be called in
// order to release associated system resources.
func (c *Cmd) Start() error {
	// Check for doubled Start calls before we defer failure cleanup. If the prior
	// call to Start succeeded, we don't want to spuriously close its pipes.
	if c.Process != nil {
		return errors.New("exec: already started")
	}

	started := false
	defer func() {
		closeDescriptors(c.childIOFiles)
		c.childIOFiles = nil

		if !started {
			closeDescriptors(c.parentIOPipes)
			c.parentIOPipes = nil
		}
	}()

	if c.Path == "" && c.Err == nil && c.lookPathErr == nil {
		c.Err = errors.New("exec: no command")
	}
	if c.Err != nil || c.lookPathErr != nil {
		if c.lookPathErr != nil {
			return c.lookPathErr
		}
		return c.Err
	}
	lp := c.Path
	if runtime.GOOS == "windows" {
		if c.Path == c.cachedLookExtensions.in {
			// If Command was called with an absolute path, we already resolved
			// its extension and shouldn't need to do so again (provided c.Path
			// wasn't set to another value between the calls to Command and Start).
			lp = c.cachedLookExtensions.out
		} else {
			// If *Cmd was made without using Command at all, or if Command was
			// called with a relative path, we had to wait until now to resolve
			// it in case c.Dir was changed.
			//
			// Unfortunately, we cannot write the result back to c.Path because programs
			// may assume that they can call Start concurrently with reading the path.
			// (It is safe and non-racy to do so on Unix platforms, and users might not
			// test with the race detector on all platforms;
			// see https://go.dev/issue/62596.)
			//
			// So we will pass the fully resolved path to os.StartProcess, but leave
			// c.Path as is: missing a bit of logging information seems less harmful
			// than triggering a surprising data race, and if the user really cares
			// about that bit of logging they can always use LookPath to resolve it.
			var err error
			lp, err = lookExtensions(c.Path, c.Dir)
			if err != nil {
				return err
			}
		}
	}
	if c.Cancel != nil && c.ctx == nil {
		return errors.New("exec: command with a non-nil Cancel was not created with CommandContext")
	}
	if c.ctx != nil {
		select {
		case <-c.ctx.Done():
			return c.ctx.Err()
		default:
		}
	}

	childFiles := make([]*os.File, 0, 3+len(c.ExtraFiles))
	stdin, err := c.childStdin()
	if err != nil {
		return err
	}
	childFiles = append(childFiles, stdin)
	stdout, err := c.childStdout()
	if err != nil {
		return err
	}
	childFiles = append(childFiles, stdout)
	stderr, err := c.childStderr(stdout)
	if err != nil {
		return err
	}
	childFiles = append(childFiles, stderr)
	childFiles = append(childFiles, c.ExtraFiles...)

	env, err := c.environ()
	if err != nil {
		return err
	}

	c.Process, err = os.StartProcess(lp, c.argv(), &os.ProcAttr{
		Dir:   c.Dir,
		Files: childFiles,
		Env:   env,
		Sys:   c.SysProcAttr,
	})
	if err != nil {
		return err
	}
	started = true

	// Don't allocate the goroutineErr channel unless there are goroutines to start.
	if len(c.goroutine) > 0 {
		goroutineErr := make(chan error, 1)
		c.goroutineErr = goroutineErr

		type goroutineStatus struct {
			running  int
			firstErr error
		}
		statusc := make(chan goroutineStatus, 1)
		statusc <- goroutineStatus{running: len(c.goroutine)}
		for _, fn := range c.goroutine {
			go func(fn func() error) {
				err := fn()

				status := <-statusc
				if status.firstErr == nil {
					status.firstErr = err
				}
				status.running--
				if status.running == 0 {
					goroutineErr <- status.firstErr
				} else {
					statusc <- status
				}
			}(fn)
		}
		c.goroutine = nil // Allow the goroutines' closures to be GC'd when they complete.
	}

	// If we have anything to do when the command's Context expires,
	// start a goroutine to watch for cancellation.
	//
	// (Even if the command was created by CommandContext, a helper library may
	// have explicitly set its Cancel field back to nil, indicating that it should
	// be allowed to continue running after cancellation after all.)
	if (c.Cancel != nil || c.WaitDelay != 0) && c.ctx != nil && c.ctx.Done() != nil {
		resultc := make(chan ctxResult)
		c.ctxResult = resultc
		go c.watchCtx(resultc)
	}

	return nil
}

// watchCtx watches c.ctx until it is able to send a result to resultc.
//
// If c.ctx is done before a result can be sent, watchCtx calls c.Cancel,
// and/or kills cmd.Process it after c.WaitDelay has elapsed.
//
// watchCtx manipulates c.goroutineErr, so its result must be received before
// c.awaitGoroutines is called.
func (c *Cmd) watchCtx(resultc chan<- ctxResult) {
	select {
	case resultc <- ctxResult{}:
		return
	case <-c.ctx.Done():
	}

	var err error
	if c.Cancel != nil {
		if interruptErr := c.Cancel(); interruptErr == nil {
			// We appear to have successfully interrupted the command, so any
			// program behavior from this point may be due to ctx even if the
			// command exits with code 0.
			err = c.ctx.Err()
		} else if errors.Is(interruptErr, os.ErrProcessDone) {
			// The process already finished: we just didn't notice it yet.
			// (Perhaps c.Wait hadn't been called, or perhaps it happened to race with
			// c.ctx being canceled.) Don't inject a needless error.
		} else {
			err = wrappedError{
				prefix: "exec: canceling Cmd",
				err:    interruptErr,
			}
		}
	}
	if c.WaitDelay == 0 {
		resultc <- ctxResult{err: err}
		return
	}

	timer := time.NewTimer(c.WaitDelay)
	select {
	case resultc <- ctxResult{err: err, timer: timer}:
		// c.Process.Wait returned and we've handed the timer off to c.Wait.
		// It will take care of goroutine shutdown from here.
		return
	case <-timer.C:
	}

	killed := false
	if killErr := c.Process.Kill(); killErr == nil {
		// We appear to have killed the process. c.Process.Wait should return a
		// non-nil error to c.Wait unless the Kill signal races with a successful
		// exit, and if that does happen we shouldn't report a spurious error,
		// so don't set err to anything here.
		killed = true
	} else if !errors.Is(killErr, os.ErrProcessDone) {
		err = wrappedError{
			prefix: "exec: killing Cmd",
			err:    killErr,
		}
	}

	if c.goroutineErr != nil {
		select {
		case goroutineErr := <-c.goroutineErr:
			// Forward goroutineErr only if we don't have reason to believe it was
			// caused by a call to Cancel or Kill above.
			if err == nil && !killed {
				err = goroutineErr
			}
		default:
			// Close the child process's I/O pipes, in case it abandoned some
			// subprocess that inherited them and is still holding them open
			// (see https://go.dev/issue/23019).
			//
			// We close the goroutine pipes only after we have sent any signals we're
			// going to send to the process (via Signal or Kill above): if we send
			// SIGKILL to the process, we would prefer for it to die of SIGKILL, not
			// SIGPIPE. (However, this may still cause any orphaned subprocesses to
			// terminate with SIGPIPE.)
			closeDescriptors(c.parentIOPipes)
			// Wait for the copying goroutines to finish, but report ErrWaitDelay for
			// the error: any other error here could result from closing the pipes.
			_ = <-c.goroutineErr
			if err == nil {
				err = ErrWaitDelay
			}
		}

		// Since we have already received the only result from c.goroutineErr,
		// set it to nil to prevent awaitGoroutines from blocking on it.
		c.goroutineErr = nil
	}

	resultc <- ctxResult{err: err}
}

// An ExitError reports an unsuccessful exit by a command.
type ExitError struct {
	*os.ProcessState

	// Stderr holds a subset of the standard error output from the
	// Cmd.Output method if standard error was not otherwise being
	// collected.
	//
	// If the error output is long, Stderr may contain only a prefix
	// and suffix of the output, with the middle replaced with
	// text about the number of omitted bytes.
	//
	// Stderr is provided for debugging, for inclusion in error messages.
	// Users with other needs should redirect Cmd.Stderr as needed.
	Stderr []byte
}

func (e *ExitError) Error() string {
	return e.ProcessState.String()
}

// Wait waits for the command to exit and waits for any copying to
// stdin or copying from stdout or stderr to complete.
//
// The command must have been started by [Cmd.Start].
//
// The returned error is nil if the command runs, has no problems
// copying stdin, stdout, and stderr, and exits with a zero exit
// status.
//
// If the command fails to run or doesn't complete successfully, the
// error is of type [*ExitError]. Other error types may be
// returned for I/O problems.
//
// If any of c.Stdin, c.Stdout or c.Stderr are not an [*os.File], Wait also waits
// for the respective I/O loop copying to or from the process to complete.
//
// Wait releases any resources associated with the [Cmd].
func (c *Cmd) Wait() error {
	if c.Process == nil {
		return errors.New("exec: not started")
	}
	if c.ProcessState != nil {
		return errors.New("exec: Wait was already called")
	}

	state, err := c.Process.Wait()
	if err == nil && !state.Success() {
		err = &ExitError{ProcessState: state}
	}
	c.ProcessState = state

	var timer *time.Timer
	if c.ctxResult != nil {
		watch := <-c.ctxResult
		timer = watch.timer
		// If c.Process.Wait returned an error, prefer that.
		// Otherwise, report any error from the watchCtx goroutine,
		// such as a Context cancellation or a WaitDelay overrun.
		if err == nil && watch.err != nil {
			err = watch.err
		}
	}

	if goroutineErr := c.awaitGoroutines(timer); err == nil {
		// Report an error from the copying goroutines only if the program otherwise
		// exited normally on its own. Otherwise, the copying error may be due to the
		// abnormal termination.
		err = goroutineErr
	}
	closeDescriptors(c.parentIOPipes)
	c.parentIOPipes = nil

	return err
}

// awaitGoroutines waits for the results of the goroutines copying data to or
// from the command's I/O pipes.
//
// If c.WaitDelay elapses before the goroutines complete, awaitGoroutines
// forcibly closes their pipes and returns ErrWaitDelay.
//
// If timer is non-nil, it must send to timer.C at the end of c.WaitDelay.
func (c *Cmd) awaitGoroutines(timer *time.Timer) error {
	defer func() {
		if timer != nil {
			timer.Stop()
		}
		c.goroutineErr = nil
	}()

	if c.goroutineErr == nil {
		return nil // No running goroutines to await.
	}

	if timer == nil {
		if c.WaitDelay == 0 {
			return <-c.goroutineErr
		}

		select {
		case err := <-c.goroutineErr:
			// Avoid the overhead of starting a timer.
			return err
		default:
		}

		// No existing timer was started: either there is no Context associated with
		// the command, or c.Process.Wait completed before the Context was done.
		timer = time.NewTimer(c.WaitDelay)
	}

	select {
	case <-timer.C:
		closeDescriptors(c.parentIOPipes)
		// Wait for the copying goroutines to finish, but ignore any error
		// (since it was probably caused by closing the pipes).
		_ = <-c.goroutineErr
		return ErrWaitDelay

	case err := <-c.goroutineErr:
		return err
	}
}

// Output runs the command and returns its standard output.
// Any returned error will usually be of type [*ExitError].
// If c.Stderr was nil, Output populates [ExitError.Stderr].
func (c *Cmd) Output() ([]byte, error) {
	if c.Stdout != nil {
		return nil, errors.New("exec: Stdout already set")
	}
	var stdout bytes.Buffer
	c.Stdout = &stdout

	captureErr := c.Stderr == nil
	if captureErr {
		c.Stderr = &prefixSuffixSaver{N: 32 << 10}
	}

	err := c.Run()
	if err != nil && captureErr {
		if ee, ok := err.(*ExitError); ok {
			ee.Stderr = c.Stderr.(*prefixSuffixSaver).Bytes()
		}
	}
	return stdout.Bytes(), err
}

// CombinedOutput runs the command and returns its combined standard
// output and standard error.
func (c *Cmd) CombinedOutput() ([]byte, error) {
	if c.Stdout != nil {
		return nil, errors.New("exec: Stdout already set")
	}
	if c.Stderr != nil {
		return nil, errors.New("exec: Stderr already set")
	}
	var b bytes.Buffer
	c.Stdout = &b
	c.Stderr = &b
	err := c.Run()
	return b.Bytes(), err
}

// StdinPipe returns a pipe that will be connected to the command's
// standard input when the command starts.
// The pipe will be closed automatically after [Cmd.Wait] sees the command exit.
// A caller need only call Close to force the pipe to close sooner.
// For example, if the command being run will not exit until standard input
// is closed, the caller must close the pipe.
func (c *Cmd) StdinPipe() (io.WriteCloser, error) {
	if c.Stdin != nil {
		return nil, errors.New("exec: Stdin already set")
	}
	if c.Process != nil {
		return nil, errors.New("exec: StdinPipe after process started")
	}
	pr, pw, err := os.Pipe()
	if err != nil {
		return nil, err
	}
	c.Stdin = pr
	c.childIOFiles = append(c.childIOFiles, pr)
	c.parentIOPipes = append(c.parentIOPipes, pw)
	return pw, nil
}

// StdoutPipe returns a pipe that will be connected to the command's
// standard output when the command starts.
//
// [Cmd.Wait] will close the pipe after seeing the command exit, so most callers
// need not close the pipe themselves. It is thus incorrect to call Wait
// before all reads from the pipe have completed.
// For the same reason, it is incorrect to call [Cmd.Run] when using StdoutPipe.
// See the example for idiomatic usage.
func (c *Cmd) StdoutPipe() (io.ReadCloser, error) {
	if c.Stdout != nil {
		return nil, errors.New("exec: Stdout already set")
	}
	if c.Process != nil {
		return nil, errors.New("exec: StdoutPipe after process started")
	}
	pr, pw, err := os.Pipe()
	if err != nil {
		return nil, err
	}
	c.Stdout = pw
	c.childIOFiles = append(c.childIOFiles, pw)
	c.parentIOPipes = append(c.parentIOPipes, pr)
	return pr, nil
}

// StderrPipe returns a pipe that will be connected to the command's
// standard error when the command starts.
//
// [Cmd.Wait] will close the pipe after seeing the command exit, so most callers
// need not close the pipe themselves. It is thus incorrect to call Wait
// before all reads from the pipe have completed.
// For the same reason, it is incorrect to use [Cmd.Run] when using StderrPipe.
// See the StdoutPipe example for idiomatic usage.
func (c *Cmd) StderrPipe() (io.ReadCloser, error) {
	if c.Stderr != nil {
		return nil, errors.New("exec: Stderr already set")
	}
	if c.Process != nil {
		return nil, errors.New("exec: StderrPipe after process started")
	}
	pr, pw, err := os.Pipe()
	if err != nil {
		return nil, err
	}
	c.Stderr = pw
	c.childIOFiles = append(c.childIOFiles, pw)
	c.parentIOPipes = append(c.parentIOPipes, pr)
	return pr, nil
}

// prefixSuffixSaver is an io.Writer which retains the first N bytes
// and the last N bytes written to it. The Bytes() methods reconstructs
// it with a pretty error message.
type prefixSuffixSaver struct {
	N         int // max size of prefix or suffix
	prefix    []byte
	suffix    []byte // ring buffer once len(suffix) == N
	suffixOff int    // offset to write into suffix
	skipped   int64

	// TODO(bradfitz): we could keep one large []byte and use part of it for
	// the prefix, reserve space for the '... Omitting N bytes ...' message,
	// then the ring buffer suffix, and just rearrange the ring buffer
	// suffix when Bytes() is called, but it doesn't seem worth it for
	// now just for error messages. It's only ~64KB anyway.
}

func (w *prefixSuffixSaver) Write(p []byte) (n int, err error) {
	lenp := len(p)
	p = w.fill(&w.prefix, p)

	// Only keep the last w.N bytes of suffix data.
	if overage := len(p) - w.N; overage > 0 {
		p = p[overage:]
		w.skipped += int64(overage)
	}
	p = w.fill(&w.suffix, p)

	// w.suffix is full now if p is non-empty. Overwrite it in a circle.
	for len(p) > 0 { // 0, 1, or 2 iterations.
		n := copy(w.suffix[w.suffixOff:], p)
		p = p[n:]
		w.skipped += int64(n)
		w.suffixOff += n
		if w.suffixOff == w.N {
			w.suffixOff = 0
		}
	}
	return lenp, nil
}

// fill appends up to len(p) bytes of p to *dst, such that *dst does not
// grow larger than w.N. It returns the un-appended suffix of p.
func (w *prefixSuffixSaver) fill(dst *[]byte, p []byte) (pRemain []byte) {
	if remain := w.N - len(*dst); remain > 0 {
		add := min(len(p), remain)
		*dst = append(*dst, p[:add]...)
		p = p[add:]
	}
	return p
}

func (w *prefixSuffixSaver) Bytes() []byte {
	if w.suffix == nil {
		return w.prefix
	}
	if w.skipped == 0 {
		return append(w.prefix, w.suffix...)
	}
	var buf bytes.Buffer
	buf.Grow(len(w.prefix) + len(w.suffix) + 50)
	buf.Write(w.prefix)
	buf.WriteString("\n... omitting ")
	buf.WriteString(strconv.FormatInt(w.skipped, 10))
	buf.WriteString(" bytes ...\n")
	buf.Write(w.suffix[w.suffixOff:])
	buf.Write(w.suffix[:w.suffixOff])
	return buf.Bytes()
}

// environ returns a best-effort copy of the environment in which the command
// would be run as it is currently configured. If an error occurs in computing
// the environment, it is returned alongside the best-effort copy.
func (c *Cmd) environ() ([]string, error) {
	var err error

	env := c.Env
	if env == nil {
		env, err = execenv.Default(c.SysProcAttr)
		if err != nil {
			env = os.Environ()
			// Note that the non-nil err is preserved despite env being overridden.
		}

		if c.Dir != "" {
			switch runtime.GOOS {
			case "windows", "plan9":
				// Windows and Plan 9 do not use the PWD variable, so we don't need to
				// keep it accurate.
			default:
				// On POSIX platforms, PWD represents “an absolute pathname of the
				// current working directory.” Since we are changing the working
				// directory for the command, we should also update PWD to reflect that.
				//
				// Unfortunately, we didn't always do that, so (as proposed in
				// https://go.dev/issue/50599) to avoid unintended collateral damage we
				// only implicitly update PWD when Env is nil. That way, we're much
				// less likely to override an intentional change to the variable.
				if pwd, absErr := filepath.Abs(c.Dir); absErr == nil {
					env = append(env, "PWD="+pwd)
				} else if err == nil {
					err = absErr
				}
			}
		}
	}

	env, dedupErr := dedupEnv(env)
	if err == nil {
		err = dedupErr
	}
	return addCriticalEnv(env), err
}

// Environ returns a copy of the environment in which the command would be run
// as it is currently configured.
func (c *Cmd) Environ() []string {
	//  Intentionally ignore errors: environ returns a best-effort environment no matter what.
	env, _ := c.environ()
	return env
}

// dedupEnv returns a copy of env with any duplicates removed, in favor of
// later values.
// Items not of the normal environment "key=value" form are preserved unchanged.
// Except on Plan 9, items containing NUL characters are removed, and
// an error is returned along with the remaining values.
func dedupEnv(env []string) ([]string, error) {
	return dedupEnvCase(runtime.GOOS == "windows", runtime.GOOS == "plan9", env)
}

// dedupEnvCase is dedupEnv with a case option for testing.
// If caseInsensitive is true, the case of keys is ignored.
// If nulOK is false, items containing NUL characters are allowed.
func dedupEnvCase(caseInsensitive, nulOK bool, env []string) ([]string, error) {
	// Construct the output in reverse order, to preserve the
	// last occurrence of each key.
	var err error
	out := make([]string, 0, len(env))
	saw := make(map[string]bool, len(env))
	for n := len(env); n > 0; n-- {
		kv := env[n-1]

		// Reject NUL in environment variables to prevent security issues (#56284);
		// except on Plan 9, which uses NUL as os.PathListSeparator (#56544).
		if !nulOK && strings.IndexByte(kv, 0) != -1 {
			err = errors.New("exec: environment variable contains NUL")
			continue
		}

		i := strings.Index(kv, "=")
		if i == 0 {
			// We observe in practice keys with a single leading "=" on Windows.
			// TODO(#49886): Should we consume only the first leading "=" as part
			// of the key, or parse through arbitrarily many of them until a non-"="?
			i = strings.Index(kv[1:], "=") + 1
		}
		if i < 0 {
			if kv != "" {
				// The entry is not of the form "key=value" (as it is required to be).
				// Leave it as-is for now.
				// TODO(#52436): should we strip or reject these bogus entries?
				out = append(out, kv)
			}
			continue
		}
		k := kv[:i]
		if caseInsensitive {
			k = strings.ToLower(k)
		}
		if saw[k] {
			continue
		}

		saw[k] = true
		out = append(out, kv)
	}

	// Now reverse the slice to restore the original order.
	for i := 0; i < len(out)/2; i++ {
		j := len(out) - i - 1
		out[i], out[j] = out[j], out[i]
	}

	return out, err
}

// addCriticalEnv adds any critical environment variables that are required
// (or at least almost always required) on the operating system.
// Currently this is only used for Windows.
func addCriticalEnv(env []string) []string {
	if runtime.GOOS != "windows" {
		return env
	}
	for _, kv := range env {
		k, _, ok := strings.Cut(kv, "=")
		if !ok {
			continue
		}
		if strings.EqualFold(k, "SYSTEMROOT") {
			// We already have it.
			return env
		}
	}
	return append(env, "SYSTEMROOT="+os.Getenv("SYSTEMROOT"))
}

// ErrDot indicates that a path lookup resolved to an executable
// in the current directory due to ‘.’ being in the path, either
// implicitly or explicitly. See the package documentation for details.
//
// Note that functions in this package do not return ErrDot directly.
// Code should use errors.Is(err, ErrDot), not err == ErrDot,
// to test whether a returned error err is due to this condition.
var ErrDot = errors.New("cannot run executable found relative to current directory")
