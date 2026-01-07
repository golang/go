// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Use an external test to avoid os/exec -> net/http -> crypto/x509 -> os/exec
// circular dependency on non-cgo darwin.

package exec_test

import (
	"bufio"
	"bytes"
	"context"
	"errors"
	"flag"
	"fmt"
	"internal/poll"
	"internal/testenv"
	"io"
	"log"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"os/exec/internal/fdtest"
	"os/signal"
	"path/filepath"
	"runtime"
	"runtime/debug"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// haveUnexpectedFDs is set at init time to report whether any file descriptors
// were open at program start.
var haveUnexpectedFDs bool

func init() {
	godebug := os.Getenv("GODEBUG")
	if godebug != "" {
		godebug += ","
	}
	godebug += "execwait=2"
	os.Setenv("GODEBUG", godebug)

	if os.Getenv("GO_EXEC_TEST_PID") != "" {
		return
	}
	if runtime.GOOS == "windows" {
		return
	}
	for fd := uintptr(3); fd <= 100; fd++ {
		if poll.IsPollDescriptor(fd) {
			continue
		}

		if fdtest.Exists(fd) {
			haveUnexpectedFDs = true
			return
		}
	}
}

// TestMain allows the test binary to impersonate many other binaries,
// some of which may manipulate os.Stdin, os.Stdout, and/or os.Stderr
// (and thus cannot run as an ordinary Test function, since the testing
// package monkey-patches those variables before running tests).
func TestMain(m *testing.M) {
	flag.Parse()

	pid := os.Getpid()
	if os.Getenv("GO_EXEC_TEST_PID") == "" {
		os.Setenv("GO_EXEC_TEST_PID", strconv.Itoa(pid))

		if runtime.GOOS == "windows" {
			// Normalize environment so that test behavior is consistent.
			// (The behavior of LookPath varies depending on this variable.)
			//
			// Ideally we would test both with the variable set and with it cleared,
			// but I (bcmills) am not sure that that's feasible: it may already be set
			// in the Windows registry, and I'm not sure if it is possible to remove
			// a registry variable in a program's environment.
			//
			// Per https://learn.microsoft.com/en-us/windows/win32/api/processenv/nf-processenv-needcurrentdirectoryforexepathw#remarks,
			// “the existence of the NoDefaultCurrentDirectoryInExePath environment
			// variable is checked, and not its value.”
			os.Setenv("NoDefaultCurrentDirectoryInExePath", "TRUE")
		}

		code := m.Run()
		if code == 0 && flag.Lookup("test.run").Value.String() == "" && flag.Lookup("test.list").Value.String() == "" {
			for cmd := range helperCommands {
				if _, ok := helperCommandUsed.Load(cmd); !ok {
					fmt.Fprintf(os.Stderr, "helper command unused: %q\n", cmd)
					code = 1
				}
			}
		}

		if !testing.Short() {
			// Run a couple of GC cycles to increase the odds of detecting
			// process leaks using the finalizers installed by GODEBUG=execwait=2.
			runtime.GC()
			runtime.GC()
		}

		os.Exit(code)
	}

	args := flag.Args()
	if len(args) == 0 {
		fmt.Fprintf(os.Stderr, "No command\n")
		os.Exit(2)
	}

	cmd, args := args[0], args[1:]
	f, ok := helperCommands[cmd]
	if !ok {
		fmt.Fprintf(os.Stderr, "Unknown command %q\n", cmd)
		os.Exit(2)
	}
	f(args...)
	os.Exit(0)
}

// registerHelperCommand registers a command that the test process can impersonate.
// A command should be registered in the same source file in which it is used.
// If all tests are run and pass, all registered commands must be used.
// (This prevents stale commands from accreting if tests are removed or
// refactored over time.)
func registerHelperCommand(name string, f func(...string)) {
	if helperCommands[name] != nil {
		panic("duplicate command registered: " + name)
	}
	helperCommands[name] = f
}

// maySkipHelperCommand records that the test that uses the named helper command
// was invoked, but may call Skip on the test before actually calling
// helperCommand.
func maySkipHelperCommand(name string) {
	helperCommandUsed.Store(name, true)
}

// helperCommand returns an exec.Cmd that will run the named helper command.
func helperCommand(t *testing.T, name string, args ...string) *exec.Cmd {
	t.Helper()
	return helperCommandContext(t, nil, name, args...)
}

// helperCommandContext is like helperCommand, but also accepts a Context under
// which to run the command.
func helperCommandContext(t *testing.T, ctx context.Context, name string, args ...string) (cmd *exec.Cmd) {
	helperCommandUsed.LoadOrStore(name, true)

	t.Helper()
	exe := testenv.Executable(t)
	cs := append([]string{name}, args...)
	if ctx != nil {
		cmd = exec.CommandContext(ctx, exe, cs...)
	} else {
		cmd = exec.Command(exe, cs...)
	}
	return cmd
}

var helperCommandUsed sync.Map

var helperCommands = map[string]func(...string){
	"echo":          cmdEcho,
	"echoenv":       cmdEchoEnv,
	"cat":           cmdCat,
	"pipetest":      cmdPipeTest,
	"stdinClose":    cmdStdinClose,
	"exit":          cmdExit,
	"describefiles": cmdDescribeFiles,
	"stderrfail":    cmdStderrFail,
	"yes":           cmdYes,
	"hang":          cmdHang,
}

func cmdEcho(args ...string) {
	iargs := []any{}
	for _, s := range args {
		iargs = append(iargs, s)
	}
	fmt.Println(iargs...)
}

func cmdEchoEnv(args ...string) {
	for _, s := range args {
		fmt.Println(os.Getenv(s))
	}
}

func cmdCat(args ...string) {
	if len(args) == 0 {
		io.Copy(os.Stdout, os.Stdin)
		return
	}
	exit := 0
	for _, fn := range args {
		f, err := os.Open(fn)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error: %v\n", err)
			exit = 2
		} else {
			defer f.Close()
			io.Copy(os.Stdout, f)
		}
	}
	os.Exit(exit)
}

func cmdPipeTest(...string) {
	bufr := bufio.NewReader(os.Stdin)
	for {
		line, _, err := bufr.ReadLine()
		if err == io.EOF {
			break
		} else if err != nil {
			os.Exit(1)
		}
		if bytes.HasPrefix(line, []byte("O:")) {
			os.Stdout.Write(line)
			os.Stdout.Write([]byte{'\n'})
		} else if bytes.HasPrefix(line, []byte("E:")) {
			os.Stderr.Write(line)
			os.Stderr.Write([]byte{'\n'})
		} else {
			os.Exit(1)
		}
	}
}

func cmdStdinClose(...string) {
	b, err := io.ReadAll(os.Stdin)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
	if s := string(b); s != stdinCloseTestString {
		fmt.Fprintf(os.Stderr, "Error: Read %q, want %q", s, stdinCloseTestString)
		os.Exit(1)
	}
}

func cmdExit(args ...string) {
	n, _ := strconv.Atoi(args[0])
	os.Exit(n)
}

func cmdDescribeFiles(args ...string) {
	f := os.NewFile(3, "fd3")
	ln, err := net.FileListener(f)
	if err == nil {
		fmt.Printf("fd3: listener %s\n", ln.Addr())
		ln.Close()
	}
}

func cmdStderrFail(...string) {
	fmt.Fprintf(os.Stderr, "some stderr text\n")
	os.Exit(1)
}

func cmdYes(args ...string) {
	if len(args) == 0 {
		args = []string{"y"}
	}
	s := strings.Join(args, " ") + "\n"
	for {
		_, err := os.Stdout.WriteString(s)
		if err != nil {
			os.Exit(1)
		}
	}
}

func TestEcho(t *testing.T) {
	t.Parallel()

	bs, err := helperCommand(t, "echo", "foo bar", "baz").Output()
	if err != nil {
		t.Errorf("echo: %v", err)
	}
	if g, e := string(bs), "foo bar baz\n"; g != e {
		t.Errorf("echo: want %q, got %q", e, g)
	}
}

func TestCommandRelativeName(t *testing.T) {
	t.Parallel()

	cmd := helperCommand(t, "echo", "foo")

	// Run our own binary as a relative path
	// (e.g. "_test/exec.test") our parent directory.
	base := filepath.Base(os.Args[0]) // "exec.test"
	dir := filepath.Dir(os.Args[0])   // "/tmp/go-buildNNNN/os/exec/_test"
	if dir == "." {
		t.Skip("skipping; running test at root somehow")
	}
	parentDir := filepath.Dir(dir) // "/tmp/go-buildNNNN/os/exec"
	dirBase := filepath.Base(dir)  // "_test"
	if dirBase == "." {
		t.Skipf("skipping; unexpected shallow dir of %q", dir)
	}

	cmd.Path = filepath.Join(dirBase, base)
	cmd.Dir = parentDir

	out, err := cmd.Output()
	if err != nil {
		t.Errorf("echo: %v", err)
	}
	if g, e := string(out), "foo\n"; g != e {
		t.Errorf("echo: want %q, got %q", e, g)
	}
}

func TestCatStdin(t *testing.T) {
	t.Parallel()

	// Cat, testing stdin and stdout.
	input := "Input string\nLine 2"
	p := helperCommand(t, "cat")
	p.Stdin = strings.NewReader(input)
	bs, err := p.Output()
	if err != nil {
		t.Errorf("cat: %v", err)
	}
	s := string(bs)
	if s != input {
		t.Errorf("cat: want %q, got %q", input, s)
	}
}

func TestEchoFileRace(t *testing.T) {
	t.Parallel()

	cmd := helperCommand(t, "echo")
	stdin, err := cmd.StdinPipe()
	if err != nil {
		t.Fatalf("StdinPipe: %v", err)
	}
	if err := cmd.Start(); err != nil {
		t.Fatalf("Start: %v", err)
	}
	wrote := make(chan bool)
	go func() {
		defer close(wrote)
		fmt.Fprint(stdin, "echo\n")
	}()
	if err := cmd.Wait(); err != nil {
		t.Fatalf("Wait: %v", err)
	}
	<-wrote
}

func TestCatGoodAndBadFile(t *testing.T) {
	t.Parallel()

	// Testing combined output and error values.
	bs, err := helperCommand(t, "cat", "/bogus/file.foo", "exec_test.go").CombinedOutput()
	if _, ok := err.(*exec.ExitError); !ok {
		t.Errorf("expected *exec.ExitError from cat combined; got %T: %v", err, err)
	}
	errLine, body, ok := strings.Cut(string(bs), "\n")
	if !ok {
		t.Fatalf("expected two lines from cat; got %q", bs)
	}
	if !strings.HasPrefix(errLine, "Error: open /bogus/file.foo") {
		t.Errorf("expected stderr to complain about file; got %q", errLine)
	}
	if !strings.Contains(body, "func TestCatGoodAndBadFile(t *testing.T)") {
		t.Errorf("expected test code; got %q (len %d)", body, len(body))
	}
}

func TestNoExistExecutable(t *testing.T) {
	t.Parallel()

	// Can't run a non-existent executable
	err := exec.Command("/no-exist-executable").Run()
	if err == nil {
		t.Error("expected error from /no-exist-executable")
	}
}

func TestExitStatus(t *testing.T) {
	t.Parallel()

	// Test that exit values are returned correctly
	cmd := helperCommand(t, "exit", "42")
	err := cmd.Run()
	want := "exit status 42"
	switch runtime.GOOS {
	case "plan9":
		want = fmt.Sprintf("exit status: '%s %d: 42'", filepath.Base(cmd.Path), cmd.ProcessState.Pid())
	}
	if werr, ok := err.(*exec.ExitError); ok {
		if s := werr.Error(); s != want {
			t.Errorf("from exit 42 got exit %q, want %q", s, want)
		}
	} else {
		t.Fatalf("expected *exec.ExitError from exit 42; got %T: %v", err, err)
	}
}

func TestExitCode(t *testing.T) {
	t.Parallel()

	// Test that exit code are returned correctly
	cmd := helperCommand(t, "exit", "42")
	cmd.Run()
	want := 42
	if runtime.GOOS == "plan9" {
		want = 1
	}
	got := cmd.ProcessState.ExitCode()
	if want != got {
		t.Errorf("ExitCode got %d, want %d", got, want)
	}

	cmd = helperCommand(t, "/no-exist-executable")
	cmd.Run()
	want = 2
	if runtime.GOOS == "plan9" {
		want = 1
	}
	got = cmd.ProcessState.ExitCode()
	if want != got {
		t.Errorf("ExitCode got %d, want %d", got, want)
	}

	cmd = helperCommand(t, "exit", "255")
	cmd.Run()
	want = 255
	if runtime.GOOS == "plan9" {
		want = 1
	}
	got = cmd.ProcessState.ExitCode()
	if want != got {
		t.Errorf("ExitCode got %d, want %d", got, want)
	}

	cmd = helperCommand(t, "cat")
	cmd.Run()
	want = 0
	got = cmd.ProcessState.ExitCode()
	if want != got {
		t.Errorf("ExitCode got %d, want %d", got, want)
	}

	// Test when command does not call Run().
	cmd = helperCommand(t, "cat")
	want = -1
	got = cmd.ProcessState.ExitCode()
	if want != got {
		t.Errorf("ExitCode got %d, want %d", got, want)
	}
}

func TestPipes(t *testing.T) {
	t.Parallel()

	check := func(what string, err error) {
		if err != nil {
			t.Fatalf("%s: %v", what, err)
		}
	}
	// Cat, testing stdin and stdout.
	c := helperCommand(t, "pipetest")
	stdin, err := c.StdinPipe()
	check("StdinPipe", err)
	stdout, err := c.StdoutPipe()
	check("StdoutPipe", err)
	stderr, err := c.StderrPipe()
	check("StderrPipe", err)

	outbr := bufio.NewReader(stdout)
	errbr := bufio.NewReader(stderr)
	line := func(what string, br *bufio.Reader) string {
		line, _, err := br.ReadLine()
		if err != nil {
			t.Fatalf("%s: %v", what, err)
		}
		return string(line)
	}

	err = c.Start()
	check("Start", err)

	_, err = stdin.Write([]byte("O:I am output\n"))
	check("first stdin Write", err)
	if g, e := line("first output line", outbr), "O:I am output"; g != e {
		t.Errorf("got %q, want %q", g, e)
	}

	_, err = stdin.Write([]byte("E:I am error\n"))
	check("second stdin Write", err)
	if g, e := line("first error line", errbr), "E:I am error"; g != e {
		t.Errorf("got %q, want %q", g, e)
	}

	_, err = stdin.Write([]byte("O:I am output2\n"))
	check("third stdin Write 3", err)
	if g, e := line("second output line", outbr), "O:I am output2"; g != e {
		t.Errorf("got %q, want %q", g, e)
	}

	stdin.Close()
	err = c.Wait()
	check("Wait", err)
}

const stdinCloseTestString = "Some test string."

// Issue 6270.
func TestStdinClose(t *testing.T) {
	t.Parallel()

	check := func(what string, err error) {
		if err != nil {
			t.Fatalf("%s: %v", what, err)
		}
	}
	cmd := helperCommand(t, "stdinClose")
	stdin, err := cmd.StdinPipe()
	check("StdinPipe", err)
	// Check that we can access methods of the underlying os.File.`
	if _, ok := stdin.(interface {
		Fd() uintptr
	}); !ok {
		t.Error("can't access methods of underlying *os.File")
	}
	check("Start", cmd.Start())

	var wg sync.WaitGroup
	wg.Add(1)
	defer wg.Wait()
	go func() {
		defer wg.Done()

		_, err := io.Copy(stdin, strings.NewReader(stdinCloseTestString))
		check("Copy", err)

		// Before the fix, this next line would race with cmd.Wait.
		if err := stdin.Close(); err != nil && !errors.Is(err, os.ErrClosed) {
			t.Errorf("Close: %v", err)
		}
	}()

	check("Wait", cmd.Wait())
}

// Issue 17647.
// It used to be the case that TestStdinClose, above, would fail when
// run under the race detector. This test is a variant of TestStdinClose
// that also used to fail when run under the race detector.
// This test is run by cmd/dist under the race detector to verify that
// the race detector no longer reports any problems.
func TestStdinCloseRace(t *testing.T) {
	t.Parallel()

	cmd := helperCommand(t, "stdinClose")
	stdin, err := cmd.StdinPipe()
	if err != nil {
		t.Fatalf("StdinPipe: %v", err)
	}
	if err := cmd.Start(); err != nil {
		t.Fatalf("Start: %v", err)

	}

	var wg sync.WaitGroup
	wg.Add(2)
	defer wg.Wait()

	go func() {
		defer wg.Done()
		// We don't check the error return of Kill. It is
		// possible that the process has already exited, in
		// which case Kill will return an error "process
		// already finished". The purpose of this test is to
		// see whether the race detector reports an error; it
		// doesn't matter whether this Kill succeeds or not.
		cmd.Process.Kill()
	}()

	go func() {
		defer wg.Done()
		// Send the wrong string, so that the child fails even
		// if the other goroutine doesn't manage to kill it first.
		// This test is to check that the race detector does not
		// falsely report an error, so it doesn't matter how the
		// child process fails.
		io.Copy(stdin, strings.NewReader("unexpected string"))
		if err := stdin.Close(); err != nil && !errors.Is(err, os.ErrClosed) {
			t.Errorf("stdin.Close: %v", err)
		}
	}()

	if err := cmd.Wait(); err == nil {
		t.Fatalf("Wait: succeeded unexpectedly")
	}
}

// Issue 5071
func TestPipeLookPathLeak(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("we don't currently suppore counting open handles on windows")
	}
	// Not parallel: checks for leaked file descriptors

	openFDs := func() []uintptr {
		var fds []uintptr
		for i := uintptr(0); i < 100; i++ {
			if fdtest.Exists(i) {
				fds = append(fds, i)
			}
		}
		return fds
	}

	old := map[uintptr]bool{}
	for _, fd := range openFDs() {
		old[fd] = true
	}

	for i := 0; i < 6; i++ {
		cmd := exec.Command("something-that-does-not-exist-executable")
		cmd.StdoutPipe()
		cmd.StderrPipe()
		cmd.StdinPipe()
		if err := cmd.Run(); err == nil {
			t.Fatal("unexpected success")
		}
	}

	// Since this test is not running in parallel, we don't expect any new file
	// descriptors to be opened while it runs. However, if there are additional
	// FDs present at the start of the test (for example, opened by libc), those
	// may be closed due to a timeout of some sort. Allow those to go away, but
	// check that no new FDs are added.
	for _, fd := range openFDs() {
		if !old[fd] {
			t.Errorf("leaked file descriptor %v", fd)
		}
	}
}

func TestExtraFiles(t *testing.T) {
	if testing.Short() {
		t.Skipf("skipping test in short mode that would build a helper binary")
	}

	if haveUnexpectedFDs {
		// The point of this test is to make sure that any
		// descriptors we open are marked close-on-exec.
		// If haveUnexpectedFDs is true then there were other
		// descriptors open when we started the test,
		// so those descriptors are clearly not close-on-exec,
		// and they will confuse the test. We could modify
		// the test to expect those descriptors to remain open,
		// but since we don't know where they came from or what
		// they are doing, that seems fragile. For example,
		// perhaps they are from the startup code on this
		// system for some reason. Also, this test is not
		// system-specific; as long as most systems do not skip
		// the test, we will still be testing what we care about.
		t.Skip("skipping test because test was run with FDs open")
	}

	testenv.MustHaveExec(t)
	testenv.MustHaveGoBuild(t)

	// This test runs with cgo disabled. External linking needs cgo, so
	// it doesn't work if external linking is required.
	//
	// N.B. go build below explictly doesn't pass through
	// -asan/-msan/-race, so we don't care about those.
	testenv.MustInternalLink(t, testenv.NoSpecialBuildTypes)

	if runtime.GOOS == "windows" {
		t.Skipf("skipping test on %q", runtime.GOOS)
	}

	// Force network usage, to verify the epoll (or whatever) fd
	// doesn't leak to the child,
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer ln.Close()

	// Make sure duplicated fds don't leak to the child.
	f, err := ln.(*net.TCPListener).File()
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	ln2, err := net.FileListener(f)
	if err != nil {
		t.Fatal(err)
	}
	defer ln2.Close()

	// Force TLS root certs to be loaded (which might involve
	// cgo), to make sure none of that potential C code leaks fds.
	ts := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
	// quiet expected TLS handshake error "remote error: bad certificate"
	ts.Config.ErrorLog = log.New(io.Discard, "", 0)
	ts.StartTLS()
	defer ts.Close()
	_, err = http.Get(ts.URL)
	if err == nil {
		t.Errorf("success trying to fetch %s; want an error", ts.URL)
	}

	tf, err := os.CreateTemp("", "")
	if err != nil {
		t.Fatalf("TempFile: %v", err)
	}
	defer os.Remove(tf.Name())
	defer tf.Close()

	const text = "Hello, fd 3!"
	_, err = tf.Write([]byte(text))
	if err != nil {
		t.Fatalf("Write: %v", err)
	}
	_, err = tf.Seek(0, io.SeekStart)
	if err != nil {
		t.Fatalf("Seek: %v", err)
	}

	tempdir := t.TempDir()
	exe := filepath.Join(tempdir, "read3.exe")

	c := testenv.Command(t, testenv.GoToolPath(t), "build", "-o", exe, "read3.go")
	// Build the test without cgo, so that C library functions don't
	// open descriptors unexpectedly. See issue 25628.
	c.Env = append(os.Environ(), "CGO_ENABLED=0")
	if output, err := c.CombinedOutput(); err != nil {
		t.Logf("go build -o %s read3.go\n%s", exe, output)
		t.Fatalf("go build failed: %v", err)
	}

	// Use a deadline to try to get some output even if the program hangs.
	ctx := context.Background()
	if deadline, ok := t.Deadline(); ok {
		// Leave a 20% grace period to flush output, which may be large on the
		// linux/386 builders because we're running the subprocess under strace.
		deadline = deadline.Add(-time.Until(deadline) / 5)

		var cancel context.CancelFunc
		ctx, cancel = context.WithDeadline(ctx, deadline)
		defer cancel()
	}

	c = exec.CommandContext(ctx, exe)
	var stdout, stderr strings.Builder
	c.Stdout = &stdout
	c.Stderr = &stderr
	c.ExtraFiles = []*os.File{tf}
	if runtime.GOOS == "illumos" {
		// Some facilities in illumos are implemented via access
		// to /proc by libc; such accesses can briefly occupy a
		// low-numbered fd.  If this occurs concurrently with the
		// test that checks for leaked descriptors, the check can
		// become confused and report a spurious leaked descriptor.
		// (See issue #42431 for more detailed analysis.)
		//
		// Attempt to constrain the use of additional threads in the
		// child process to make this test less flaky:
		c.Env = append(os.Environ(), "GOMAXPROCS=1")
	}
	err = c.Run()
	if err != nil {
		t.Fatalf("Run: %v\n--- stdout:\n%s--- stderr:\n%s", err, stdout.String(), stderr.String())
	}
	if stdout.String() != text {
		t.Errorf("got stdout %q, stderr %q; want %q on stdout", stdout.String(), stderr.String(), text)
	}
}

func TestExtraFilesRace(t *testing.T) {
	if runtime.GOOS == "windows" {
		maySkipHelperCommand("describefiles")
		t.Skip("no operating system support; skipping")
	}
	t.Parallel()

	listen := func() net.Listener {
		ln, err := net.Listen("tcp", "127.0.0.1:0")
		if err != nil {
			t.Fatal(err)
		}
		return ln
	}
	listenerFile := func(ln net.Listener) *os.File {
		f, err := ln.(*net.TCPListener).File()
		if err != nil {
			t.Fatal(err)
		}
		return f
	}
	runCommand := func(c *exec.Cmd, out chan<- string) {
		bout, err := c.CombinedOutput()
		if err != nil {
			out <- "ERROR:" + err.Error()
		} else {
			out <- string(bout)
		}
	}

	for i := 0; i < 10; i++ {
		if testing.Short() && i >= 3 {
			break
		}
		la := listen()
		ca := helperCommand(t, "describefiles")
		ca.ExtraFiles = []*os.File{listenerFile(la)}
		lb := listen()
		cb := helperCommand(t, "describefiles")
		cb.ExtraFiles = []*os.File{listenerFile(lb)}
		ares := make(chan string)
		bres := make(chan string)
		go runCommand(ca, ares)
		go runCommand(cb, bres)
		if got, want := <-ares, fmt.Sprintf("fd3: listener %s\n", la.Addr()); got != want {
			t.Errorf("iteration %d, process A got:\n%s\nwant:\n%s\n", i, got, want)
		}
		if got, want := <-bres, fmt.Sprintf("fd3: listener %s\n", lb.Addr()); got != want {
			t.Errorf("iteration %d, process B got:\n%s\nwant:\n%s\n", i, got, want)
		}
		la.Close()
		lb.Close()
		for _, f := range ca.ExtraFiles {
			f.Close()
		}
		for _, f := range cb.ExtraFiles {
			f.Close()
		}
	}
}

type delayedInfiniteReader struct{}

func (delayedInfiniteReader) Read(b []byte) (int, error) {
	time.Sleep(100 * time.Millisecond)
	for i := range b {
		b[i] = 'x'
	}
	return len(b), nil
}

// Issue 9173: ignore stdin pipe writes if the program completes successfully.
func TestIgnorePipeErrorOnSuccess(t *testing.T) {
	t.Parallel()

	testWith := func(r io.Reader) func(*testing.T) {
		return func(t *testing.T) {
			t.Parallel()

			cmd := helperCommand(t, "echo", "foo")
			var out strings.Builder
			cmd.Stdin = r
			cmd.Stdout = &out
			if err := cmd.Run(); err != nil {
				t.Fatal(err)
			}
			if got, want := out.String(), "foo\n"; got != want {
				t.Errorf("output = %q; want %q", got, want)
			}
		}
	}
	t.Run("10MB", testWith(strings.NewReader(strings.Repeat("x", 10<<20))))
	t.Run("Infinite", testWith(delayedInfiniteReader{}))
}

type badWriter struct{}

func (w *badWriter) Write(data []byte) (int, error) {
	return 0, io.ErrUnexpectedEOF
}

func TestClosePipeOnCopyError(t *testing.T) {
	t.Parallel()

	cmd := helperCommand(t, "yes")
	cmd.Stdout = new(badWriter)
	err := cmd.Run()
	if err == nil {
		t.Errorf("yes unexpectedly completed successfully")
	}
}

func TestOutputStderrCapture(t *testing.T) {
	t.Parallel()

	cmd := helperCommand(t, "stderrfail")
	_, err := cmd.Output()
	ee, ok := err.(*exec.ExitError)
	if !ok {
		t.Fatalf("Output error type = %T; want ExitError", err)
	}
	got := string(ee.Stderr)
	want := "some stderr text\n"
	if got != want {
		t.Errorf("ExitError.Stderr = %q; want %q", got, want)
	}
}

func TestContext(t *testing.T) {
	t.Parallel()

	ctx, cancel := context.WithCancel(context.Background())
	c := helperCommandContext(t, ctx, "pipetest")
	stdin, err := c.StdinPipe()
	if err != nil {
		t.Fatal(err)
	}
	stdout, err := c.StdoutPipe()
	if err != nil {
		t.Fatal(err)
	}
	if err := c.Start(); err != nil {
		t.Fatal(err)
	}

	if _, err := stdin.Write([]byte("O:hi\n")); err != nil {
		t.Fatal(err)
	}
	buf := make([]byte, 5)
	n, err := io.ReadFull(stdout, buf)
	if n != len(buf) || err != nil || string(buf) != "O:hi\n" {
		t.Fatalf("ReadFull = %d, %v, %q", n, err, buf[:n])
	}
	go cancel()

	if err := c.Wait(); err == nil {
		t.Fatal("expected Wait failure")
	}
}

func TestContextCancel(t *testing.T) {
	if runtime.GOOS == "netbsd" && runtime.GOARCH == "arm64" {
		maySkipHelperCommand("cat")
		testenv.SkipFlaky(t, 42061)
	}

	// To reduce noise in the final goroutine dump,
	// let other parallel tests complete if possible.
	t.Parallel()

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	c := helperCommandContext(t, ctx, "cat")

	stdin, err := c.StdinPipe()
	if err != nil {
		t.Fatal(err)
	}
	defer stdin.Close()

	if err := c.Start(); err != nil {
		t.Fatal(err)
	}

	// At this point the process is alive. Ensure it by sending data to stdin.
	if _, err := io.WriteString(stdin, "echo"); err != nil {
		t.Fatal(err)
	}

	cancel()

	// Calling cancel should have killed the process, so writes
	// should now fail.  Give the process a little while to die.
	start := time.Now()
	delay := 1 * time.Millisecond
	for {
		if _, err := io.WriteString(stdin, "echo"); err != nil {
			break
		}

		if time.Since(start) > time.Minute {
			// Panic instead of calling t.Fatal so that we get a goroutine dump.
			// We want to know exactly what the os/exec goroutines got stuck on.
			debug.SetTraceback("system")
			panic("canceling context did not stop program")
		}

		// Back off exponentially (up to 1-second sleeps) to give the OS time to
		// terminate the process.
		delay *= 2
		if delay > 1*time.Second {
			delay = 1 * time.Second
		}
		time.Sleep(delay)
	}

	if err := c.Wait(); err == nil {
		t.Error("program unexpectedly exited successfully")
	} else {
		t.Logf("exit status: %v", err)
	}
}

// test that environment variables are de-duped.
func TestDedupEnvEcho(t *testing.T) {
	t.Parallel()

	cmd := helperCommand(t, "echoenv", "FOO")
	cmd.Env = append(cmd.Environ(), "FOO=bad", "FOO=good")
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatal(err)
	}
	if got, want := strings.TrimSpace(string(out)), "good"; got != want {
		t.Errorf("output = %q; want %q", got, want)
	}
}

func TestEnvNULCharacter(t *testing.T) {
	if runtime.GOOS == "plan9" {
		t.Skip("plan9 explicitly allows NUL in the environment")
	}
	cmd := helperCommand(t, "echoenv", "FOO", "BAR")
	cmd.Env = append(cmd.Environ(), "FOO=foo\x00BAR=bar")
	out, err := cmd.CombinedOutput()
	if err == nil {
		t.Errorf("output = %q; want error", string(out))
	}
}

func TestString(t *testing.T) {
	t.Parallel()

	echoPath, err := exec.LookPath("echo")
	if err != nil {
		t.Skip(err)
	}
	tests := [...]struct {
		path string
		args []string
		want string
	}{
		{"echo", nil, echoPath},
		{"echo", []string{"a"}, echoPath + " a"},
		{"echo", []string{"a", "b"}, echoPath + " a b"},
	}
	for _, test := range tests {
		cmd := exec.Command(test.path, test.args...)
		if got := cmd.String(); got != test.want {
			t.Errorf("String(%q, %q) = %q, want %q", test.path, test.args, got, test.want)
		}
	}
}

func TestStringPathNotResolved(t *testing.T) {
	t.Parallel()

	_, err := exec.LookPath("makemeasandwich")
	if err == nil {
		t.Skip("wow, thanks")
	}

	cmd := exec.Command("makemeasandwich", "-lettuce")
	want := "makemeasandwich -lettuce"
	if got := cmd.String(); got != want {
		t.Errorf("String(%q, %q) = %q, want %q", "makemeasandwich", "-lettuce", got, want)
	}
}

func TestNoPath(t *testing.T) {
	err := new(exec.Cmd).Start()
	want := "exec: no command"
	if err == nil || err.Error() != want {
		t.Errorf("new(Cmd).Start() = %v, want %q", err, want)
	}
}

// TestDoubleStartLeavesPipesOpen checks for a regression in which calling
// Start twice, which returns an error on the second call, would spuriously
// close the pipes established in the first call.
func TestDoubleStartLeavesPipesOpen(t *testing.T) {
	t.Parallel()

	cmd := helperCommand(t, "pipetest")
	in, err := cmd.StdinPipe()
	if err != nil {
		t.Fatal(err)
	}
	out, err := cmd.StdoutPipe()
	if err != nil {
		t.Fatal(err)
	}

	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := cmd.Wait(); err != nil {
			t.Error(err)
		}
	})

	if err := cmd.Start(); err == nil || !strings.HasSuffix(err.Error(), "already started") {
		t.Fatalf("second call to Start returned a nil; want an 'already started' error")
	}

	outc := make(chan []byte, 1)
	go func() {
		b, err := io.ReadAll(out)
		if err != nil {
			t.Error(err)
		}
		outc <- b
	}()

	const msg = "O:Hello, pipe!\n"

	_, err = io.WriteString(in, msg)
	if err != nil {
		t.Fatal(err)
	}
	in.Close()

	b := <-outc
	if !bytes.Equal(b, []byte(msg)) {
		t.Fatalf("read %q from stdout pipe; want %q", b, msg)
	}
}

func cmdHang(args ...string) {
	sleep, err := time.ParseDuration(args[0])
	if err != nil {
		panic(err)
	}

	fs := flag.NewFlagSet("hang", flag.ExitOnError)
	exitOnInterrupt := fs.Bool("interrupt", false, "if true, commands should exit 0 on os.Interrupt")
	subsleep := fs.Duration("subsleep", 0, "amount of time for the 'hang' helper to leave an orphaned subprocess sleeping with stderr open")
	probe := fs.Duration("probe", 0, "if nonzero, the 'hang' helper should write to stderr at this interval, and exit nonzero if a write fails")
	read := fs.Bool("read", false, "if true, the 'hang' helper should read stdin to completion before sleeping")
	fs.Parse(args[1:])

	pid := os.Getpid()

	if *subsleep != 0 {
		cmd := exec.Command(testenv.Executable(nil), "hang", subsleep.String(), "-read=true", "-probe="+probe.String())
		cmd.Stdin = os.Stdin
		cmd.Stderr = os.Stderr
		out, err := cmd.StdoutPipe()
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		cmd.Start()

		buf := new(strings.Builder)
		if _, err := io.Copy(buf, out); err != nil {
			fmt.Fprintln(os.Stderr, err)
			cmd.Process.Kill()
			cmd.Wait()
			os.Exit(1)
		}
		fmt.Fprintf(os.Stderr, "%d: started %d: %v\n", pid, cmd.Process.Pid, cmd)
		go cmd.Wait() // Release resources if cmd happens not to outlive this process.
	}

	if *exitOnInterrupt {
		c := make(chan os.Signal, 1)
		signal.Notify(c, os.Interrupt)
		go func() {
			sig := <-c
			fmt.Fprintf(os.Stderr, "%d: received %v\n", pid, sig)
			os.Exit(0)
		}()
	} else {
		signal.Ignore(os.Interrupt)
	}

	// Signal that the process is set up by closing stdout.
	os.Stdout.Close()

	if *read {
		if pipeSignal != nil {
			signal.Ignore(pipeSignal)
		}
		r := bufio.NewReader(os.Stdin)
		for {
			line, err := r.ReadBytes('\n')
			if len(line) > 0 {
				// Ignore write errors: we want to keep reading even if stderr is closed.
				fmt.Fprintf(os.Stderr, "%d: read %s", pid, line)
			}
			if err != nil {
				fmt.Fprintf(os.Stderr, "%d: finished read: %v", pid, err)
				break
			}
		}
	}

	if *probe != 0 {
		ticker := time.NewTicker(*probe)
		go func() {
			for range ticker.C {
				if _, err := fmt.Fprintf(os.Stderr, "%d: ok\n", pid); err != nil {
					os.Exit(1)
				}
			}
		}()
	}

	if sleep != 0 {
		time.Sleep(sleep)
		fmt.Fprintf(os.Stderr, "%d: slept %v\n", pid, sleep)
	}
}

// A tickReader reads an unbounded sequence of timestamps at no more than a
// fixed interval.
type tickReader struct {
	interval time.Duration
	lastTick time.Time
	s        string
}

func newTickReader(interval time.Duration) *tickReader {
	return &tickReader{interval: interval}
}

func (r *tickReader) Read(p []byte) (n int, err error) {
	if len(r.s) == 0 {
		if d := r.interval - time.Since(r.lastTick); d > 0 {
			time.Sleep(d)
		}
		r.lastTick = time.Now()
		r.s = r.lastTick.Format(time.RFC3339Nano + "\n")
	}

	n = copy(p, r.s)
	r.s = r.s[n:]
	return n, nil
}

func startHang(t *testing.T, ctx context.Context, hangTime time.Duration, interrupt os.Signal, waitDelay time.Duration, flags ...string) *exec.Cmd {
	t.Helper()

	args := append([]string{hangTime.String()}, flags...)
	cmd := helperCommandContext(t, ctx, "hang", args...)
	cmd.Stdin = newTickReader(1 * time.Millisecond)
	cmd.Stderr = new(strings.Builder)
	if interrupt == nil {
		cmd.Cancel = nil
	} else {
		cmd.Cancel = func() error {
			return cmd.Process.Signal(interrupt)
		}
	}
	cmd.WaitDelay = waitDelay
	out, err := cmd.StdoutPipe()
	if err != nil {
		t.Fatal(err)
	}

	t.Log(cmd)
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}

	// Wait for cmd to close stdout to signal that its handlers are installed.
	buf := new(strings.Builder)
	if _, err := io.Copy(buf, out); err != nil {
		t.Error(err)
		cmd.Process.Kill()
		cmd.Wait()
		t.FailNow()
	}
	if buf.Len() > 0 {
		t.Logf("stdout %v:\n%s", cmd.Args, buf)
	}

	return cmd
}

func TestWaitInterrupt(t *testing.T) {
	t.Parallel()

	// tooLong is an arbitrary duration that is expected to be much longer than
	// the test runs, but short enough that leaked processes will eventually exit
	// on their own.
	const tooLong = 10 * time.Minute

	// Control case: with no cancellation and no WaitDelay, we should wait for the
	// process to exit.
	t.Run("Wait", func(t *testing.T) {
		t.Parallel()
		cmd := startHang(t, context.Background(), 1*time.Millisecond, os.Kill, 0)
		err := cmd.Wait()
		t.Logf("stderr:\n%s", cmd.Stderr)
		t.Logf("[%d] %v", cmd.Process.Pid, err)

		if err != nil {
			t.Errorf("Wait: %v; want <nil>", err)
		}
		if ps := cmd.ProcessState; !ps.Exited() {
			t.Errorf("cmd did not exit: %v", ps)
		} else if code := ps.ExitCode(); code != 0 {
			t.Errorf("cmd.ProcessState.ExitCode() = %v; want 0", code)
		}
	})

	// With a very long WaitDelay and no Cancel function, we should wait for the
	// process to exit even if the command's Context is canceled.
	t.Run("WaitDelay", func(t *testing.T) {
		if runtime.GOOS == "windows" {
			t.Skipf("skipping: os.Interrupt is not implemented on Windows")
		}
		t.Parallel()

		ctx, cancel := context.WithCancel(context.Background())
		cmd := startHang(t, ctx, tooLong, nil, tooLong, "-interrupt=true")
		cancel()

		time.Sleep(1 * time.Millisecond)
		// At this point cmd should still be running (because we passed nil to
		// startHang for the cancel signal). Sending it an explicit Interrupt signal
		// should succeed.
		if err := cmd.Process.Signal(os.Interrupt); err != nil {
			t.Error(err)
		}

		err := cmd.Wait()
		t.Logf("stderr:\n%s", cmd.Stderr)
		t.Logf("[%d] %v", cmd.Process.Pid, err)

		// This program exits with status 0,
		// but pretty much always does so during the wait delay.
		// Since the Cmd itself didn't do anything to stop the process when the
		// context expired, a successful exit is valid (even if late) and does
		// not merit a non-nil error.
		if err != nil {
			t.Errorf("Wait: %v; want nil", err)
		}
		if ps := cmd.ProcessState; !ps.Exited() {
			t.Errorf("cmd did not exit: %v", ps)
		} else if code := ps.ExitCode(); code != 0 {
			t.Errorf("cmd.ProcessState.ExitCode() = %v; want 0", code)
		}
	})

	// If the context is canceled and the Cancel function sends os.Kill,
	// the process should be terminated immediately, and its output
	// pipes should be closed (causing Wait to return) after WaitDelay
	// even if a child process is still writing to them.
	t.Run("SIGKILL-hang", func(t *testing.T) {
		t.Parallel()

		ctx, cancel := context.WithCancel(context.Background())
		cmd := startHang(t, ctx, tooLong, os.Kill, 10*time.Millisecond, "-subsleep=10m", "-probe=1ms")
		cancel()
		err := cmd.Wait()
		t.Logf("stderr:\n%s", cmd.Stderr)
		t.Logf("[%d] %v", cmd.Process.Pid, err)

		// This test should kill the child process after 10ms,
		// leaving a grandchild process writing probes in a loop.
		// The child process should be reported as failed,
		// and the grandchild will exit (or die by SIGPIPE) once the
		// stderr pipe is closed.
		if ee, ok := errors.AsType[*exec.ExitError](err); !ok {
			t.Errorf("Wait error = %v; want %T", err, ee)
		}
	})

	// If the process exits with status 0 but leaves a child behind writing
	// to its output pipes, Wait should only wait for WaitDelay before
	// closing the pipes and returning.  Wait should return ErrWaitDelay
	// to indicate that the piped output may be incomplete even though the
	// command returned a “success” code.
	t.Run("Exit-hang", func(t *testing.T) {
		t.Parallel()

		cmd := startHang(t, context.Background(), 1*time.Millisecond, nil, 10*time.Millisecond, "-subsleep=10m", "-probe=1ms")
		err := cmd.Wait()
		t.Logf("stderr:\n%s", cmd.Stderr)
		t.Logf("[%d] %v", cmd.Process.Pid, err)

		// This child process should exit immediately,
		// leaving a grandchild process writing probes in a loop.
		// Since the child has no ExitError to report but we did not
		// read all of its output, Wait should return ErrWaitDelay.
		if !errors.Is(err, exec.ErrWaitDelay) {
			t.Errorf("Wait error = %v; want %T", err, exec.ErrWaitDelay)
		}
	})

	// If the Cancel function sends a signal that the process can handle, and it
	// handles that signal without actually exiting, then it should be terminated
	// after the WaitDelay.
	t.Run("SIGINT-ignored", func(t *testing.T) {
		if runtime.GOOS == "windows" {
			t.Skipf("skipping: os.Interrupt is not implemented on Windows")
		}
		t.Parallel()

		ctx, cancel := context.WithCancel(context.Background())
		cmd := startHang(t, ctx, tooLong, os.Interrupt, 10*time.Millisecond, "-interrupt=false")
		cancel()
		err := cmd.Wait()
		t.Logf("stderr:\n%s", cmd.Stderr)
		t.Logf("[%d] %v", cmd.Process.Pid, err)

		// This command ignores SIGINT, sleeping until it is killed.
		// Wait should return the usual error for a killed process.
		if ee, ok := errors.AsType[*exec.ExitError](err); !ok {
			t.Errorf("Wait error = %v; want %T", err, ee)
		}
	})

	// If the process handles the cancellation signal and exits with status 0,
	// Wait should report a non-nil error (because the process had to be
	// interrupted), and it should be a context error (because there is no error
	// to report from the child process itself).
	t.Run("SIGINT-handled", func(t *testing.T) {
		if runtime.GOOS == "windows" {
			t.Skipf("skipping: os.Interrupt is not implemented on Windows")
		}
		t.Parallel()

		ctx, cancel := context.WithCancel(context.Background())
		cmd := startHang(t, ctx, tooLong, os.Interrupt, 0, "-interrupt=true")
		cancel()
		err := cmd.Wait()
		t.Logf("stderr:\n%s", cmd.Stderr)
		t.Logf("[%d] %v", cmd.Process.Pid, err)

		if !errors.Is(err, ctx.Err()) {
			t.Errorf("Wait error = %v; want %v", err, ctx.Err())
		}
		if ps := cmd.ProcessState; !ps.Exited() {
			t.Errorf("cmd did not exit: %v", ps)
		} else if code := ps.ExitCode(); code != 0 {
			t.Errorf("cmd.ProcessState.ExitCode() = %v; want 0", code)
		}
	})

	// If the Cancel function sends SIGQUIT, it should be handled in the usual
	// way: a Go program should dump its goroutines and exit with non-success
	// status. (We expect SIGQUIT to be a common pattern in real-world use.)
	t.Run("SIGQUIT", func(t *testing.T) {
		if quitSignal == nil {
			t.Skipf("skipping: SIGQUIT is not supported on %v", runtime.GOOS)
		}
		t.Parallel()

		ctx, cancel := context.WithCancel(context.Background())
		cmd := startHang(t, ctx, tooLong, quitSignal, 0)
		cancel()
		err := cmd.Wait()
		t.Logf("stderr:\n%s", cmd.Stderr)
		t.Logf("[%d] %v", cmd.Process.Pid, err)

		if _, ok := errors.AsType[*exec.ExitError](err); !ok {
			t.Errorf("Wait error = %v; want %v", err, ctx.Err())
		}

		if ps := cmd.ProcessState; !ps.Exited() {
			t.Errorf("cmd did not exit: %v", ps)
		} else if code := ps.ExitCode(); code != 2 {
			// The default os/signal handler exits with code 2.
			t.Errorf("cmd.ProcessState.ExitCode() = %v; want 2", code)
		}

		if !strings.Contains(fmt.Sprint(cmd.Stderr), "\n\ngoroutine ") {
			t.Errorf("cmd.Stderr does not contain a goroutine dump")
		}
	})
}

func TestCancelErrors(t *testing.T) {
	t.Parallel()

	// If Cancel returns a non-ErrProcessDone error and the process
	// exits successfully, Wait should wrap the error from Cancel.
	t.Run("success after error", func(t *testing.T) {
		t.Parallel()

		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()

		cmd := helperCommandContext(t, ctx, "pipetest")
		stdin, err := cmd.StdinPipe()
		if err != nil {
			t.Fatal(err)
		}

		errArbitrary := errors.New("arbitrary error")
		cmd.Cancel = func() error {
			stdin.Close()
			t.Logf("Cancel returning %v", errArbitrary)
			return errArbitrary
		}
		if err := cmd.Start(); err != nil {
			t.Fatal(err)
		}
		cancel()

		err = cmd.Wait()
		t.Logf("[%d] %v", cmd.Process.Pid, err)
		if !errors.Is(err, errArbitrary) || err == errArbitrary {
			t.Errorf("Wait error = %v; want an error wrapping %v", err, errArbitrary)
		}
	})

	// If Cancel returns an error equivalent to ErrProcessDone,
	// Wait should ignore that error. (ErrProcessDone indicates that the
	// process was already done before we tried to interrupt it — maybe we
	// just didn't notice because Wait hadn't been called yet.)
	t.Run("success after ErrProcessDone", func(t *testing.T) {
		t.Parallel()

		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()

		cmd := helperCommandContext(t, ctx, "pipetest")
		stdin, err := cmd.StdinPipe()
		if err != nil {
			t.Fatal(err)
		}

		stdout, err := cmd.StdoutPipe()
		if err != nil {
			t.Fatal(err)
		}

		// We intentionally race Cancel against the process exiting,
		// but ensure that the process wins the race (and return ErrProcessDone
		// from Cancel to report that).
		interruptCalled := make(chan struct{})
		done := make(chan struct{})
		cmd.Cancel = func() error {
			close(interruptCalled)
			<-done
			t.Logf("Cancel returning an error wrapping ErrProcessDone")
			return fmt.Errorf("%w: stdout closed", os.ErrProcessDone)
		}

		if err := cmd.Start(); err != nil {
			t.Fatal(err)
		}

		cancel()
		<-interruptCalled
		stdin.Close()
		io.Copy(io.Discard, stdout) // reaches EOF when the process exits
		close(done)

		err = cmd.Wait()
		t.Logf("[%d] %v", cmd.Process.Pid, err)
		if err != nil {
			t.Errorf("Wait error = %v; want nil", err)
		}
	})

	// If Cancel returns an error and the process is killed after
	// WaitDelay, Wait should report the usual SIGKILL ExitError, not the
	// error from Cancel.
	t.Run("killed after error", func(t *testing.T) {
		t.Parallel()

		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()

		cmd := helperCommandContext(t, ctx, "pipetest")
		stdin, err := cmd.StdinPipe()
		if err != nil {
			t.Fatal(err)
		}
		defer stdin.Close()

		errArbitrary := errors.New("arbitrary error")
		var interruptCalled atomic.Bool
		cmd.Cancel = func() error {
			t.Logf("Cancel called")
			interruptCalled.Store(true)
			return errArbitrary
		}
		cmd.WaitDelay = 1 * time.Millisecond
		if err := cmd.Start(); err != nil {
			t.Fatal(err)
		}
		cancel()

		err = cmd.Wait()
		t.Logf("[%d] %v", cmd.Process.Pid, err)

		// Ensure that Cancel actually had the opportunity to
		// return the error.
		if !interruptCalled.Load() {
			t.Errorf("Cancel was not called when the context was canceled")
		}

		// This test should kill the child process after 1ms,
		// To maximize compatibility with existing uses of exec.CommandContext, the
		// resulting error should be an exec.ExitError without additional wrapping.
		if _, ok := err.(*exec.ExitError); !ok {
			t.Errorf("Wait error = %v; want *exec.ExitError", err)
		}
	})

	// If Cancel returns ErrProcessDone but the process is not actually done
	// (and has to be killed), Wait should report the usual SIGKILL ExitError,
	// not the error from Cancel.
	t.Run("killed after spurious ErrProcessDone", func(t *testing.T) {
		t.Parallel()

		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()

		cmd := helperCommandContext(t, ctx, "pipetest")
		stdin, err := cmd.StdinPipe()
		if err != nil {
			t.Fatal(err)
		}
		defer stdin.Close()

		var interruptCalled atomic.Bool
		cmd.Cancel = func() error {
			t.Logf("Cancel returning an error wrapping ErrProcessDone")
			interruptCalled.Store(true)
			return fmt.Errorf("%w: stdout closed", os.ErrProcessDone)
		}
		cmd.WaitDelay = 1 * time.Millisecond
		if err := cmd.Start(); err != nil {
			t.Fatal(err)
		}
		cancel()

		err = cmd.Wait()
		t.Logf("[%d] %v", cmd.Process.Pid, err)

		// Ensure that Cancel actually had the opportunity to
		// return the error.
		if !interruptCalled.Load() {
			t.Errorf("Cancel was not called when the context was canceled")
		}

		// This test should kill the child process after 1ms,
		// To maximize compatibility with existing uses of exec.CommandContext, the
		// resulting error should be an exec.ExitError without additional wrapping.
		if ee, ok := err.(*exec.ExitError); !ok {
			t.Errorf("Wait error of type %T; want %T", err, ee)
		}
	})

	// If Cancel returns an error and the process exits with an
	// unsuccessful exit code, the process error should take precedence over the
	// Cancel error.
	t.Run("nonzero exit after error", func(t *testing.T) {
		t.Parallel()

		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()

		cmd := helperCommandContext(t, ctx, "stderrfail")
		stderr, err := cmd.StderrPipe()
		if err != nil {
			t.Fatal(err)
		}

		errArbitrary := errors.New("arbitrary error")
		interrupted := make(chan struct{})
		cmd.Cancel = func() error {
			close(interrupted)
			return errArbitrary
		}
		if err := cmd.Start(); err != nil {
			t.Fatal(err)
		}
		cancel()
		<-interrupted
		io.Copy(io.Discard, stderr)

		err = cmd.Wait()
		t.Logf("[%d] %v", cmd.Process.Pid, err)

		if ee, ok := err.(*exec.ExitError); !ok || ee.ProcessState.ExitCode() != 1 {
			t.Errorf("Wait error = %v; want exit status 1", err)
		}
	})
}

// TestConcurrentExec is a regression test for https://go.dev/issue/61080.
//
// Forking multiple child processes concurrently would sometimes hang on darwin.
// (This test hung on a gomote with -count=100 after only a few iterations.)
func TestConcurrentExec(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())

	// This test will spawn nHangs subprocesses that hang reading from stdin,
	// and nExits subprocesses that exit immediately.
	//
	// When issue #61080 was present, a long-lived "hang" subprocess would
	// occasionally inherit the fork/exec status pipe from an "exit" subprocess,
	// causing the parent process (which expects to see an EOF on that pipe almost
	// immediately) to unexpectedly block on reading from the pipe.
	var (
		nHangs       = runtime.GOMAXPROCS(0)
		nExits       = runtime.GOMAXPROCS(0)
		hangs, exits sync.WaitGroup
	)
	hangs.Add(nHangs)
	exits.Add(nExits)

	// ready is done when the goroutines have done as much work as possible to
	// prepare to create subprocesses. It isn't strictly necessary for the test,
	// but helps to increase the repro rate by making it more likely that calls to
	// syscall.StartProcess for the "hang" and "exit" goroutines overlap.
	var ready sync.WaitGroup
	ready.Add(nHangs + nExits)

	for i := 0; i < nHangs; i++ {
		go func() {
			defer hangs.Done()

			cmd := helperCommandContext(t, ctx, "pipetest")
			stdin, err := cmd.StdinPipe()
			if err != nil {
				ready.Done()
				t.Error(err)
				return
			}
			cmd.Cancel = stdin.Close
			ready.Done()

			ready.Wait()
			if err := cmd.Start(); err != nil {
				if !errors.Is(err, context.Canceled) {
					t.Error(err)
				}
				return
			}

			cmd.Wait()
		}()
	}

	for i := 0; i < nExits; i++ {
		go func() {
			defer exits.Done()

			cmd := helperCommandContext(t, ctx, "exit", "0")
			ready.Done()

			ready.Wait()
			if err := cmd.Run(); err != nil {
				t.Error(err)
			}
		}()
	}

	exits.Wait()
	cancel()
	hangs.Wait()
}

// TestPathRace tests that [Cmd.String] can be called concurrently
// with [Cmd.Start].
func TestPathRace(t *testing.T) {
	cmd := helperCommand(t, "exit", "0")

	done := make(chan struct{})
	go func() {
		out, err := cmd.CombinedOutput()
		t.Logf("%v: %v\n%s", cmd, err, out)
		close(done)
	}()

	t.Logf("running in background: %v", cmd)
	<-done
}

func TestAbsPathExec(t *testing.T) {
	testenv.MustHaveExec(t)
	testenv.MustHaveGoBuild(t) // must have GOROOT/bin/{go,gofmt}

	// A simple exec of a full path should work.
	// Go 1.22 broke this on Windows, requiring ".exe"; see #66586.
	exe := filepath.Join(testenv.GOROOT(t), "bin/gofmt")
	cmd := exec.Command(exe)
	if cmd.Path != exe {
		t.Errorf("exec.Command(%#q) set Path=%#q", exe, cmd.Path)
	}
	err := cmd.Run()
	if err != nil {
		t.Errorf("using exec.Command(%#q): %v", exe, err)
	}

	cmd = &exec.Cmd{Path: exe}
	err = cmd.Run()
	if err != nil {
		t.Errorf("using exec.Cmd{Path: %#q}: %v", cmd.Path, err)
	}

	cmd = &exec.Cmd{Path: "gofmt", Dir: "/"}
	err = cmd.Run()
	if err == nil {
		t.Errorf("using exec.Cmd{Path: %#q}: unexpected success", cmd.Path)
	}

	// A simple exec after modifying Cmd.Path should work.
	// This broke on Windows. See go.dev/issue/68314.
	t.Run("modified", func(t *testing.T) {
		if exec.Command(filepath.Join(testenv.GOROOT(t), "bin/go")).Run() == nil {
			// The implementation of the test case below relies on the go binary
			// exiting with a non-zero exit code when run without any arguments.
			// In the unlikely case that changes, we need to use another binary.
			t.Fatal("test case needs updating to verify fix for go.dev/issue/68314")
		}
		exe1 := filepath.Join(testenv.GOROOT(t), "bin/go")
		exe2 := filepath.Join(testenv.GOROOT(t), "bin/gofmt")
		cmd := exec.Command(exe1)
		cmd.Path = exe2
		cmd.Args = []string{cmd.Path}
		err := cmd.Run()
		if err != nil {
			t.Error("ran wrong binary")
		}
	})
}

// Calling Start twice is an error, regardless of outcome.
func TestStart_twice(t *testing.T) {
	testenv.MustHaveExec(t)

	cmd := exec.Command("/bin/nonesuch")
	for i, want := range []string{
		cond(runtime.GOOS == "windows",
			`exec: "/bin/nonesuch": executable file not found in %PATH%`,
			"fork/exec /bin/nonesuch: no such file or directory"),
		"exec: already started",
	} {
		err := cmd.Start()
		if got := fmt.Sprint(err); got != want {
			t.Errorf("Start call #%d return err %q, want %q", i+1, got, want)
		}
	}
}

func cond[T any](cond bool, t, f T) T {
	if cond {
		return t
	} else {
		return f
	}
}
