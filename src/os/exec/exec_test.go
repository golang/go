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
	"path/filepath"
	"reflect"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"
)

// haveUnexpectedFDs is set at init time to report whether any file descriptors
// were open at program start.
var haveUnexpectedFDs bool

func init() {
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

		code := m.Run()
		if code == 0 && flag.Lookup("test.run").Value.String() == "" && flag.Lookup("test.list").Value.String() == "" {
			for cmd := range helperCommands {
				if _, ok := helperCommandUsed.Load(cmd); !ok {
					fmt.Fprintf(os.Stderr, "helper command unused: %q\n", cmd)
					code = 1
				}
			}
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
	testenv.MustHaveExec(t)

	cs := append([]string{name}, args...)
	if ctx != nil {
		cmd = exec.CommandContext(ctx, exePath(t), cs...)
	} else {
		cmd = exec.Command(exePath(t), cs...)
	}
	return cmd
}

// exePath returns the path to the running executable.
func exePath(t testing.TB) string {
	exeOnce.Do(func() {
		// Use os.Executable instead of os.Args[0] in case the caller modifies
		// cmd.Dir: if the test binary is invoked like "./exec.test", it should
		// not fail spuriously.
		exeOnce.path, exeOnce.err = os.Executable()
	})

	if exeOnce.err != nil {
		if t == nil {
			panic(exeOnce.err)
		}
		t.Fatal(exeOnce.err)
	}

	return exeOnce.path
}

var exeOnce struct {
	path string
	err  error
	sync.Once
}

var helperCommandUsed sync.Map

var helperCommands = map[string]func(...string){
	"echo":               cmdEcho,
	"echoenv":            cmdEchoEnv,
	"cat":                cmdCat,
	"pipetest":           cmdPipeTest,
	"stdinClose":         cmdStdinClose,
	"exit":               cmdExit,
	"describefiles":      cmdDescribeFiles,
	"extraFilesAndPipes": cmdExtraFilesAndPipes,
	"stderrfail":         cmdStderrFail,
	"yes":                cmdYes,
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
	f := os.NewFile(3, fmt.Sprintf("fd3"))
	ln, err := net.FileListener(f)
	if err == nil {
		fmt.Printf("fd3: listener %s\n", ln.Addr())
		ln.Close()
	}
}

func cmdExtraFilesAndPipes(args ...string) {
	n, _ := strconv.Atoi(args[0])
	pipes := make([]*os.File, n)
	for i := 0; i < n; i++ {
		pipes[i] = os.NewFile(uintptr(3+i), strconv.Itoa(i))
	}
	response := ""
	for i, r := range pipes {
		buf := make([]byte, 10)
		n, err := r.Read(buf)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Child: read error: %v on pipe %d\n", err, i)
			os.Exit(1)
		}
		response = response + string(buf[:n])
	}
	fmt.Fprintf(os.Stderr, "child: %s", response)
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
	bs, err := helperCommand(t, "echo", "foo bar", "baz").Output()
	if err != nil {
		t.Errorf("echo: %v", err)
	}
	if g, e := string(bs), "foo bar baz\n"; g != e {
		t.Errorf("echo: want %q, got %q", e, g)
	}
}

func TestCommandRelativeName(t *testing.T) {
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
	// Can't run a non-existent executable
	err := exec.Command("/no-exist-executable").Run()
	if err == nil {
		t.Error("expected error from /no-exist-executable")
	}
}

func TestExitStatus(t *testing.T) {
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
	go func() {
		_, err := io.Copy(stdin, strings.NewReader(stdinCloseTestString))
		check("Copy", err)
		// Before the fix, this next line would race with cmd.Wait.
		check("Close", stdin.Close())
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
	cmd := helperCommand(t, "stdinClose")
	stdin, err := cmd.StdinPipe()
	if err != nil {
		t.Fatalf("StdinPipe: %v", err)
	}
	if err := cmd.Start(); err != nil {
		t.Fatalf("Start: %v", err)
	}
	go func() {
		// We don't check the error return of Kill. It is
		// possible that the process has already exited, in
		// which case Kill will return an error "process
		// already finished". The purpose of this test is to
		// see whether the race detector reports an error; it
		// doesn't matter whether this Kill succeeds or not.
		cmd.Process.Kill()
	}()
	go func() {
		// Send the wrong string, so that the child fails even
		// if the other goroutine doesn't manage to kill it first.
		// This test is to check that the race detector does not
		// falsely report an error, so it doesn't matter how the
		// child process fails.
		io.Copy(stdin, strings.NewReader("unexpected string"))
		if err := stdin.Close(); err != nil {
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

	openFDs := func() []uintptr {
		var fds []uintptr
		for i := uintptr(0); i < 100; i++ {
			if fdtest.Exists(i) {
				fds = append(fds, i)
			}
		}
		return fds
	}

	want := openFDs()
	for i := 0; i < 6; i++ {
		cmd := exec.Command("something-that-does-not-exist-executable")
		cmd.StdoutPipe()
		cmd.StderrPipe()
		cmd.StdinPipe()
		if err := cmd.Run(); err == nil {
			t.Fatal("unexpected success")
		}
	}
	got := openFDs()
	if !reflect.DeepEqual(got, want) {
		t.Errorf("set of open file descriptors changed: got %v, want %v", got, want)
	}
}

func TestExtraFilesFDShuffle(t *testing.T) {
	maySkipHelperCommand("extraFilesAndPipes")
	testenv.SkipFlaky(t, 5780)
	switch runtime.GOOS {
	case "windows":
		t.Skip("no operating system support; skipping")
	}

	// syscall.StartProcess maps all the FDs passed to it in
	// ProcAttr.Files (the concatenation of stdin,stdout,stderr and
	// ExtraFiles) into consecutive FDs in the child, that is:
	// Files{11, 12, 6, 7, 9, 3} should result in the file
	// represented by FD 11 in the parent being made available as 0
	// in the child, 12 as 1, etc.
	//
	// We want to test that FDs in the child do not get overwritten
	// by one another as this shuffle occurs. The original implementation
	// was buggy in that in some data dependent cases it would overwrite
	// stderr in the child with one of the ExtraFile members.
	// Testing for this case is difficult because it relies on using
	// the same FD values as that case. In particular, an FD of 3
	// must be at an index of 4 or higher in ProcAttr.Files and
	// the FD of the write end of the Stderr pipe (as obtained by
	// StderrPipe()) must be the same as the size of ProcAttr.Files;
	// therefore we test that the read end of this pipe (which is what
	// is returned to the parent by StderrPipe() being one less than
	// the size of ProcAttr.Files, i.e. 3+len(cmd.ExtraFiles).
	//
	// Moving this test case around within the overall tests may
	// affect the FDs obtained and hence the checks to catch these cases.
	npipes := 2
	c := helperCommand(t, "extraFilesAndPipes", strconv.Itoa(npipes+1))
	rd, wr, _ := os.Pipe()
	defer rd.Close()
	if rd.Fd() != 3 {
		t.Errorf("bad test value for test pipe: fd %d", rd.Fd())
	}
	stderr, _ := c.StderrPipe()
	wr.WriteString("_LAST")
	wr.Close()

	pipes := make([]struct {
		r, w *os.File
	}, npipes)
	data := []string{"a", "b"}

	for i := 0; i < npipes; i++ {
		r, w, err := os.Pipe()
		if err != nil {
			t.Fatalf("unexpected error creating pipe: %s", err)
		}
		pipes[i].r = r
		pipes[i].w = w
		w.WriteString(data[i])
		c.ExtraFiles = append(c.ExtraFiles, pipes[i].r)
		defer func() {
			r.Close()
			w.Close()
		}()
	}
	// Put fd 3 at the end.
	c.ExtraFiles = append(c.ExtraFiles, rd)

	stderrFd := int(stderr.(*os.File).Fd())
	if stderrFd != ((len(c.ExtraFiles) + 3) - 1) {
		t.Errorf("bad test value for stderr pipe")
	}

	expected := "child: " + strings.Join(data, "") + "_LAST"

	err := c.Start()
	if err != nil {
		t.Fatalf("Run: %v", err)
	}

	buf := make([]byte, 512)
	n, err := stderr.Read(buf)
	if err != nil {
		t.Errorf("Read: %s", err)
	} else {
		if m := string(buf[:n]); m != expected {
			t.Errorf("Read: '%s' not '%s'", m, expected)
		}
	}
	c.Wait()
}

func TestExtraFiles(t *testing.T) {
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
	testenv.MustInternalLink(t)

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

	c := exec.Command(testenv.GoToolPath(t), "build", "-o", exe, "read3.go")
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
	testWith := func(r io.Reader) func(*testing.T) {
		return func(t *testing.T) {
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
	cmd := helperCommand(t, "yes")
	cmd.Stdout = new(badWriter)
	err := cmd.Run()
	if err == nil {
		t.Errorf("yes unexpectedly completed successfully")
	}
}

func TestOutputStderrCapture(t *testing.T) {
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

func TestString(t *testing.T) {
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
