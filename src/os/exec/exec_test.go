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
	"fmt"
	"internal/poll"
	"internal/testenv"
	"io"
	"io/ioutil"
	"log"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"testing"
	"time"
)

// haveUnexpectedFDs is set at init time to report whether any
// file descriptors were open at program start.
var haveUnexpectedFDs bool

// unfinalizedFiles holds files that should not be finalized,
// because that would close the associated file descriptor,
// which we don't want to do.
var unfinalizedFiles []*os.File

func init() {
	if os.Getenv("GO_WANT_HELPER_PROCESS") == "1" {
		return
	}
	if runtime.GOOS == "windows" {
		return
	}
	for fd := uintptr(3); fd <= 100; fd++ {
		if poll.IsPollDescriptor(fd) {
			continue
		}
		// We have no good portable way to check whether an FD is open.
		// We use NewFile to create a *os.File, which lets us
		// know whether it is open, but then we have to cope with
		// the finalizer on the *os.File.
		f := os.NewFile(fd, "")
		if _, err := f.Stat(); err != nil {
			// Close the file to clear the finalizer.
			// We expect the Close to fail.
			f.Close()
		} else {
			fmt.Printf("fd %d open at test start\n", fd)
			haveUnexpectedFDs = true
			// Use a global variable to avoid running
			// the finalizer, which would close the FD.
			unfinalizedFiles = append(unfinalizedFiles, f)
		}
	}
}

func helperCommandContext(t *testing.T, ctx context.Context, s ...string) (cmd *exec.Cmd) {
	testenv.MustHaveExec(t)

	cs := []string{"-test.run=TestHelperProcess", "--"}
	cs = append(cs, s...)
	if ctx != nil {
		cmd = exec.CommandContext(ctx, os.Args[0], cs...)
	} else {
		cmd = exec.Command(os.Args[0], cs...)
	}
	cmd.Env = append(os.Environ(), "GO_WANT_HELPER_PROCESS=1")
	return cmd
}

func helperCommand(t *testing.T, s ...string) *exec.Cmd {
	return helperCommandContext(t, nil, s...)
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
	testenv.MustHaveExec(t)

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

	cmd := exec.Command(filepath.Join(dirBase, base), "-test.run=TestHelperProcess", "--", "echo", "foo")
	cmd.Dir = parentDir
	cmd.Env = []string{"GO_WANT_HELPER_PROCESS=1"}

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
	s := string(bs)
	sp := strings.SplitN(s, "\n", 2)
	if len(sp) != 2 {
		t.Fatalf("expected two lines from cat; got %q", s)
	}
	errLine, body := sp[0], sp[1]
	if !strings.HasPrefix(errLine, "Error: open /bogus/file.foo") {
		t.Errorf("expected stderr to complain about file; got %q", errLine)
	}
	if !strings.Contains(body, "func TestHelperProcess(t *testing.T)") {
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
	// If we are reading from /proc/self/fd we (should) get an exact result.
	tolerance := 0

	// Reading /proc/self/fd is more reliable than calling lsof, so try that
	// first.
	numOpenFDs := func() (int, []byte, error) {
		fds, err := ioutil.ReadDir("/proc/self/fd")
		if err != nil {
			return 0, nil, err
		}
		return len(fds), nil, nil
	}
	want, before, err := numOpenFDs()
	if err != nil {
		// We encountered a problem reading /proc/self/fd (we might be on
		// a platform that doesn't have it). Fall back onto lsof.
		t.Logf("using lsof because: %v", err)
		numOpenFDs = func() (int, []byte, error) {
			// Android's stock lsof does not obey the -p option,
			// so extra filtering is needed.
			// https://golang.org/issue/10206
			if runtime.GOOS == "android" {
				// numOpenFDsAndroid handles errors itself and
				// might skip or fail the test.
				n, lsof := numOpenFDsAndroid(t)
				return n, lsof, nil
			}
			lsof, err := exec.Command("lsof", "-b", "-n", "-p", strconv.Itoa(os.Getpid())).Output()
			return bytes.Count(lsof, []byte("\n")), lsof, err
		}

		// lsof may see file descriptors associated with the fork itself,
		// so we allow some extra margin if we have to use it.
		// https://golang.org/issue/19243
		tolerance = 5

		// Retry reading the number of open file descriptors.
		want, before, err = numOpenFDs()
		if err != nil {
			t.Log(err)
			t.Skipf("skipping test; error finding or running lsof")
		}
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
	got, after, err := numOpenFDs()
	if err != nil {
		// numOpenFDs has already succeeded once, it should work here.
		t.Errorf("unexpected failure: %v", err)
	}
	if got-want > tolerance {
		t.Errorf("number of open file descriptors changed: got %v, want %v", got, want)
		if before != nil {
			t.Errorf("before:\n%v\n", before)
		}
		if after != nil {
			t.Errorf("after:\n%v\n", after)
		}
	}
}

func numOpenFDsAndroid(t *testing.T) (n int, lsof []byte) {
	raw, err := exec.Command("lsof").Output()
	if err != nil {
		t.Skip("skipping test; error finding or running lsof")
	}

	// First find the PID column index by parsing the first line, and
	// select lines containing pid in the column.
	pid := []byte(strconv.Itoa(os.Getpid()))
	pidCol := -1

	s := bufio.NewScanner(bytes.NewReader(raw))
	for s.Scan() {
		line := s.Bytes()
		fields := bytes.Fields(line)
		if pidCol < 0 {
			for i, v := range fields {
				if bytes.Equal(v, []byte("PID")) {
					pidCol = i
					break
				}
			}
			lsof = append(lsof, line...)
			continue
		}
		if bytes.Equal(fields[pidCol], pid) {
			lsof = append(lsof, '\n')
			lsof = append(lsof, line...)
		}
	}
	if pidCol < 0 {
		t.Fatal("error processing lsof output: unexpected header format")
	}
	if err := s.Err(); err != nil {
		t.Fatalf("error processing lsof output: %v", err)
	}
	return bytes.Count(lsof, []byte("\n")), lsof
}

// basefds returns the number of expected file descriptors
// to be present in a process at start.
// stdin, stdout, stderr, epoll/kqueue, epoll/kqueue pipe, maybe testlog
func basefds() uintptr {
	n := os.Stderr.Fd() + 1
	// The poll (epoll/kqueue) descriptor can be numerically
	// either between stderr and the testlog-fd, or after
	// testlog-fd.
	for poll.IsPollDescriptor(n) {
		n++
	}
	for _, arg := range os.Args {
		if strings.HasPrefix(arg, "-test.testlogfile=") {
			n++
		}
	}
	return n
}

func TestExtraFilesFDShuffle(t *testing.T) {
	t.Skip("flaky test; see https://golang.org/issue/5780")
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
	ch := make(chan string, 1)
	go func(ch chan string) {
		buf := make([]byte, 512)
		n, err := stderr.Read(buf)
		if err != nil {
			t.Errorf("Read: %s", err)
			ch <- err.Error()
		} else {
			ch <- string(buf[:n])
		}
		close(ch)
	}(ch)
	select {
	case m := <-ch:
		if m != expected {
			t.Errorf("Read: '%s' not '%s'", m, expected)
		}
	case <-time.After(5 * time.Second):
		t.Errorf("Read timedout")
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
	ts.Config.ErrorLog = log.New(ioutil.Discard, "", 0)
	ts.StartTLS()
	defer ts.Close()
	_, err = http.Get(ts.URL)
	if err == nil {
		t.Errorf("success trying to fetch %s; want an error", ts.URL)
	}

	tf, err := ioutil.TempFile("", "")
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

	c := helperCommand(t, "read3")
	var stdout, stderr bytes.Buffer
	c.Stdout = &stdout
	c.Stderr = &stderr
	c.ExtraFiles = []*os.File{tf}
	err = c.Run()
	if err != nil {
		t.Fatalf("Run: %v\n--- stdout:\n%s--- stderr:\n%s", err, stdout.Bytes(), stderr.Bytes())
	}
	if stdout.String() != text {
		t.Errorf("got stdout %q, stderr %q; want %q on stdout", stdout.String(), stderr.String(), text)
	}
}

func TestExtraFilesRace(t *testing.T) {
	if runtime.GOOS == "windows" {
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

// TestHelperProcess isn't a real test. It's used as a helper process
// for TestParameterRun.
func TestHelperProcess(*testing.T) {
	if os.Getenv("GO_WANT_HELPER_PROCESS") != "1" {
		return
	}
	defer os.Exit(0)

	// Determine which command to use to display open files.
	ofcmd := "lsof"
	switch runtime.GOOS {
	case "dragonfly", "freebsd", "netbsd", "openbsd":
		ofcmd = "fstat"
	case "plan9":
		ofcmd = "/bin/cat"
	case "aix":
		ofcmd = "procfiles"
	}

	args := os.Args
	for len(args) > 0 {
		if args[0] == "--" {
			args = args[1:]
			break
		}
		args = args[1:]
	}
	if len(args) == 0 {
		fmt.Fprintf(os.Stderr, "No command\n")
		os.Exit(2)
	}

	cmd, args := args[0], args[1:]
	switch cmd {
	case "echo":
		iargs := []interface{}{}
		for _, s := range args {
			iargs = append(iargs, s)
		}
		fmt.Println(iargs...)
	case "echoenv":
		for _, s := range args {
			fmt.Println(os.Getenv(s))
		}
		os.Exit(0)
	case "cat":
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
	case "pipetest":
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
	case "stdinClose":
		b, err := ioutil.ReadAll(os.Stdin)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error: %v\n", err)
			os.Exit(1)
		}
		if s := string(b); s != stdinCloseTestString {
			fmt.Fprintf(os.Stderr, "Error: Read %q, want %q", s, stdinCloseTestString)
			os.Exit(1)
		}
		os.Exit(0)
	case "read3": // read fd 3
		fd3 := os.NewFile(3, "fd3")
		bs, err := ioutil.ReadAll(fd3)
		if err != nil {
			fmt.Printf("ReadAll from fd 3: %v", err)
			os.Exit(1)
		}
		// Now verify that there are no other open fds.
		var files []*os.File
		for wantfd := basefds() + 1; wantfd <= 100; wantfd++ {
			if poll.IsPollDescriptor(wantfd) {
				continue
			}
			f, err := os.Open(os.Args[0])
			if err != nil {
				fmt.Printf("error opening file with expected fd %d: %v", wantfd, err)
				os.Exit(1)
			}
			if got := f.Fd(); got != wantfd {
				fmt.Printf("leaked parent file. fd = %d; want %d\n", got, wantfd)
				var args []string
				switch runtime.GOOS {
				case "plan9":
					args = []string{fmt.Sprintf("/proc/%d/fd", os.Getpid())}
				case "aix":
					args = []string{fmt.Sprint(os.Getpid())}
				default:
					args = []string{"-p", fmt.Sprint(os.Getpid())}
				}
				cmd := exec.Command(ofcmd, args...)
				out, err := cmd.CombinedOutput()
				if err != nil {
					fmt.Fprintf(os.Stderr, "%s failed: %v\n", strings.Join(cmd.Args, " "), err)
				}
				fmt.Printf("%s", out)
				os.Exit(1)
			}
			files = append(files, f)
		}
		for _, f := range files {
			f.Close()
		}
		// Referring to fd3 here ensures that it is not
		// garbage collected, and therefore closed, while
		// executing the wantfd loop above. It doesn't matter
		// what we do with fd3 as long as we refer to it;
		// closing it is the easy choice.
		fd3.Close()
		os.Stdout.Write(bs)
	case "exit":
		n, _ := strconv.Atoi(args[0])
		os.Exit(n)
	case "describefiles":
		f := os.NewFile(3, fmt.Sprintf("fd3"))
		ln, err := net.FileListener(f)
		if err == nil {
			fmt.Printf("fd3: listener %s\n", ln.Addr())
			ln.Close()
		}
		os.Exit(0)
	case "extraFilesAndPipes":
		n, _ := strconv.Atoi(args[0])
		pipes := make([]*os.File, n)
		for i := 0; i < n; i++ {
			pipes[i] = os.NewFile(uintptr(3+i), strconv.Itoa(i))
		}
		response := ""
		for i, r := range pipes {
			ch := make(chan string, 1)
			go func(c chan string) {
				buf := make([]byte, 10)
				n, err := r.Read(buf)
				if err != nil {
					fmt.Fprintf(os.Stderr, "Child: read error: %v on pipe %d\n", err, i)
					os.Exit(1)
				}
				c <- string(buf[:n])
				close(c)
			}(ch)
			select {
			case m := <-ch:
				response = response + m
			case <-time.After(5 * time.Second):
				fmt.Fprintf(os.Stderr, "Child: Timeout reading from pipe: %d\n", i)
				os.Exit(1)
			}
		}
		fmt.Fprintf(os.Stderr, "child: %s", response)
		os.Exit(0)
	case "exec":
		cmd := exec.Command(args[1])
		cmd.Dir = args[0]
		output, err := cmd.CombinedOutput()
		if err != nil {
			fmt.Fprintf(os.Stderr, "Child: %s %s", err, string(output))
			os.Exit(1)
		}
		fmt.Printf("%s", string(output))
		os.Exit(0)
	case "lookpath":
		p, err := exec.LookPath(args[0])
		if err != nil {
			fmt.Fprintf(os.Stderr, "LookPath failed: %v\n", err)
			os.Exit(1)
		}
		fmt.Print(p)
		os.Exit(0)
	case "stderrfail":
		fmt.Fprintf(os.Stderr, "some stderr text\n")
		os.Exit(1)
	case "sleep":
		time.Sleep(3 * time.Second)
		os.Exit(0)
	default:
		fmt.Fprintf(os.Stderr, "Unknown command %q\n", cmd)
		os.Exit(2)
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
	testenv.MustHaveExec(t)

	testWith := func(r io.Reader) func(*testing.T) {
		return func(t *testing.T) {
			cmd := helperCommand(t, "echo", "foo")
			var out bytes.Buffer
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
	testenv.MustHaveExec(t)

	if runtime.GOOS == "windows" || runtime.GOOS == "plan9" {
		t.Skipf("skipping test on %s - no yes command", runtime.GOOS)
	}
	cmd := exec.Command("yes")
	cmd.Stdout = new(badWriter)
	c := make(chan int, 1)
	go func() {
		err := cmd.Run()
		if err == nil {
			t.Errorf("yes completed successfully")
		}
		c <- 1
	}()
	select {
	case <-c:
		// ok
	case <-time.After(5 * time.Second):
		t.Fatalf("yes got stuck writing to bad writer")
	}
}

func TestOutputStderrCapture(t *testing.T) {
	testenv.MustHaveExec(t)

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
	waitErr := make(chan error, 1)
	go func() {
		waitErr <- c.Wait()
	}()
	cancel()
	select {
	case err := <-waitErr:
		if err == nil {
			t.Fatal("expected Wait failure")
		}
	case <-time.After(3 * time.Second):
		t.Fatal("timeout waiting for child process death")
	}
}

func TestContextCancel(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	c := helperCommandContext(t, ctx, "cat")

	r, w, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	c.Stdin = r

	stdout, err := c.StdoutPipe()
	if err != nil {
		t.Fatal(err)
	}
	readDone := make(chan struct{})
	go func() {
		defer close(readDone)
		var a [1024]byte
		for {
			n, err := stdout.Read(a[:])
			if err != nil {
				if err != io.EOF {
					t.Errorf("unexpected read error: %v", err)
				}
				return
			}
			t.Logf("%s", a[:n])
		}
	}()

	if err := c.Start(); err != nil {
		t.Fatal(err)
	}

	if err := r.Close(); err != nil {
		t.Fatal(err)
	}

	if _, err := io.WriteString(w, "echo"); err != nil {
		t.Fatal(err)
	}

	cancel()

	// Calling cancel should have killed the process, so writes
	// should now fail.  Give the process a little while to die.
	start := time.Now()
	for {
		if _, err := io.WriteString(w, "echo"); err != nil {
			break
		}
		if time.Since(start) > time.Second {
			t.Fatal("canceling context did not stop program")
		}
		time.Sleep(time.Millisecond)
	}

	if err := w.Close(); err != nil {
		t.Errorf("error closing write end of pipe: %v", err)
	}
	<-readDone

	if err := c.Wait(); err == nil {
		t.Error("program unexpectedly exited successfully")
	} else {
		t.Logf("exit status: %v", err)
	}
}

// test that environment variables are de-duped.
func TestDedupEnvEcho(t *testing.T) {
	testenv.MustHaveExec(t)

	cmd := helperCommand(t, "echoenv", "FOO")
	cmd.Env = append(cmd.Env, "FOO=bad", "FOO=good")
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

// start a child process without the user code explicitly starting
// with a copy of the parent's. (The Windows SYSTEMROOT issue: Issue
// 25210)
func TestChildCriticalEnv(t *testing.T) {
	testenv.MustHaveExec(t)
	if runtime.GOOS != "windows" {
		t.Skip("only testing on Windows")
	}
	cmd := helperCommand(t, "echoenv", "SYSTEMROOT")
	cmd.Env = []string{"GO_WANT_HELPER_PROCESS=1"}
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatal(err)
	}
	if strings.TrimSpace(string(out)) == "" {
		t.Error("no SYSTEMROOT found")
	}
}
