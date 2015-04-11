// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Use an external test to avoid os/exec -> net/http -> crypto/x509 -> os/exec
// circular dependency on non-cgo darwin.

package exec_test

import (
	"bufio"
	"bytes"
	"fmt"
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

// iOS cannot fork
var iOS = runtime.GOOS == "darwin" && (runtime.GOARCH == "arm" || runtime.GOARCH == "arm64")

func helperCommand(t *testing.T, s ...string) *exec.Cmd {
	if runtime.GOOS == "nacl" || iOS {
		t.Skipf("skipping on %s/%s, cannot fork", runtime.GOOS, runtime.GOARCH)
	}
	cs := []string{"-test.run=TestHelperProcess", "--"}
	cs = append(cs, s...)
	cmd := exec.Command(os.Args[0], cs...)
	cmd.Env = []string{"GO_WANT_HELPER_PROCESS=1"}
	return cmd
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
	if iOS {
		t.Skip("skipping on darwin/%s, cannot fork", runtime.GOARCH)
	}

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

func TestNoExistBinary(t *testing.T) {
	// Can't run a non-existent binary
	err := exec.Command("/no-exist-binary").Run()
	if err == nil {
		t.Error("expected error from /no-exist-binary")
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

// Issue 5071
func TestPipeLookPathLeak(t *testing.T) {
	fd0, lsof0 := numOpenFDS(t)
	for i := 0; i < 4; i++ {
		cmd := exec.Command("something-that-does-not-exist-binary")
		cmd.StdoutPipe()
		cmd.StderrPipe()
		cmd.StdinPipe()
		if err := cmd.Run(); err == nil {
			t.Fatal("unexpected success")
		}
	}
	for triesLeft := 3; triesLeft >= 0; triesLeft-- {
		open, lsof := numOpenFDS(t)
		fdGrowth := open - fd0
		if fdGrowth > 2 {
			if triesLeft > 0 {
				// Work around what appears to be a race with Linux's
				// proc filesystem (as used by lsof). It seems to only
				// be eventually consistent. Give it awhile to settle.
				// See golang.org/issue/7808
				time.Sleep(100 * time.Millisecond)
				continue
			}
			t.Errorf("leaked %d fds; want ~0; have:\n%s\noriginally:\n%s", fdGrowth, lsof, lsof0)
		}
		break
	}
}

func numOpenFDS(t *testing.T) (n int, lsof []byte) {
	if runtime.GOOS == "android" {
		// Android's stock lsof does not obey the -p option,
		// so extra filtering is needed. (golang.org/issue/10206)
		return numOpenFDsAndroid(t)
	}

	lsof, err := exec.Command("lsof", "-b", "-n", "-p", strconv.Itoa(os.Getpid())).Output()
	if err != nil {
		t.Skip("skipping test; error finding or running lsof")
	}
	return bytes.Count(lsof, []byte("\n")), lsof
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

var testedAlreadyLeaked = false

// basefds returns the number of expected file descriptors
// to be present in a process at start.
func basefds() uintptr {
	return os.Stderr.Fd() + 1
}

func closeUnexpectedFds(t *testing.T, m string) {
	for fd := basefds(); fd <= 101; fd++ {
		err := os.NewFile(fd, "").Close()
		if err == nil {
			t.Logf("%s: Something already leaked - closed fd %d", m, fd)
		}
	}
}

func TestExtraFilesFDShuffle(t *testing.T) {
	t.Skip("flaky test; see http://golang.org/issue/5780")
	switch runtime.GOOS {
	case "darwin":
		// TODO(cnicolaou): http://golang.org/issue/2603
		// leads to leaked file descriptors in this test when it's
		// run from a builder.
		closeUnexpectedFds(t, "TestExtraFilesFDShuffle")
	case "netbsd":
		// http://golang.org/issue/3955
		closeUnexpectedFds(t, "TestExtraFilesFDShuffle")
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
	// was buggy in that in some data dependent cases it would ovewrite
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
			t.Fatalf("Read: %s", err)
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
	switch runtime.GOOS {
	case "nacl", "windows":
		t.Skipf("skipping test on %q", runtime.GOOS)
	}
	if iOS {
		t.Skipf("skipping test on %s/%s, cannot fork", runtime.GOOS, runtime.GOARCH)
	}

	// Ensure that file descriptors have not already been leaked into
	// our environment.
	if !testedAlreadyLeaked {
		testedAlreadyLeaked = true
		closeUnexpectedFds(t, "TestExtraFiles")
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
	_, err = tf.Seek(0, os.SEEK_SET)
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
		t.Fatalf("Run: %v; stdout %q, stderr %q", err, stdout.Bytes(), stderr.Bytes())
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
		switch runtime.GOOS {
		case "dragonfly":
			// TODO(jsing): Determine why DragonFly is leaking
			// file descriptors...
		case "darwin":
			// TODO(bradfitz): broken? Sometimes.
			// http://golang.org/issue/2603
			// Skip this additional part of the test for now.
		case "netbsd":
			// TODO(jsing): This currently fails on NetBSD due to
			// the cloned file descriptors that result from opening
			// /dev/urandom.
			// http://golang.org/issue/3955
		case "plan9":
			// TODO(0intro): Determine why Plan 9 is leaking
			// file descriptors.
			// http://golang.org/issue/7118
		case "solaris":
			// TODO(aram): This fails on Solaris because libc opens
			// its own files, as it sees fit. Darwin does the same,
			// see: http://golang.org/issue/2603
		default:
			// Now verify that there are no other open fds.
			var files []*os.File
			for wantfd := basefds() + 1; wantfd <= 100; wantfd++ {
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
					default:
						args = []string{"-p", fmt.Sprint(os.Getpid())}
					}
					out, _ := exec.Command(ofcmd, args...).CombinedOutput()
					fmt.Print(string(out))
					os.Exit(1)
				}
				files = append(files, f)
			}
			for _, f := range files {
				f.Close()
			}
		}
		// Referring to fd3 here ensures that it is not
		// garbage collected, and therefore closed, while
		// executing the wantfd loop above.  It doesn't matter
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
	default:
		fmt.Fprintf(os.Stderr, "Unknown command %q\n", cmd)
		os.Exit(2)
	}
}
