// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syscall_test

import (
	"bytes"
	"fmt"
	"internal/testenv"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"
	"syscall"
	"testing"
	"time"
	"unsafe"
)

func TestEscapeArg(t *testing.T) {
	var tests = []struct {
		input, output string
	}{
		{``, `""`},
		{`a`, `a`},
		{` `, `" "`},
		{`\`, `\`},
		{`"`, `\"`},
		{`\"`, `\\\"`},
		{`\\"`, `\\\\\"`},
		{`\\ `, `"\\ "`},
		{` \\`, `" \\\\"`},
		{`a `, `"a "`},
		{`C:\`, `C:\`},
		{`C:\Program Files (x32)\Common\`, `"C:\Program Files (x32)\Common\\"`},
		{`C:\Users\Игорь\`, `C:\Users\Игорь\`},
		{`Андрей\file`, `Андрей\file`},
		{`C:\Windows\temp`, `C:\Windows\temp`},
		{`c:\temp\newfile`, `c:\temp\newfile`},
		{`\\?\C:\Windows`, `\\?\C:\Windows`},
		{`\\?\`, `\\?\`},
		{`\\.\C:\Windows\`, `\\.\C:\Windows\`},
		{`\\server\share\file`, `\\server\share\file`},
		{`\\newserver\tempshare\really.txt`, `\\newserver\tempshare\really.txt`},
	}
	for _, test := range tests {
		if got := syscall.EscapeArg(test.input); got != test.output {
			t.Errorf("EscapeArg(%#q) = %#q, want %#q", test.input, got, test.output)
		}
	}
}

func TestEnvBlockSorted(t *testing.T) {
	tests := []struct {
		env  []string
		want []string
	}{
		{},
		{
			env:  []string{"A=1"},
			want: []string{"A=1"},
		},
		{
			env:  []string{"A=1", "B=2", "C=3"},
			want: []string{"A=1", "B=2", "C=3"},
		},
		{
			env:  []string{"C=3", "B=2", "A=1"},
			want: []string{"A=1", "B=2", "C=3"},
		},
		{
			env:  []string{"c=3", "B=2", "a=1"},
			want: []string{"a=1", "B=2", "c=3"},
		},
	}
	for _, tt := range tests {
		got := syscall.EnvSorted(tt.env)
		if !slices.Equal(got, tt.want) {
			t.Errorf("EnvSorted(%q) = %q, want %q", tt.env, got, tt.want)
		}
	}
}

func TestChangingProcessParent(t *testing.T) {
	if os.Getenv("GO_WANT_HELPER_PROCESS") == "parent" {
		// in parent process

		// Parent does nothing. It is just used as a parent of a child process.
		time.Sleep(time.Minute)
		os.Exit(0)
	}

	if os.Getenv("GO_WANT_HELPER_PROCESS") == "child" {
		// in child process
		dumpPath := os.Getenv("GO_WANT_HELPER_PROCESS_FILE")
		if dumpPath == "" {
			fmt.Fprintf(os.Stderr, "Dump file path cannot be blank.")
			os.Exit(1)
		}
		err := os.WriteFile(dumpPath, []byte(fmt.Sprintf("%d", os.Getppid())), 0644)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error writing dump file: %v", err)
			os.Exit(2)
		}
		os.Exit(0)
	}

	// run parent process

	parent := exec.Command(testenv.Executable(t), "-test.run=^TestChangingProcessParent$")
	parent.Env = append(os.Environ(), "GO_WANT_HELPER_PROCESS=parent")
	err := parent.Start()
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		parent.Process.Kill()
		parent.Wait()
	}()

	// run child process

	const _PROCESS_CREATE_PROCESS = 0x0080
	const _PROCESS_DUP_HANDLE = 0x0040
	childDumpPath := filepath.Join(t.TempDir(), "ppid.txt")
	ph, err := syscall.OpenProcess(_PROCESS_CREATE_PROCESS|_PROCESS_DUP_HANDLE|syscall.PROCESS_QUERY_INFORMATION,
		false, uint32(parent.Process.Pid))
	if err != nil {
		t.Fatal(err)
	}
	defer syscall.CloseHandle(ph)

	child := exec.Command(testenv.Executable(t), "-test.run=^TestChangingProcessParent$")
	child.Env = append(os.Environ(),
		"GO_WANT_HELPER_PROCESS=child",
		"GO_WANT_HELPER_PROCESS_FILE="+childDumpPath)
	child.SysProcAttr = &syscall.SysProcAttr{ParentProcess: ph}
	childOutput, err := child.CombinedOutput()
	if err != nil {
		t.Errorf("child failed: %v: %v", err, string(childOutput))
	}
	childOutput, err = os.ReadFile(childDumpPath)
	if err != nil {
		t.Fatalf("reading child output failed: %v", err)
	}
	if got, want := string(childOutput), fmt.Sprintf("%d", parent.Process.Pid); got != want {
		t.Fatalf("child output: want %q, got %q", want, got)
	}
}

func TestPseudoConsoleProcess(t *testing.T) {
	pty, err := newConPty()
	if err != nil {
		t.Errorf("create pty failed: %v", err)
	}

	defer pty.Close()
	cmd := exec.Command("cmd")
	cmd.SysProcAttr = &syscall.SysProcAttr{
		PseudoConsole: syscall.Handle(pty.handle),
	}

	if err := cmd.Start(); err != nil {
		t.Errorf("start cmd failed: %v", err)
	}

	var outBuf bytes.Buffer
	go func() { pty.inPipe.Write([]byte("exit\r\n")) }()
	go io.Copy(&outBuf, pty.outPipe)

	_ = cmd.Wait()

	if got, want := outBuf.String(), "Microsoft Windows"; !strings.Contains(got, want) {
		t.Errorf("cmd output: want %q, got %q", want, got)
	}
}

var (
	kernel32 = syscall.MustLoadDLL("kernel32.dll")

	procCreatePseudoConsole = kernel32.MustFindProc("CreatePseudoConsole")
	procClosePseudoConsole  = kernel32.MustFindProc("ClosePseudoConsole")
)

type conPty struct {
	handle  syscall.Handle
	inPipe  *os.File
	outPipe *os.File
}

func (c *conPty) Close() error {
	closePseudoConsole(c.handle)
	if err := c.inPipe.Close(); err != nil {
		return err
	}
	return c.outPipe.Close()
}

// See https://learn.microsoft.com/en-us/windows/console/creating-a-pseudoconsole-session
func newConPty() (*conPty, error) {
	inputRead, inputWrite, err := os.Pipe()
	if err != nil {
		return nil, err
	}

	outputRead, outputWrite, err := os.Pipe()
	if err != nil {
		return nil, err
	}

	var handle syscall.Handle
	coord := uint32(25<<16) | 80 // 80x25 screen buffer
	err = createPseudoConsole(coord, syscall.Handle(inputRead.Fd()), syscall.Handle(outputWrite.Fd()), 0, &handle)
	if err != nil {
		return nil, err
	}

	if err := outputWrite.Close(); err != nil {
		return nil, err
	}
	if err := inputRead.Close(); err != nil {
		return nil, err
	}

	return &conPty{
		handle:  handle,
		inPipe:  inputWrite,
		outPipe: outputRead,
	}, nil
}

func createPseudoConsole(size uint32, in syscall.Handle, out syscall.Handle, flags uint32, pconsole *syscall.Handle) (hr error) {
	r0, _, _ := syscall.Syscall6(procCreatePseudoConsole.Addr(), 5, uintptr(size), uintptr(in), uintptr(out), uintptr(flags), uintptr(unsafe.Pointer(pconsole)), 0)
	if r0 != 0 {
		hr = syscall.Errno(r0)
	}
	return
}

func closePseudoConsole(console syscall.Handle) {
	syscall.Syscall(procClosePseudoConsole.Addr(), 1, uintptr(console), 0, 0)
	return
}
