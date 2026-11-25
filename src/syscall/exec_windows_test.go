// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syscall_test

import (
	"bytes"
	"crypto/rand"
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

	"internal/syscall/windows"
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

func createJobHandleToKillProcesses(t *testing.T) syscall.Handle {
	h, err := windows.CreateJobObject(nil, nil)
	if err != nil {
		t.Fatalf("CreateJobObject failed: %v", err)
	}

	// Configure the job to kill all processes when the handle closes
	var info windows.JOBOBJECT_EXTENDED_LIMIT_INFORMATION
	info.BasicLimitInformation.LimitFlags = windows.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
	err = windows.SetInformationJobObject(
		h,
		windows.JobObjectExtendedLimitInformation,
		uintptr(unsafe.Pointer(&info)),
		uint32(unsafe.Sizeof(info)),
	)
	if err != nil {
		t.Fatalf("SetInformationJobObject failed: %v", err)
	}
	return h
}

func createTestJobChildCmd(t *testing.T, jobs []syscall.Handle) (*exec.Cmd, *bytes.Buffer) {
	var output bytes.Buffer
	cmd := exec.Command(testenv.Executable(t), "-test.run=^TestJobHandle$")
	cmd.Env = append(os.Environ(), "GO_WANT_HELPER_PROCESS=1")
	cmd.Stdout = &output
	cmd.Stderr = &output
	cmd.SysProcAttr = &syscall.SysProcAttr{
		Jobs: jobs,
	}
	return cmd, &output
}

func TestJobHandle(t *testing.T) {
	if os.Getenv("GO_WANT_HELPER_PROCESS") == "1" {
		// in child process
		fmt.Printf("Hello world")
		time.Sleep(time.Hour)
		os.Exit(0)
	}

	t.Run("Delete process", func(t *testing.T) {
		jobHandle := createJobHandleToKillProcesses(t)
		defer syscall.Close(jobHandle)

		// start child
		child, output := createTestJobChildCmd(t, []syscall.Handle{jobHandle})
		err := child.Start()
		if err != nil {
			t.Fatal(err)
		}

		// kill child
		err = syscall.CloseHandle(jobHandle)
		if err != nil {
			t.Fatalf("CloseHandle failed: %v", err)
		}

		err = child.Wait()
		if err != nil {
			t.Fatalf("child.Wait failed: %v: %v", err, output.String())
		}
	})

	tests := []struct {
		name    string
		jobs    []syscall.Handle
		wanterr string
	}{
		{"jobs is nil", nil, ""},
		{"jobs is []syscall.Handle{}", []syscall.Handle{}, ""},
		{"jobs is []syscall.Handle{0}", []syscall.Handle{0}, "The handle is invalid."},
		{"jobs is []syscall.Handle{syscall.InvalidHandle}", []syscall.Handle{syscall.InvalidHandle}, "The handle is invalid."},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cmd, _ := createTestJobChildCmd(t, tt.jobs)
			err := cmd.Start()
			if err != nil && !strings.Contains(err.Error(), tt.wanterr) {
				t.Errorf("unexpected cmd.Start error: want %q, got %q", tt.wanterr, err)
			}
			if err != nil {
				// child failed to start - just get out
				return
			}

			err = cmd.Process.Kill()
			if err != nil {
				t.Errorf("failed to kill process: %v", err)
			}
			err = cmd.Wait()
			if err != nil {
				if ee, ok := err.(*exec.ExitError); !ok || ee.ProcessState.ExitCode() != 1 {
					t.Errorf("Wait error = %v; want exit status 1", err)
				}
			}
		})
	}
}

func TestPseudoConsoleProcess(t *testing.T) {
	testenv.MustHaveExec(t)

	kernel32, err := syscall.LoadLibrary("kernel32.dll")
	if err != nil {
		t.Fatal(err)
	}
	defer syscall.FreeLibrary(kernel32)

	createPseudoConsole, err := syscall.GetProcAddress(kernel32, "CreatePseudoConsole")
	if err != nil {
		t.Fatal(err)
	}
	closePseudoConsole, err := syscall.GetProcAddress(kernel32, "ClosePseudoConsole")
	if err != nil {
		t.Fatal(err)
	}

	inputRead, inputWrite, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	defer inputRead.Close()
	defer inputWrite.Close()

	outputRead, outputWrite, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	defer outputRead.Close()
	defer outputWrite.Close()

	var console syscall.Handle
	const size = 80 | 25<<16 // COORD for an 80-column, 25-row console.
	hr, _, _ := syscall.SyscallN(createPseudoConsole,
		size, inputRead.Fd(), outputWrite.Fd(), 0, uintptr(unsafe.Pointer(&console)),
	)
	if hr != 0 {
		t.Fatalf("CreatePseudoConsole failed: HRESULT %#x", uint32(hr))
	}

	const message = "go-pseudoconsole-test"
	var output bytes.Buffer
	readDone := make(chan error, 1)
	go func() {
		_, err := io.Copy(&output, outputRead)
		readDone <- err
	}()
	closeConsole := func() error {
		if console == 0 {
			return nil
		}
		outputWrite.Close()
		// ClosePseudoConsole can wait for its output to drain.
		syscall.SyscallN(closePseudoConsole, uintptr(console))
		console = 0
		return <-readDone
	}
	defer func() {
		if err := closeConsole(); err != nil {
			t.Errorf("read console output: %v", err)
		}
	}()

	if err := inputRead.Close(); err != nil {
		t.Fatal(err)
	}
	if err := outputWrite.Close(); err != nil {
		t.Fatal(err)
	}

	cmd := testenv.Command(t, "cmd.exe", "/d", "/c", "echo", message)
	cmd.SysProcAttr = &syscall.SysProcAttr{Console: console}
	if err := cmd.Run(); err != nil {
		t.Fatalf("run child: %v", err)
	}
	if err := closeConsole(); err != nil {
		t.Fatalf("read console output: %v", err)
	}
	if got := output.String(); !strings.Contains(got, message) {
		t.Errorf("console output = %q, want it to contain %q", got, message)
	}
}

func TestAppContainer(t *testing.T) {
	testenv.MustHaveExec(t)

	// AppContainer profiles are registered persistently in the current Windows
	// user's profile (not scoped to the process or temp directory), so use a
	// unique name to avoid colliding with concurrent test runs or profiles
	// leaked by previously interrupted runs.
	var appContainerSid *syscall.SID
	name := "gotest-" + rand.Text()
	appContainerName := syscall.StringToUTF16Ptr(name)
	displayName := syscall.StringToUTF16Ptr("Go test AppContainer")
	description := syscall.StringToUTF16Ptr("An isolated Go test sandbox application profile.")
	err := windows.CreateAppContainerProfile(appContainerName, displayName, description, nil, 0, &appContainerSid)
	if err != nil {
		t.Fatalf("CreateAppContainerProfile(%q) failed: %v", name, err)
	}
	defer func() {
		if appContainerSid != nil {
			if windows.FreeSid(appContainerSid) != nil {
				t.Error("FreeSid failed")
			}
		}
		// DeleteAppContainerProfile can fail transiently right after the child
		// process exits while Windows finishes tearing down the container's
		// storage directory and registry state (similar to issue #50051).
		const arbitraryTimeout = 2 * time.Second
		var (
			start     time.Time
			nextSleep = 1 * time.Millisecond
		)
		for {
			err := windows.DeleteAppContainerProfile(appContainerName)
			if err == nil {
				break
			}
			if start.IsZero() {
				start = time.Now()
			}
			elapsed := time.Since(start)
			if elapsed >= arbitraryTimeout {
				t.Errorf("DeleteAppContainerProfile(%q) failed: %v", name, err)
				break
			}
			time.Sleep(min(nextSleep, arbitraryTimeout-elapsed))
			nextSleep *= 2
		}
	}()

	message := "Hello from AppContainer"
	cmd := testenv.Command(t, `C:\Windows\System32\cmd.exe`, "/c", "echo", message)
	cmd.SysProcAttr = &syscall.SysProcAttr{
		AppContainer: &syscall.AppContainer{
			SID: appContainerSid,
		},
	}
	output, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("child failed: %v: %v", err, string(output))
	}
	if got, want := string(output), syscall.EscapeArg(message)+"\r\n"; got != want {
		t.Errorf("child output invalid: got=%q want=%q", got, want)
	}
}
