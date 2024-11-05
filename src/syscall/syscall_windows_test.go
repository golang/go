// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syscall_test

import (
	"fmt"
	"internal/testenv"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"syscall"
	"testing"
)

func TestOpen_Dir(t *testing.T) {
	dir := t.TempDir()

	h, err := syscall.Open(dir, syscall.O_RDONLY, 0)
	if err != nil {
		t.Fatalf("Open failed: %v", err)
	}
	syscall.CloseHandle(h)
	h, err = syscall.Open(dir, syscall.O_RDONLY|syscall.O_TRUNC, 0)
	if err == nil {
		t.Error("Open should have failed")
	} else {
		syscall.CloseHandle(h)
	}
	h, err = syscall.Open(dir, syscall.O_RDONLY|syscall.O_CREAT, 0)
	if err == nil {
		t.Error("Open should have failed")
	} else {
		syscall.CloseHandle(h)
	}
}

func TestComputerName(t *testing.T) {
	name, err := syscall.ComputerName()
	if err != nil {
		t.Fatalf("ComputerName failed: %v", err)
	}
	if len(name) == 0 {
		t.Error("ComputerName returned empty string")
	}
}

func TestWin32finddata(t *testing.T) {
	dir := t.TempDir()

	path := filepath.Join(dir, "long_name.and_extension")
	f, err := os.Create(path)
	if err != nil {
		t.Fatalf("failed to create %v: %v", path, err)
	}
	f.Close()

	type X struct {
		fd  syscall.Win32finddata
		got byte
		pad [10]byte // to protect ourselves

	}
	var want byte = 2 // it is unlikely to have this character in the filename
	x := X{got: want}

	pathp, _ := syscall.UTF16PtrFromString(path)
	h, err := syscall.FindFirstFile(pathp, &(x.fd))
	if err != nil {
		t.Fatalf("FindFirstFile failed: %v", err)
	}
	err = syscall.FindClose(h)
	if err != nil {
		t.Fatalf("FindClose failed: %v", err)
	}

	if x.got != want {
		t.Fatalf("memory corruption: want=%d got=%d", want, x.got)
	}
}

func abort(funcname string, err error) {
	panic(funcname + " failed: " + err.Error())
}

func ExampleLoadLibrary() {
	h, err := syscall.LoadLibrary("kernel32.dll")
	if err != nil {
		abort("LoadLibrary", err)
	}
	defer syscall.FreeLibrary(h)
	proc, err := syscall.GetProcAddress(h, "GetVersion")
	if err != nil {
		abort("GetProcAddress", err)
	}
	r, _, _ := syscall.Syscall(uintptr(proc), 0, 0, 0, 0)
	major := byte(r)
	minor := uint8(r >> 8)
	build := uint16(r >> 16)
	print("windows version ", major, ".", minor, " (Build ", build, ")\n")
}

func TestTOKEN_ALL_ACCESS(t *testing.T) {
	if syscall.TOKEN_ALL_ACCESS != 0xF01FF {
		t.Errorf("TOKEN_ALL_ACCESS = %x, want 0xF01FF", syscall.TOKEN_ALL_ACCESS)
	}
}

func TestStdioAreInheritable(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	testenv.MustHaveCGO(t)
	testenv.MustHaveExecPath(t, "gcc")

	tmpdir := t.TempDir()

	// build go dll
	const dlltext = `
package main

import "C"
import (
	"fmt"
)

//export HelloWorld
func HelloWorld() {
	fmt.Println("Hello World")
}

func main() {}
`
	dllsrc := filepath.Join(tmpdir, "helloworld.go")
	err := os.WriteFile(dllsrc, []byte(dlltext), 0644)
	if err != nil {
		t.Fatal(err)
	}
	dll := filepath.Join(tmpdir, "helloworld.dll")
	cmd := exec.Command(testenv.GoToolPath(t), "build", "-o", dll, "-buildmode", "c-shared", dllsrc)
	out, err := testenv.CleanCmdEnv(cmd).CombinedOutput()
	if err != nil {
		t.Fatalf("failed to build go library: %s\n%s", err, out)
	}

	// build c exe
	const exetext = `
#include <stdlib.h>
#include <windows.h>
int main(int argc, char *argv[])
{
	system("hostname");
	((void(*)(void))GetProcAddress(LoadLibraryA(%q), "HelloWorld"))();
	system("hostname");
	return 0;
}
`
	exe := filepath.Join(tmpdir, "helloworld.exe")
	cmd = exec.Command("gcc", "-o", exe, "-xc", "-")
	cmd.Stdin = strings.NewReader(fmt.Sprintf(exetext, dll))
	out, err = testenv.CleanCmdEnv(cmd).CombinedOutput()
	if err != nil {
		t.Fatalf("failed to build c executable: %s\n%s", err, out)
	}
	out, err = exec.Command(exe).Output()
	if err != nil {
		t.Fatalf("c program execution failed: %v: %v", err, string(out))
	}

	hostname, err := os.Hostname()
	if err != nil {
		t.Fatal(err)
	}

	have := strings.ReplaceAll(string(out), "\n", "")
	have = strings.ReplaceAll(have, "\r", "")
	want := fmt.Sprintf("%sHello World%s", hostname, hostname)
	if have != want {
		t.Fatalf("c program output is wrong: got %q, want %q", have, want)
	}
}

func TestGetwd_DoesNotPanicWhenPathIsLong(t *testing.T) {
	// Regression test for https://github.com/golang/go/issues/60051.

	// The length of a filename is also limited, so we can't reproduce the
	// crash by creating a single directory with a very long name; we need two
	// layers.
	a200 := strings.Repeat("a", 200)
	dirname := filepath.Join(t.TempDir(), a200, a200)

	err := os.MkdirAll(dirname, 0o700)
	if err != nil {
		t.Skipf("MkdirAll failed: %v", err)
	}
	err = os.Chdir(dirname)
	if err != nil {
		t.Skipf("Chdir failed: %v", err)
	}
	// Change out of the temporary directory so that we don't inhibit its
	// removal during test cleanup.
	defer os.Chdir(`\`)

	syscall.Getwd()
}

func TestGetStartupInfo(t *testing.T) {
	var si syscall.StartupInfo
	err := syscall.GetStartupInfo(&si)
	if err != nil {
		// see https://go.dev/issue/31316
		t.Fatalf("GetStartupInfo: got error %v, want nil", err)
	}
}

func TestSyscallAllocations(t *testing.T) {
	testenv.SkipIfOptimizationOff(t)

	// Test that syscall.SyscallN arguments do not escape.
	// The function used (in this case GetVersion) doesn't matter
	// as long as it is always available and doesn't panic.
	h, err := syscall.LoadLibrary("kernel32.dll")
	if err != nil {
		t.Fatal(err)
	}
	defer syscall.FreeLibrary(h)
	proc, err := syscall.GetProcAddress(h, "GetVersion")
	if err != nil {
		t.Fatal(err)
	}

	testAllocs := func(t *testing.T, name string, fn func() error) {
		t.Run(name, func(t *testing.T) {
			n := int(testing.AllocsPerRun(10, func() {
				if err := fn(); err != nil {
					t.Fatalf("%s: %v", name, err)
				}
			}))
			if n > 0 {
				t.Errorf("allocs = %d, want 0", n)
			}
		})
	}

	testAllocs(t, "SyscallN", func() error {
		r0, _, e1 := syscall.SyscallN(proc, 0, 0, 0)
		if r0 == 0 {
			return syscall.Errno(e1)
		}
		return nil
	})
	testAllocs(t, "Syscall", func() error {
		r0, _, e1 := syscall.Syscall(proc, 3, 0, 0, 0)
		if r0 == 0 {
			return syscall.Errno(e1)
		}
		return nil
	})
}

func FuzzUTF16FromString(f *testing.F) {
	f.Add("hi")           // ASCII
	f.Add("â")            // latin1
	f.Add("ねこ")           // plane 0
	f.Add("😃")            // extra Plane 0
	f.Add("\x90")         // invalid byte
	f.Add("\xe3\x81")     // truncated
	f.Add("\xe3\xc1\x81") // invalid middle byte

	f.Fuzz(func(t *testing.T, tst string) {
		res, err := syscall.UTF16FromString(tst)
		if err != nil {
			if strings.Contains(tst, "\x00") {
				t.Skipf("input %q contains a NUL byte", tst)
			}
			t.Fatalf("UTF16FromString(%q): %v", tst, err)
		}
		t.Logf("UTF16FromString(%q) = %04x", tst, res)

		if len(res) < 1 || res[len(res)-1] != 0 {
			t.Fatalf("missing NUL terminator")
		}
		if len(res) > len(tst)+1 {
			t.Fatalf("len(%04x) > len(%q)+1", res, tst)
		}
	})
}
