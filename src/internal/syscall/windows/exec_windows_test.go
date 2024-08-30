// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build windows

package windows_test

import (
	"fmt"
	"internal/syscall/windows"
	"os"
	"os/exec"
	"syscall"
	"testing"
	"unsafe"
)

func TestRunAtLowIntegrity(t *testing.T) {
	if os.Getenv("GO_WANT_HELPER_PROCESS") == "1" {
		wil, err := getProcessIntegrityLevel()
		if err != nil {
			fmt.Fprintf(os.Stderr, "error: %s\n", err.Error())
			os.Exit(9)
			return
		}
		fmt.Printf("%s", wil)
		os.Exit(0)
		return
	}

	cmd := exec.Command(os.Args[0], "-test.run=^TestRunAtLowIntegrity$", "--")
	cmd.Env = []string{"GO_WANT_HELPER_PROCESS=1"}

	token, err := getIntegrityLevelToken(sidWilLow)
	if err != nil {
		t.Fatal(err)
	}
	defer token.Close()

	cmd.SysProcAttr = &syscall.SysProcAttr{
		Token: token,
	}

	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatal(err)
	}

	if string(out) != sidWilLow {
		t.Fatalf("Child process did not run as low integrity level: %s", string(out))
	}
}

const (
	sidWilLow = `S-1-16-4096`
)

func getProcessIntegrityLevel() (string, error) {
	procToken, err := syscall.OpenCurrentProcessToken()
	if err != nil {
		return "", err
	}
	defer procToken.Close()

	p, err := tokenGetInfo(procToken, syscall.TokenIntegrityLevel, 64)
	if err != nil {
		return "", err
	}

	tml := (*windows.TOKEN_MANDATORY_LABEL)(p)

	sid := (*syscall.SID)(unsafe.Pointer(tml.Label.Sid))

	return sid.String()
}

func tokenGetInfo(t syscall.Token, class uint32, initSize int) (unsafe.Pointer, error) {
	n := uint32(initSize)
	for {
		b := make([]byte, n)
		e := syscall.GetTokenInformation(t, class, &b[0], uint32(len(b)), &n)
		if e == nil {
			return unsafe.Pointer(&b[0]), nil
		}
		if e != syscall.ERROR_INSUFFICIENT_BUFFER {
			return nil, e
		}
		if n <= uint32(len(b)) {
			return nil, e
		}
	}
}

func getIntegrityLevelToken(wns string) (syscall.Token, error) {
	var procToken, token syscall.Token

	proc, err := syscall.GetCurrentProcess()
	if err != nil {
		return 0, err
	}
	defer syscall.CloseHandle(proc)

	err = syscall.OpenProcessToken(proc,
		syscall.TOKEN_DUPLICATE|
			syscall.TOKEN_ADJUST_DEFAULT|
			syscall.TOKEN_QUERY|
			syscall.TOKEN_ASSIGN_PRIMARY,
		&procToken)
	if err != nil {
		return 0, err
	}
	defer procToken.Close()

	sid, err := syscall.StringToSid(wns)
	if err != nil {
		return 0, err
	}

	tml := &windows.TOKEN_MANDATORY_LABEL{}
	tml.Label.Attributes = windows.SE_GROUP_INTEGRITY
	tml.Label.Sid = sid

	err = windows.DuplicateTokenEx(procToken, 0, nil, windows.SecurityImpersonation,
		windows.TokenPrimary, &token)
	if err != nil {
		return 0, err
	}

	err = windows.SetTokenInformation(token,
		syscall.TokenIntegrityLevel,
		uintptr(unsafe.Pointer(tml)),
		tml.Size())
	if err != nil {
		token.Close()
		return 0, err
	}
	return token, nil
}
