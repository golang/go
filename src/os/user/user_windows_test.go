// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package user

import (
	"crypto/rand"
	"encoding/base64"
	"errors"
	"internal/syscall/windows"
	"runtime"
	"strconv"
	"syscall"
	"testing"
	"unsafe"
)

// windowsTestAcount creates a test user and returns a token for that user.
// If the user already exists, it will be deleted and recreated.
// The caller is responsible for closing the token.
func windowsTestAcount(t *testing.T) syscall.Token {
	var password [33]byte
	rand.Read(password[:])
	// Add special chars to ensure it satisfies password requirements.
	pwd := base64.StdEncoding.EncodeToString(password[:]) + "_-As@!%*(1)4#2"
	name, err := syscall.UTF16PtrFromString("GoStdTestUser01")
	if err != nil {
		t.Fatal(err)
	}
	pwd16, err := syscall.UTF16PtrFromString(pwd)
	if err != nil {
		t.Fatal(err)
	}
	userInfo := windows.UserInfo1{
		Name:     name,
		Password: pwd16,
		Priv:     windows.USER_PRIV_USER,
	}
	// Create user.
	err = windows.NetUserAdd(nil, 1, (*byte)(unsafe.Pointer(&userInfo)), nil)
	if errors.Is(err, syscall.ERROR_ACCESS_DENIED) {
		t.Skip("skipping test; don't have permission to create user")
	}
	if errors.Is(err, windows.NERR_UserExists) {
		// User already exists, delete and recreate.
		if err = windows.NetUserDel(nil, name); err != nil {
			t.Fatal(err)
		}
		if err = windows.NetUserAdd(nil, 1, (*byte)(unsafe.Pointer(&userInfo)), nil); err != nil {
			t.Fatal(err)
		}
	} else if err != nil {
		t.Fatal(err)
	}
	domain, err := syscall.UTF16PtrFromString(".")
	if err != nil {
		t.Fatal(err)
	}
	const LOGON32_PROVIDER_DEFAULT = 0
	const LOGON32_LOGON_INTERACTIVE = 2
	var token syscall.Token
	if err = windows.LogonUser(name, domain, pwd16, LOGON32_LOGON_INTERACTIVE, LOGON32_PROVIDER_DEFAULT, &token); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		token.Close()
		if err = windows.NetUserDel(nil, name); err != nil {
			if !errors.Is(err, windows.NERR_UserNotFound) {
				t.Fatal(err)
			}
		}
	})
	return token
}

func TestImpersonatedSelf(t *testing.T) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	want, err := current()
	if err != nil {
		t.Fatal(err)
	}

	levels := []uint32{
		windows.SecurityAnonymous,
		windows.SecurityIdentification,
		windows.SecurityImpersonation,
		windows.SecurityDelegation,
	}
	for _, level := range levels {
		t.Run(strconv.Itoa(int(level)), func(t *testing.T) {
			if err = windows.ImpersonateSelf(level); err != nil {
				t.Fatal(err)
			}
			defer windows.RevertToSelf()

			got, err := current()
			if level == windows.SecurityAnonymous {
				// We can't get the process token when using an anonymous token,
				// so we expect an error here.
				if err == nil {
					t.Fatal("expected error")
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			compare(t, want, got)
		})
	}
}

func TestImpersonated(t *testing.T) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	want, err := current()
	if err != nil {
		t.Fatal(err)
	}

	// Create a test user and log in as that user.
	token := windowsTestAcount(t)

	// Impersonate the test user.
	if err = windows.ImpersonateLoggedOnUser(token); err != nil {
		t.Fatal(err)
	}
	defer func() {
		err = windows.RevertToSelf()
		if err != nil {
			// If we can't revert to self, we can't continue testing.
			panic(err)
		}
	}()

	got, err := current()
	if err != nil {
		t.Fatal(err)
	}
	compare(t, want, got)
}
