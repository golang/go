// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package user

import (
	"crypto/rand"
	"encoding/base64"
	"encoding/binary"
	"errors"
	"fmt"
	"internal/syscall/windows"
	"internal/testenv"
	"os"
	"os/exec"
	"runtime"
	"slices"
	"strconv"
	"strings"
	"syscall"
	"testing"
	"unicode"
	"unicode/utf8"
	"unsafe"
)

// addUserAccount creates a local user account.
// It returns the name and password of the new account.
// Multiple programs or goroutines calling addUserAccount simultaneously will not choose the same directory.
func addUserAccount(t *testing.T) (name, password string) {
	t.TempDir()
	pattern := t.Name()
	// Windows limits the user name to 20 characters,
	// leave space for a 4 digits random suffix.
	const maxNameLen, suffixLen = 20, 4
	pattern = pattern[:min(len(pattern), maxNameLen-suffixLen)]
	// Drop unusual characters from the account name.
	mapper := func(r rune) rune {
		if r < utf8.RuneSelf {
			if '0' <= r && r <= '9' ||
				'a' <= r && r <= 'z' ||
				'A' <= r && r <= 'Z' {
				return r
			}
		} else if unicode.IsLetter(r) || unicode.IsNumber(r) {
			return r
		}
		return -1
	}
	pattern = strings.Map(mapper, pattern)

	// Generate a long random password.
	var pwd [33]byte
	rand.Read(pwd[:])
	// Add special chars to ensure it satisfies password requirements.
	password = base64.StdEncoding.EncodeToString(pwd[:]) + "_-As@!%*(1)4#2"
	password16, err := syscall.UTF16PtrFromString(password)
	if err != nil {
		t.Fatal(err)
	}

	try := 0
	for {
		// Calculate a random suffix to append to the user name.
		var suffix [2]byte
		rand.Read(suffix[:])
		suffixStr := strconv.FormatUint(uint64(binary.LittleEndian.Uint16(suffix[:])), 10)
		name := pattern + suffixStr[:min(len(suffixStr), suffixLen)]
		name16, err := syscall.UTF16PtrFromString(name)
		if err != nil {
			t.Fatal(err)
		}
		// Create user.
		userInfo := windows.UserInfo1{
			Name:     name16,
			Password: password16,
			Priv:     windows.USER_PRIV_USER,
		}
		err = windows.NetUserAdd(nil, 1, (*byte)(unsafe.Pointer(&userInfo)), nil)
		if errors.Is(err, syscall.ERROR_ACCESS_DENIED) {
			t.Skip("skipping test; don't have permission to create user")
		}
		// If the user already exists, try again with a different name.
		if errors.Is(err, windows.NERR_UserExists) {
			if try++; try < 1000 {
				t.Log("user already exists, trying again with a different name")
				continue
			}
		}
		if err != nil {
			t.Fatalf("NetUserAdd failed: %v", err)
		}
		// Delete the user when the test is done.
		t.Cleanup(func() {
			if err := windows.NetUserDel(nil, name16); err != nil {
				if !errors.Is(err, windows.NERR_UserNotFound) {
					t.Fatal(err)
				}
			}
		})
		return name, password
	}
}

// windowsTestAccount creates a test user and returns a token for that user.
// If the user already exists, it will be deleted and recreated.
// The caller is responsible for closing the token.
func windowsTestAccount(t *testing.T) (syscall.Token, *User) {
	if testenv.Builder() == "" {
		// Adding and deleting users requires special permissions.
		// Even if we have them, we don't want to create users on
		// on dev machines, as they may not be cleaned up.
		// See https://dev.go/issue/70396.
		t.Skip("skipping non-hermetic test outside of Go builders")
	}
	name, password := addUserAccount(t)
	name16, err := syscall.UTF16PtrFromString(name)
	if err != nil {
		t.Fatal(err)
	}
	pwd16, err := syscall.UTF16PtrFromString(password)
	if err != nil {
		t.Fatal(err)
	}
	domain, err := syscall.UTF16PtrFromString(".")
	if err != nil {
		t.Fatal(err)
	}
	const LOGON32_PROVIDER_DEFAULT = 0
	const LOGON32_LOGON_INTERACTIVE = 2
	var token syscall.Token
	if err = windows.LogonUser(name16, domain, pwd16, LOGON32_LOGON_INTERACTIVE, LOGON32_PROVIDER_DEFAULT, &token); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		token.Close()
	})
	usr, err := Lookup(name)
	if err != nil {
		t.Fatal(err)
	}
	return token, usr
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
	token, _ := windowsTestAccount(t)

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

func TestCurrentNetapi32(t *testing.T) {
	if os.Getenv("GO_WANT_HELPER_PROCESS") == "1" {
		// Test that Current does not load netapi32.dll.
		// First call Current.
		Current()

		// Then check if netapi32.dll is loaded.
		netapi32, err := syscall.UTF16PtrFromString("netapi32.dll")
		if err != nil {
			fmt.Fprintf(os.Stderr, "error: %s\n", err.Error())
			os.Exit(9)
			return
		}
		mod, _ := windows.GetModuleHandle(netapi32)
		if mod != 0 {
			fmt.Fprintf(os.Stderr, "netapi32.dll is loaded\n")
			os.Exit(9)
			return
		}
		os.Exit(0)
		return
	}
	exe := testenv.Executable(t)
	cmd := testenv.CleanCmdEnv(exec.Command(exe, "-test.run=^TestCurrentNetapi32$"))
	cmd.Env = append(cmd.Env, "GO_WANT_HELPER_PROCESS=1")
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("%v\n%s", err, out)
	}
}

func TestGroupIdsTestUser(t *testing.T) {
	// Create a test user and log in as that user.
	_, user := windowsTestAccount(t)

	gids, err := user.GroupIds()
	if err != nil {
		t.Fatal(err)
	}

	if err != nil {
		t.Fatalf("%+v.GroupIds(): %v", user, err)
	}
	if !slices.Contains(gids, user.Gid) {
		t.Errorf("%+v.GroupIds() = %v; does not contain user GID %s", user, gids, user.Gid)
	}
}

var serviceAccounts = []struct {
	sid  string
	name string
}{
	{"S-1-5-18", "NT AUTHORITY\\SYSTEM"},
	{"S-1-5-19", "NT AUTHORITY\\LOCAL SERVICE"},
	{"S-1-5-20", "NT AUTHORITY\\NETWORK SERVICE"},
}

func TestLookupServiceAccount(t *testing.T) {
	t.Parallel()
	for _, tt := range serviceAccounts {
		u, err := Lookup(tt.name)
		if err != nil {
			t.Errorf("Lookup(%q): %v", tt.name, err)
			continue
		}
		if u.Uid != tt.sid {
			t.Errorf("unexpected uid for %q; got %q, want %q", u.Name, u.Uid, tt.sid)
		}
	}
}

func TestLookupIdServiceAccount(t *testing.T) {
	t.Parallel()
	for _, tt := range serviceAccounts {
		u, err := LookupId(tt.sid)
		if err != nil {
			t.Errorf("LookupId(%q): %v", tt.sid, err)
			continue
		}
		if u.Gid != tt.sid {
			t.Errorf("unexpected gid for %q; got %q, want %q", u.Name, u.Gid, tt.sid)
		}
		if u.Username != tt.name {
			t.Errorf("unexpected user name for %q; got %q, want %q", u.Gid, u.Username, tt.name)
		}
	}
}

func TestLookupGroupServiceAccount(t *testing.T) {
	t.Parallel()
	for _, tt := range serviceAccounts {
		u, err := LookupGroup(tt.name)
		if err != nil {
			t.Errorf("LookupGroup(%q): %v", tt.name, err)
			continue
		}
		if u.Gid != tt.sid {
			t.Errorf("unexpected gid for %q; got %q, want %q", u.Name, u.Gid, tt.sid)
		}
	}
}

func TestLookupGroupIdServiceAccount(t *testing.T) {
	t.Parallel()
	for _, tt := range serviceAccounts {
		u, err := LookupGroupId(tt.sid)
		if err != nil {
			t.Errorf("LookupGroupId(%q): %v", tt.sid, err)
			continue
		}
		if u.Gid != tt.sid {
			t.Errorf("unexpected gid for %q; got %q, want %q", u.Name, u.Gid, tt.sid)
		}
	}
}
