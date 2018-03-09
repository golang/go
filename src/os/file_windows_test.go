// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package os

import (
	"fmt"
	"internal/syscall/windows/registry"
	"io/ioutil"
	"os"
	"path/filepath"
	"syscall"
	"testing"
)

// see https://github.com/golang/go/issues/22874
// os: Symlink creation should work on Windows without elevation

func TestSymlink(t *testing.T) {
	// is developer mode active?
	// the expected result depends on it
	devMode, _ := isDeveloperModeActive()

	t.Logf("Windows developer mode active: %v\n", devMode)

	// create dummy file to symlink
	dummyFile := filepath.Join(os.TempDir(), "issue22874.test")

	err := ioutil.WriteFile(dummyFile, []byte(""), 0644)

	if err != nil {
		t.Fatalf("Failed to create dummy file: %v", err)
	}

	defer os.Remove(dummyFile)

	// create the symlink
	linkFile := fmt.Sprintf("%v.link", dummyFile)

	err = os.Symlink(dummyFile, linkFile)

	if err != nil {
		// only the ERROR_PRIVILEGE_NOT_HELD error is allowed
		if x, ok := err.(*os.LinkError); ok {
			if xx, ok := x.Err.(syscall.Errno); ok {

				if xx == syscall.ERROR_PRIVILEGE_NOT_HELD {
					// is developer mode active?
					if devMode {
						t.Fatalf("Windows developer mode is active, but creating symlink failed with ERROR_PRIVILEGE_NOT_HELD anyway: %v", err)
					}

					// developer mode is disabled, and the error is expected
					fmt.Printf("Success: Creating symlink failed with expected ERROR_PRIVILEGE_NOT_HELD error\n")

					return nil
				}
			}
		}

		t.Fatalf("Failed to create symlink: %v", err)
	}

	// remove the link. don't care for any errors
	os.Remove(linkFile)

	t.Logf("Success: Creating symlink succeeded\n")

	return nil
}

func isDeveloperModeActive() (bool, error) {
	key, err := registry.OpenKey(registry.LOCAL_MACHINE, "SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\AppModelUnlock", registry.READ)

	if err != nil {
		return false, err
	}

	val, _, err := key.GetIntegerValue("AllowDevelopmentWithoutDevLicense")

	if err != nil {
		return false, err
	}

	return val != 0, nil
}
