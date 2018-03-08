package main

import (
	"fmt"
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"syscall"
)

// https://github.com/golang/go/issues/22874
// os: Symlink creation should work on Windows without elevation

func main() {
	// run on windows, only
	if runtime.GOOS != "windows" {
		fmt.Println("Skipping test on non-Windows system")
		return
	}

	// run test
	err := test()

	if err != nil {
		panic(err)
	}
}

func test() error {
	// is developer mode active?
	// the expected result depends on it
	devMode, _ := isDeveloperModeActive()

	fmt.Printf("Windows developer mode active: %v\n", devMode)

	// create dummy file to symlink
	dummyFile := filepath.Join(os.TempDir(), "issue22874.test")

	err := ioutil.WriteFile(dummyFile, []byte(""), 0644)

	if err != nil {
		return fmt.Errorf("Failed to create dummy file: %v", err)
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
						return fmt.Errorf("Windows developer mode is active, but creating symlink failed with ERROR_PRIVILEGE_NOT_HELD anyway: %v", err)
					}

					// developer mode is disabled, and the error is expected
					fmt.Printf("Success: Creating symlink failed with expected ERROR_PRIVILEGE_NOT_HELD error\n")

					return nil
				}
			}
		}

		return fmt.Errorf("Failed to create symlink: %v", err)
	}

	// remove the link. don't care for any errors
	os.Remove(linkFile)

	fmt.Printf("Success: Creating symlink succeeded\n")

	return nil
}

func isDeveloperModeActive() (bool, error) {
	result, err := exec.Command("reg.exe", "query", "HKLM\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\AppModelUnlock", "/v", "AllowDevelopmentWithoutDevLicense").Output()

	if err != nil {
		return false, err
	}

	return strings.Contains(string(result), "0x1"), nil
}
