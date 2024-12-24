// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This program can be used as go_ios_$GOARCH_exec by the Go tool. It executes
// binaries on the iOS Simulator using the XCode toolchain.
package main

import (
	"fmt"
	"go/build"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"syscall"
)

const debug = false

var tmpdir string

var (
	devID    string
	appID    string
	teamID   string
	bundleID string
	deviceID string
)

// lock is a file lock to serialize iOS runs. It is global to avoid the
// garbage collector finalizing it, closing the file and releasing the
// lock prematurely.
var lock *os.File

func main() {
	log.SetFlags(0)
	log.SetPrefix("go_ios_exec: ")
	if debug {
		log.Println(strings.Join(os.Args, " "))
	}
	if len(os.Args) < 2 {
		log.Fatal("usage: go_ios_exec a.out")
	}

	// For compatibility with the old builders, use a fallback bundle ID
	bundleID = "golang.gotest"

	exitCode, err := runMain()
	if err != nil {
		log.Fatalf("%v\n", err)
	}
	os.Exit(exitCode)
}

func runMain() (int, error) {
	var err error
	tmpdir, err = os.MkdirTemp("", "go_ios_exec_")
	if err != nil {
		return 1, err
	}
	if !debug {
		defer os.RemoveAll(tmpdir)
	}

	appdir := filepath.Join(tmpdir, "gotest.app")
	os.RemoveAll(appdir)

	if err := assembleApp(appdir, os.Args[1]); err != nil {
		return 1, err
	}

	// This wrapper uses complicated machinery to run iOS binaries. It
	// works, but only when running one binary at a time.
	// Use a file lock to make sure only one wrapper is running at a time.
	//
	// The lock file is never deleted, to avoid concurrent locks on distinct
	// files with the same path.
	lockName := filepath.Join(os.TempDir(), "go_ios_exec-"+deviceID+".lock")
	lock, err = os.OpenFile(lockName, os.O_CREATE|os.O_RDONLY, 0666)
	if err != nil {
		return 1, err
	}
	if err := syscall.Flock(int(lock.Fd()), syscall.LOCK_EX); err != nil {
		return 1, err
	}

	err = runOnSimulator(appdir)
	if err != nil {
		return 1, err
	}
	return 0, nil
}

func runOnSimulator(appdir string) error {
	if err := installSimulator(appdir); err != nil {
		return err
	}

	return runSimulator(appdir, bundleID, os.Args[2:])
}

func assembleApp(appdir, bin string) error {
	if err := os.MkdirAll(appdir, 0755); err != nil {
		return err
	}

	if err := cp(filepath.Join(appdir, "gotest"), bin); err != nil {
		return err
	}

	pkgpath, err := copyLocalData(appdir)
	if err != nil {
		return err
	}

	entitlementsPath := filepath.Join(tmpdir, "Entitlements.plist")
	if err := os.WriteFile(entitlementsPath, []byte(entitlementsPlist()), 0744); err != nil {
		return err
	}
	if err := os.WriteFile(filepath.Join(appdir, "Info.plist"), []byte(infoPlist(pkgpath)), 0744); err != nil {
		return err
	}
	if err := os.WriteFile(filepath.Join(appdir, "ResourceRules.plist"), []byte(resourceRules), 0744); err != nil {
		return err
	}
	return nil
}

func installSimulator(appdir string) error {
	cmd := exec.Command(
		"xcrun", "simctl", "install",
		"booted", // Install to the booted simulator.
		appdir,
	)
	if out, err := cmd.CombinedOutput(); err != nil {
		os.Stderr.Write(out)
		return fmt.Errorf("xcrun simctl install booted %q: %v", appdir, err)
	}
	return nil
}

func runSimulator(appdir, bundleID string, args []string) error {
	xcrunArgs := []string{"simctl", "spawn",
		"booted",
		appdir + "/gotest",
	}
	xcrunArgs = append(xcrunArgs, args...)
	cmd := exec.Command("xcrun", xcrunArgs...)
	cmd.Stdout, cmd.Stderr = os.Stdout, os.Stderr
	err := cmd.Run()
	if err != nil {
		return fmt.Errorf("xcrun simctl launch booted %q: %v", bundleID, err)
	}

	return nil
}

func copyLocalDir(dst, src string) error {
	if err := os.Mkdir(dst, 0755); err != nil {
		return err
	}

	d, err := os.Open(src)
	if err != nil {
		return err
	}
	defer d.Close()
	fi, err := d.Readdir(-1)
	if err != nil {
		return err
	}

	for _, f := range fi {
		if f.IsDir() {
			if f.Name() == "testdata" {
				if err := cp(dst, filepath.Join(src, f.Name())); err != nil {
					return err
				}
			}
			continue
		}
		if err := cp(dst, filepath.Join(src, f.Name())); err != nil {
			return err
		}
	}
	return nil
}

func cp(dst, src string) error {
	out, err := exec.Command("cp", "-a", src, dst).CombinedOutput()
	if err != nil {
		os.Stderr.Write(out)
	}
	return err
}

func copyLocalData(dstbase string) (pkgpath string, err error) {
	cwd, err := os.Getwd()
	if err != nil {
		return "", err
	}

	finalPkgpath, underGoRoot, err := subdir()
	if err != nil {
		return "", err
	}
	cwd = strings.TrimSuffix(cwd, finalPkgpath)

	// Copy all immediate files and testdata directories between
	// the package being tested and the source root.
	pkgpath = ""
	for _, element := range strings.Split(finalPkgpath, string(filepath.Separator)) {
		if debug {
			log.Printf("copying %s", pkgpath)
		}
		pkgpath = filepath.Join(pkgpath, element)
		dst := filepath.Join(dstbase, pkgpath)
		src := filepath.Join(cwd, pkgpath)
		if err := copyLocalDir(dst, src); err != nil {
			return "", err
		}
	}

	if underGoRoot {
		// Copy timezone file.
		//
		// Typical apps have the zoneinfo.zip in the root of their app bundle,
		// read by the time package as the working directory at initialization.
		// As we move the working directory to the GOROOT pkg directory, we
		// install the zoneinfo.zip file in the pkgpath.
		err := cp(
			filepath.Join(dstbase, pkgpath),
			filepath.Join(cwd, "lib", "time", "zoneinfo.zip"),
		)
		if err != nil {
			return "", err
		}
		// Copy src/runtime/textflag.h for (at least) Test386EndToEnd in
		// cmd/asm/internal/asm.
		runtimePath := filepath.Join(dstbase, "src", "runtime")
		if err := os.MkdirAll(runtimePath, 0755); err != nil {
			return "", err
		}
		err = cp(
			filepath.Join(runtimePath, "textflag.h"),
			filepath.Join(cwd, "src", "runtime", "textflag.h"),
		)
		if err != nil {
			return "", err
		}
	}

	return finalPkgpath, nil
}

// subdir determines the package based on the current working directory,
// and returns the path to the package source relative to $GOROOT (or $GOPATH).
func subdir() (pkgpath string, underGoRoot bool, err error) {
	cwd, err := os.Getwd()
	if err != nil {
		return "", false, err
	}
	cwd, err = filepath.EvalSymlinks(cwd)
	if err != nil {
		log.Fatal(err)
	}
	goroot, err := filepath.EvalSymlinks(runtime.GOROOT())
	if err != nil {
		return "", false, err
	}
	if strings.HasPrefix(cwd, goroot) {
		subdir, err := filepath.Rel(goroot, cwd)
		if err != nil {
			return "", false, err
		}
		return subdir, true, nil
	}

	for _, p := range filepath.SplitList(build.Default.GOPATH) {
		pabs, err := filepath.EvalSymlinks(p)
		if err != nil {
			return "", false, err
		}
		if !strings.HasPrefix(cwd, pabs) {
			continue
		}
		subdir, err := filepath.Rel(pabs, cwd)
		if err == nil {
			return subdir, false, nil
		}
	}
	return "", false, fmt.Errorf(
		"working directory %q is not in either GOROOT(%q) or GOPATH(%q)",
		cwd,
		runtime.GOROOT(),
		build.Default.GOPATH,
	)
}

func infoPlist(pkgpath string) string {
	return `<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
<key>CFBundleName</key><string>golang.gotest</string>
<key>CFBundleSupportedPlatforms</key><array><string>iPhoneOS</string></array>
<key>CFBundleExecutable</key><string>gotest</string>
<key>CFBundleVersion</key><string>1.0</string>
<key>CFBundleShortVersionString</key><string>1.0</string>
<key>CFBundleIdentifier</key><string>` + bundleID + `</string>
<key>CFBundleResourceSpecification</key><string>ResourceRules.plist</string>
<key>LSRequiresIPhoneOS</key><true/>
<key>CFBundleDisplayName</key><string>gotest</string>
<key>GoExecWrapperWorkingDirectory</key><string>` + pkgpath + `</string>
</dict>
</plist>
`
}

func entitlementsPlist() string {
	return `<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
	<key>keychain-access-groups</key>
	<array><string>` + appID + `</string></array>
	<key>get-task-allow</key>
	<true/>
	<key>application-identifier</key>
	<string>` + appID + `</string>
	<key>com.apple.developer.team-identifier</key>
	<string>` + teamID + `</string>
</dict>
</plist>
`
}

const resourceRules = `<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
	<key>rules</key>
	<dict>
		<key>.*</key>
		<true/>
		<key>Info.plist</key>
		<dict>
			<key>omit</key>
			<true/>
			<key>weight</key>
			<integer>10</integer>
		</dict>
		<key>ResourceRules.plist</key>
		<dict>
			<key>omit</key>
			<true/>
			<key>weight</key>
			<integer>100</integer>
		</dict>
	</dict>
</dict>
</plist>
`
