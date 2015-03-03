// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This program can be used as go_darwin_arm_exec by the Go tool.
// It executes binaries on an iOS device using the XCode toolchain
// and the ios-deploy program: https://github.com/phonegap/ios-deploy
package main

import (
	"bytes"
	"errors"
	"flag"
	"fmt"
	"go/build"
	"io/ioutil"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"time"
)

const debug = false

func main() {
	log.SetFlags(0)
	log.SetPrefix("go_darwin_arm_exec: ")
	if debug {
		log.Println(strings.Join(os.Args, " "))
	}
	if len(os.Args) < 2 {
		log.Fatal("usage: go_darwin_arm_exec a.out")
	}

	if err := run(os.Args[1], os.Args[2:]); err != nil {
		fmt.Fprintf(os.Stderr, "go_darwin_arm_exec: %v\n", err)
		os.Exit(1)
	}
}

func run(bin string, args []string) (err error) {
	type waitPanic struct {
		err error
	}
	defer func() {
		if r := recover(); r != nil {
			if w, ok := r.(waitPanic); ok {
				err = w.err
				return
			}
			panic(r)
		}
	}()

	defer exec.Command("killall", "ios-deploy").Run() // cleanup

	exec.Command("killall", "ios-deploy").Run()

	tmpdir, err := ioutil.TempDir("", "go_darwin_arm_exec_")
	if err != nil {
		log.Fatal(err)
	}
	if !debug {
		defer os.RemoveAll(tmpdir)
	}

	appdir := filepath.Join(tmpdir, "gotest.app")
	if err := os.MkdirAll(appdir, 0755); err != nil {
		return err
	}

	if err := cp(filepath.Join(appdir, "gotest"), bin); err != nil {
		return err
	}

	entitlementsPath := filepath.Join(tmpdir, "Entitlements.plist")
	if err := ioutil.WriteFile(entitlementsPath, []byte(entitlementsPlist), 0744); err != nil {
		return err
	}
	if err := ioutil.WriteFile(filepath.Join(appdir, "Info.plist"), []byte(infoPlist), 0744); err != nil {
		return err
	}
	if err := ioutil.WriteFile(filepath.Join(appdir, "ResourceRules.plist"), []byte(resourceRules), 0744); err != nil {
		return err
	}

	pkgpath, err := copyLocalData(appdir)
	if err != nil {
		return err
	}

	cmd := exec.Command(
		"codesign",
		"-f",
		"-s", "E8BMC3FE2Z", // certificate associated with golang.org
		"--entitlements", entitlementsPath,
		appdir,
	)
	if debug {
		log.Println(strings.Join(cmd.Args, " "))
	}
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("codesign: %v", err)
	}

	if err := os.Chdir(tmpdir); err != nil {
		return err
	}

	// ios-deploy invokes lldb to give us a shell session with the app.
	cmd = exec.Command(
		// lldb tries to be clever with terminals.
		// So we wrap it in script(1) and be clever
		// right back at it.
		"script",
		"-q", "-t", "0",
		"/dev/null",

		"ios-deploy",
		"--debug",
		"-u",
		"-r",
		"-n",
		`--args=`+strings.Join(args, " ")+``,
		"--bundle", appdir,
	)
	if debug {
		log.Println(strings.Join(cmd.Args, " "))
	}

	lldbr, lldb, err := os.Pipe()
	if err != nil {
		return err
	}
	w := new(bufWriter)
	cmd.Stdout = w
	cmd.Stderr = w // everything of interest is on stderr
	cmd.Stdin = lldbr

	if err := cmd.Start(); err != nil {
		return fmt.Errorf("ios-deploy failed to start: %v", err)
	}

	// Manage the -test.timeout here, outside of the test. There is a lot
	// of moving parts in an iOS test harness (notably lldb) that can
	// swallow useful stdio or cause its own ruckus.
	var timedout chan struct{}
	if t := parseTimeout(args); t > 1*time.Second {
		timedout = make(chan struct{})
		time.AfterFunc(t-1*time.Second, func() {
			close(timedout)
		})
	}

	exited := make(chan error)
	go func() {
		exited <- cmd.Wait()
	}()

	waitFor := func(stage, str string) error {
		select {
		case <-timedout:
			w.printBuf()
			if p := cmd.Process; p != nil {
				p.Kill()
			}
			return fmt.Errorf("timeout (stage %s)", stage)
		case err := <-exited:
			w.printBuf()
			return fmt.Errorf("failed (stage %s): %v", stage, err)
		case i := <-w.find(str):
			w.clearTo(i + len(str))
			return nil
		}
	}
	do := func(cmd string) {
		fmt.Fprintln(lldb, cmd)
		if err := waitFor(fmt.Sprintf("prompt after %q", cmd), "(lldb)"); err != nil {
			panic(waitPanic{err})
		}
	}

	// Wait for installation and connection.
	if err := waitFor("ios-deploy before run", "(lldb)     connect\r\nProcess 0 connected\r\n"); err != nil {
		return err
	}

	// Script LLDB. Oh dear.
	do(`process handle SIGHUP  --stop false --pass true --notify false`)
	do(`process handle SIGPIPE --stop false --pass true --notify false`)
	do(`process handle SIGUSR1 --stop false --pass true --notify false`)
	do(`process handle SIGSEGV --stop false --pass true --notify false`) // does not work
	do(`process handle SIGBUS  --stop false --pass true --notify false`) // does not work

	do(`breakpoint set -n getwd`) // in runtime/cgo/gcc_darwin_arm.go

	fmt.Fprintln(lldb, `run`)
	if err := waitFor("br getwd", "stop reason = breakpoint"); err != nil {
		return err
	}
	if err := waitFor("br getwd prompt", "(lldb)"); err != nil {
		return err
	}

	// Move the current working directory into the faux gopath.
	do(`breakpoint delete 1`)
	do(`expr char* $mem = (char*)malloc(512)`)
	do(`expr $mem = (char*)getwd($mem, 512)`)
	do(`expr $mem = (char*)strcat($mem, "/` + pkgpath + `")`)
	do(`call (void)chdir($mem)`)

	// Watch for SIGSEGV. Ideally lldb would never break on SIGSEGV.
	// http://golang.org/issue/10043
	go func() {
		<-w.find("stop reason = EXC_BAD_ACCESS")
		// cannot use do here, as the defer/recover is not available
		// on this goroutine.
		fmt.Fprintln(lldb, `bt`)
		waitFor("finish backtrace", "(lldb)")
		w.printBuf()
		if p := cmd.Process; p != nil {
			p.Kill()
		}
	}()

	// Run the tests.
	w.trimSuffix("(lldb) ")
	fmt.Fprintln(lldb, `process continue`)

	// Wait for the test to complete.
	select {
	case <-timedout:
		w.printBuf()
		if p := cmd.Process; p != nil {
			p.Kill()
		}
		return errors.New("timeout running tests")
	case err := <-exited:
		// The returned lldb error code is usually non-zero.
		// We check for test success by scanning for the final
		// PASS returned by the test harness, assuming the worst
		// in its absence.
		if w.isPass() {
			err = nil
		} else if err == nil {
			err = errors.New("test failure")
		}
		w.printBuf()
		return err
	}
}

type bufWriter struct {
	mu     sync.Mutex
	buf    []byte
	suffix []byte // remove from each Write

	findTxt []byte   // search buffer on each Write
	findCh  chan int // report find position
}

func (w *bufWriter) Write(in []byte) (n int, err error) {
	w.mu.Lock()
	defer w.mu.Unlock()

	n = len(in)
	in = bytes.TrimSuffix(in, w.suffix)

	w.buf = append(w.buf, in...)

	if len(w.findTxt) > 0 {
		if i := bytes.Index(w.buf, w.findTxt); i >= 0 {
			w.findCh <- i
			close(w.findCh)
			w.findTxt = nil
			w.findCh = nil
		}
	}
	return n, nil
}

func (w *bufWriter) trimSuffix(p string) {
	w.mu.Lock()
	defer w.mu.Unlock()
	w.suffix = []byte(p)
}

func (w *bufWriter) printBuf() {
	w.mu.Lock()
	defer w.mu.Unlock()
	fmt.Fprintf(os.Stderr, "%s", w.buf)
	w.buf = nil
}

func (w *bufWriter) clearTo(i int) {
	w.mu.Lock()
	defer w.mu.Unlock()
	if debug {
		fmt.Fprintf(os.Stderr, "--- go_darwin_arm_exec clear ---\n%s\n--- go_darwin_arm_exec clear ---\n", w.buf[:i])
	}
	w.buf = w.buf[i:]
}

func (w *bufWriter) find(str string) <-chan int {
	w.mu.Lock()
	defer w.mu.Unlock()
	if len(w.findTxt) > 0 {
		panic(fmt.Sprintf("find(%s): already trying to find %s", str, w.findTxt))
	}
	txt := []byte(str)
	ch := make(chan int, 1)
	if i := bytes.Index(w.buf, txt); i >= 0 {
		ch <- i
		close(ch)
	} else {
		w.findTxt = txt
		w.findCh = ch
	}
	return ch
}

func (w *bufWriter) isPass() bool {
	w.mu.Lock()
	defer w.mu.Unlock()

	// The final stdio of lldb is non-deterministic, so we
	// scan the whole buffer.
	//
	// Just to make things fun, lldb sometimes translates \n
	// into \r\n.
	return bytes.Contains(w.buf, []byte("\nPASS\n")) || bytes.Contains(w.buf, []byte("\nPASS\r"))
}

func parseTimeout(testArgs []string) (timeout time.Duration) {
	var args []string
	for _, arg := range testArgs {
		if strings.Contains(arg, "test.timeout") {
			args = append(args, arg)
		}
	}
	f := flag.NewFlagSet("", flag.ContinueOnError)
	f.DurationVar(&timeout, "test.timeout", 0, "")
	f.Parse(args)
	if debug {
		log.Printf("parseTimeout of %s, got %s", args, timeout)
	}
	return timeout
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

	// Copy timezone file.
	//
	// Typical apps have the zoneinfo.zip in the root of their app bundle,
	// read by the time package as the working directory at initialization.
	// As we move the working directory to the GOROOT pkg directory, we
	// install the zoneinfo.zip file in the pkgpath.
	if underGoRoot {
		err := cp(
			filepath.Join(dstbase, pkgpath),
			filepath.Join(cwd, "lib", "time", "zoneinfo.zip"),
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
	if root := runtime.GOROOT(); strings.HasPrefix(cwd, root) {
		subdir, err := filepath.Rel(root, cwd)
		if err != nil {
			return "", false, err
		}
		return subdir, true, nil
	}

	for _, p := range filepath.SplitList(build.Default.GOPATH) {
		if !strings.HasPrefix(cwd, p) {
			continue
		}
		subdir, err := filepath.Rel(p, cwd)
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

const infoPlist = `<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
<key>CFBundleName</key><string>golang.gotest</string>
<key>CFBundleSupportedPlatforms</key><array><string>iPhoneOS</string></array>
<key>CFBundleExecutable</key><string>gotest</string>
<key>CFBundleVersion</key><string>1.0</string>
<key>CFBundleIdentifier</key><string>golang.gotest</string>
<key>CFBundleResourceSpecification</key><string>ResourceRules.plist</string>
<key>LSRequiresIPhoneOS</key><true/>
<key>CFBundleDisplayName</key><string>gotest</string>
</dict>
</plist>
`

const devID = `YE84DJ86AZ`

const entitlementsPlist = `<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
	<key>keychain-access-groups</key>
	<array><string>` + devID + `.golang.gotest</string></array>
	<key>get-task-allow</key>
	<true/>
	<key>application-identifier</key>
	<string>` + devID + `.golang.gotest</string>
	<key>com.apple.developer.team-identifier</key>
	<string>` + devID + `</string>
</dict>
</plist>`

const resourceRules = `<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
        <key>rules</key>
        <dict>
                <key>.*</key><true/>
		<key>Info.plist</key> 
		<dict>
			<key>omit</key> <true/>
			<key>weight</key> <real>10</real>
		</dict>
		<key>ResourceRules.plist</key>
		<dict>
			<key>omit</key> <true/>
			<key>weight</key> <real>100</real>
		</dict>
	</dict>
</dict>
</plist>
`
