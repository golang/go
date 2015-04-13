// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This program can be used as go_darwin_arm_exec by the Go tool.
// It executes binaries on an iOS device using the XCode toolchain
// and the ios-deploy program: https://github.com/phonegap/ios-deploy
//
// This script supports an extra flag, -lldb, that pauses execution
// just before the main program begins and allows the user to control
// the remote lldb session. This flag is appended to the end of the
// script's arguments and is not passed through to the underlying
// binary.
//
// This script requires that three environment variables be set:
// 	GOIOS_DEV_ID: The codesigning developer id or certificate identifier
// 	GOIOS_APP_ID: The provisioning app id prefix. Must support wildcard app ids.
// 	GOIOS_TEAM_ID: The team id that owns the app id prefix.
// $GOROOT/misc/ios contains a script, detect.go, that attempts to autodetect these.
package main

import (
	"bytes"
	"errors"
	"flag"
	"fmt"
	"go/build"
	"io"
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

var errRetry = errors.New("failed to start test harness (retry attempted)")

var tmpdir string

var (
	devID  string
	appID  string
	teamID string
)

func main() {
	log.SetFlags(0)
	log.SetPrefix("go_darwin_arm_exec: ")
	if debug {
		log.Println(strings.Join(os.Args, " "))
	}
	if len(os.Args) < 2 {
		log.Fatal("usage: go_darwin_arm_exec a.out")
	}

	devID = getenv("GOIOS_DEV_ID")
	appID = getenv("GOIOS_APP_ID")
	teamID = getenv("GOIOS_TEAM_ID")

	var err error
	tmpdir, err = ioutil.TempDir("", "go_darwin_arm_exec_")
	if err != nil {
		log.Fatal(err)
	}

	// Approximately 1 in a 100 binaries fail to start. If it happens,
	// try again. These failures happen for several reasons beyond
	// our control, but all of them are safe to retry as they happen
	// before lldb encounters the initial getwd breakpoint. As we
	// know the tests haven't started, we are not hiding flaky tests
	// with this retry.
	for i := 0; i < 5; i++ {
		if i > 0 {
			fmt.Fprintln(os.Stderr, "start timeout, trying again")
		}
		err = run(os.Args[1], os.Args[2:])
		if err == nil || err != errRetry {
			break
		}
	}
	if !debug {
		os.RemoveAll(tmpdir)
	}
	if err != nil {
		fmt.Fprintf(os.Stderr, "go_darwin_arm_exec: %v\n", err)
		os.Exit(1)
	}
}

func getenv(envvar string) string {
	s := os.Getenv(envvar)
	if s == "" {
		log.Fatalf("%s not set\nrun $GOROOT/misc/ios/detect.go to attempt to autodetect", s)
	}
	return s
}

func run(bin string, args []string) (err error) {
	appdir := filepath.Join(tmpdir, "gotest.app")
	os.RemoveAll(appdir)
	if err := os.MkdirAll(appdir, 0755); err != nil {
		return err
	}

	if err := cp(filepath.Join(appdir, "gotest"), bin); err != nil {
		return err
	}

	entitlementsPath := filepath.Join(tmpdir, "Entitlements.plist")
	if err := ioutil.WriteFile(entitlementsPath, []byte(entitlementsPlist()), 0744); err != nil {
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
		"-s", devID,
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

	oldwd, err := os.Getwd()
	if err != nil {
		return err
	}
	if err := os.Chdir(filepath.Join(appdir, "..")); err != nil {
		return err
	}
	defer os.Chdir(oldwd)

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

	var opts options
	opts, args = parseArgs(args)

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
	if opts.lldb {
		mw := io.MultiWriter(w, os.Stderr)
		cmd.Stdout = mw
		cmd.Stderr = mw
	} else {
		cmd.Stdout = w
		cmd.Stderr = w // everything of interest is on stderr
	}
	cmd.Stdin = lldbr

	if err := cmd.Start(); err != nil {
		return fmt.Errorf("ios-deploy failed to start: %v", err)
	}

	// Manage the -test.timeout here, outside of the test. There is a lot
	// of moving parts in an iOS test harness (notably lldb) that can
	// swallow useful stdio or cause its own ruckus.
	var timedout chan struct{}
	if opts.timeout > 1*time.Second {
		timedout = make(chan struct{})
		time.AfterFunc(opts.timeout-1*time.Second, func() {
			close(timedout)
		})
	}

	exited := make(chan error)
	go func() {
		exited <- cmd.Wait()
	}()

	waitFor := func(stage, str string, timeout time.Duration) error {
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
		case i := <-w.find(str, timeout):
			if i < 0 {
				log.Printf("timed out on stage %q, retrying", stage)
				return errRetry
			}
			w.clearTo(i + len(str))
			return nil
		}
	}
	do := func(cmd string) {
		fmt.Fprintln(lldb, cmd)
		if err := waitFor(fmt.Sprintf("prompt after %q", cmd), "(lldb)", 0); err != nil {
			panic(waitPanic{err})
		}
	}

	// Wait for installation and connection.
	if err := waitFor("ios-deploy before run", "(lldb)     connect\r\nProcess 0 connected\r\n", 0); err != nil {
		// Retry if we see a rare and longstanding ios-deploy bug.
		// https://github.com/phonegap/ios-deploy/issues/11
		//	Assertion failed: (AMDeviceStartService(device, CFSTR("com.apple.debugserver"), &gdbfd, NULL) == 0)
		log.Printf("%v, retrying", err)
		return errRetry
	}

	// Script LLDB. Oh dear.
	do(`process handle SIGHUP  --stop false --pass true --notify false`)
	do(`process handle SIGPIPE --stop false --pass true --notify false`)
	do(`process handle SIGUSR1 --stop false --pass true --notify false`)
	do(`process handle SIGSEGV --stop false --pass true --notify false`) // does not work
	do(`process handle SIGBUS  --stop false --pass true --notify false`) // does not work

	if opts.lldb {
		_, err := io.Copy(lldb, os.Stdin)
		if err != io.EOF {
			return err
		}
		return nil
	}

	do(`breakpoint set -n getwd`) // in runtime/cgo/gcc_darwin_arm.go

	fmt.Fprintln(lldb, `run`)
	if err := waitFor("br getwd", "stop reason = breakpoint", 20*time.Second); err != nil {
		// At this point we see several flaky errors from the iOS
		// build infrastructure. The most common is never reaching
		// the breakpoint, which we catch with a timeout. Very
		// occasionally lldb can produce errors like:
		//
		//	Breakpoint 1: no locations (pending).
		//	WARNING:  Unable to resolve breakpoint to any actual locations.
		//
		// As no actual test code has been executed by this point,
		// we treat all errors as recoverable.
		if err != errRetry {
			log.Printf("%v, retrying", err)
			err = errRetry
		}
		return err
	}
	if err := waitFor("br getwd prompt", "(lldb)", 0); err != nil {
		return err
	}

	// Move the current working directory into the faux gopath.
	if pkgpath != "src" {
		do(`breakpoint delete 1`)
		do(`expr char* $mem = (char*)malloc(512)`)
		do(`expr $mem = (char*)getwd($mem, 512)`)
		do(`expr $mem = (char*)strcat($mem, "/` + pkgpath + `")`)
		do(`call (void)chdir($mem)`)
	}

	// Watch for SIGSEGV. Ideally lldb would never break on SIGSEGV.
	// http://golang.org/issue/10043
	go func() {
		<-w.find("stop reason = EXC_BAD_ACCESS", 0)
		// cannot use do here, as the defer/recover is not available
		// on this goroutine.
		fmt.Fprintln(lldb, `bt`)
		waitFor("finish backtrace", "(lldb)", 0)
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

	findTxt   []byte   // search buffer on each Write
	findCh    chan int // report find position
	findAfter *time.Timer
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
			if w.findAfter != nil {
				w.findAfter.Stop()
				w.findAfter = nil
			}
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

// find returns a channel that will have exactly one byte index sent
// to it when the text str appears in the buffer. If the text does not
// appear before timeout, -1 is sent.
//
// A timeout of zero means no timeout.
func (w *bufWriter) find(str string, timeout time.Duration) <-chan int {
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
		if timeout > 0 {
			w.findAfter = time.AfterFunc(timeout, func() {
				w.mu.Lock()
				defer w.mu.Unlock()
				if w.findCh == ch {
					w.findTxt = nil
					w.findCh = nil
					w.findAfter = nil
					ch <- -1
					close(ch)
				}
			})
		}
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

type options struct {
	timeout time.Duration
	lldb    bool
}

func parseArgs(binArgs []string) (opts options, remainingArgs []string) {
	var flagArgs []string
	for _, arg := range binArgs {
		if strings.Contains(arg, "-test.timeout") {
			flagArgs = append(flagArgs, arg)
		}
		if strings.Contains(arg, "-lldb") {
			flagArgs = append(flagArgs, arg)
			continue
		}
		remainingArgs = append(remainingArgs, arg)
	}
	f := flag.NewFlagSet("", flag.ContinueOnError)
	f.DurationVar(&opts.timeout, "test.timeout", 0, "")
	f.BoolVar(&opts.lldb, "lldb", false, "")
	f.Parse(flagArgs)
	return opts, remainingArgs

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

func entitlementsPlist() string {
	return `<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
	<key>keychain-access-groups</key>
	<array><string>` + teamID + `.golang.gotest</string></array>
	<key>get-task-allow</key>
	<true/>
	<key>application-identifier</key>
	<string>` + teamID + `.golang.gotest</string>
	<key>com.apple.developer.team-identifier</key>
	<string>` + teamID + `</string>
</dict>
</plist>`
}

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
