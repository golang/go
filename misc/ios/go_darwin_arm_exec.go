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
	"syscall"
	"time"
)

const debug = false

var errRetry = errors.New("failed to start test harness (retry attempted)")

var tmpdir string

var (
	devID    string
	appID    string
	teamID   string
	bundleID string
)

// lock is a file lock to serialize iOS runs. It is global to avoid the
// garbage collector finalizing it, closing the file and releasing the
// lock prematurely.
var lock *os.File

func main() {
	log.SetFlags(0)
	log.SetPrefix("go_darwin_arm_exec: ")
	if debug {
		log.Println(strings.Join(os.Args, " "))
	}
	if len(os.Args) < 2 {
		log.Fatal("usage: go_darwin_arm_exec a.out")
	}

	// e.g. B393DDEB490947F5A463FD074299B6C0AXXXXXXX
	devID = getenv("GOIOS_DEV_ID")

	// e.g. Z8B3JBXXXX.org.golang.sample, Z8B3JBXXXX prefix is available at
	// https://developer.apple.com/membercenter/index.action#accountSummary as Team ID.
	appID = getenv("GOIOS_APP_ID")

	// e.g. Z8B3JBXXXX, available at
	// https://developer.apple.com/membercenter/index.action#accountSummary as Team ID.
	teamID = getenv("GOIOS_TEAM_ID")

	parts := strings.SplitN(appID, ".", 2)
	// For compatibility with the old builders, use a fallback bundle ID
	bundleID = "golang.gotest"
	if len(parts) == 2 {
		bundleID = parts[1]
	}

	var err error
	tmpdir, err = ioutil.TempDir("", "go_darwin_arm_exec_")
	if err != nil {
		log.Fatal(err)
	}

	// This wrapper uses complicated machinery to run iOS binaries. It
	// works, but only when running one binary at a time.
	// Use a file lock to make sure only one wrapper is running at a time.
	//
	// The lock file is never deleted, to avoid concurrent locks on distinct
	// files with the same path.
	lockName := filepath.Join(os.TempDir(), "go_darwin_arm_exec.lock")
	lock, err = os.OpenFile(lockName, os.O_CREATE|os.O_RDONLY, 0666)
	if err != nil {
		log.Fatal(err)
	}
	if err := syscall.Flock(int(lock.Fd()), syscall.LOCK_EX); err != nil {
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
		log.Fatalf("%s not set\nrun $GOROOT/misc/ios/detect.go to attempt to autodetect", envvar)
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

	pkgpath, err := copyLocalData(appdir)
	if err != nil {
		return err
	}

	entitlementsPath := filepath.Join(tmpdir, "Entitlements.plist")
	if err := ioutil.WriteFile(entitlementsPath, []byte(entitlementsPlist()), 0744); err != nil {
		return err
	}
	if err := ioutil.WriteFile(filepath.Join(appdir, "Info.plist"), []byte(infoPlist(pkgpath)), 0744); err != nil {
		return err
	}
	if err := ioutil.WriteFile(filepath.Join(appdir, "ResourceRules.plist"), []byte(resourceRules), 0744); err != nil {
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

	// Setting up lldb is flaky. The test binary itself runs when
	// started is set to true. Everything before that is considered
	// part of the setup and is retried.
	started := false
	defer func() {
		if r := recover(); r != nil {
			if w, ok := r.(waitPanic); ok {
				err = w.err
				if !started {
					fmt.Printf("lldb setup error: %v\n", err)
					err = errRetry
				}
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
	s, err := newSession(appdir, args, opts)
	if err != nil {
		return err
	}
	defer func() {
		b := s.out.Bytes()
		if err == nil && !debug {
			i := bytes.Index(b, []byte("(lldb) process continue"))
			if i > 0 {
				b = b[i:]
			}
		}
		os.Stdout.Write(b)
	}()

	// Script LLDB. Oh dear.
	s.do(`process handle SIGHUP  --stop false --pass true --notify false`)
	s.do(`process handle SIGPIPE --stop false --pass true --notify false`)
	s.do(`process handle SIGUSR1 --stop false --pass true --notify false`)
	s.do(`process handle SIGCONT --stop false --pass true --notify false`)
	s.do(`process handle SIGSEGV --stop false --pass true --notify false`) // does not work
	s.do(`process handle SIGBUS  --stop false --pass true --notify false`) // does not work

	if opts.lldb {
		_, err := io.Copy(s.in, os.Stdin)
		if err != io.EOF {
			return err
		}
		return nil
	}

	started = true

	s.doCmd("run", "stop reason = signal SIGINT", 20*time.Second)

	startTestsLen := s.out.Len()
	fmt.Fprintln(s.in, `process continue`)

	passed := func(out *buf) bool {
		// Just to make things fun, lldb sometimes translates \n into \r\n.
		return s.out.LastIndex([]byte("\nPASS\n")) > startTestsLen ||
			s.out.LastIndex([]byte("\nPASS\r")) > startTestsLen ||
			s.out.LastIndex([]byte("\n(lldb) PASS\n")) > startTestsLen ||
			s.out.LastIndex([]byte("\n(lldb) PASS\r")) > startTestsLen ||
			s.out.LastIndex([]byte("exited with status = 0 (0x00000000) \n")) > startTestsLen ||
			s.out.LastIndex([]byte("exited with status = 0 (0x00000000) \r")) > startTestsLen
	}
	err = s.wait("test completion", passed, opts.timeout)
	if passed(s.out) {
		// The returned lldb error code is usually non-zero.
		// We check for test success by scanning for the final
		// PASS returned by the test harness, assuming the worst
		// in its absence.
		return nil
	}
	return err
}

type lldbSession struct {
	cmd      *exec.Cmd
	in       *os.File
	out      *buf
	timedout chan struct{}
	exited   chan error
}

func newSession(appdir string, args []string, opts options) (*lldbSession, error) {
	lldbr, in, err := os.Pipe()
	if err != nil {
		return nil, err
	}
	s := &lldbSession{
		in:     in,
		out:    new(buf),
		exited: make(chan error),
	}

	iosdPath, err := exec.LookPath("ios-deploy")
	if err != nil {
		return nil, err
	}
	s.cmd = exec.Command(
		// lldb tries to be clever with terminals.
		// So we wrap it in script(1) and be clever
		// right back at it.
		"script",
		"-q", "-t", "0",
		"/dev/null",

		iosdPath,
		"--debug",
		"-u",
		"-r",
		"-n",
		`--args=`+strings.Join(args, " ")+``,
		"--bundle", appdir,
	)
	if debug {
		log.Println(strings.Join(s.cmd.Args, " "))
	}

	var out io.Writer = s.out
	if opts.lldb {
		out = io.MultiWriter(out, os.Stderr)
	}
	s.cmd.Stdout = out
	s.cmd.Stderr = out // everything of interest is on stderr
	s.cmd.Stdin = lldbr

	if err := s.cmd.Start(); err != nil {
		return nil, fmt.Errorf("ios-deploy failed to start: %v", err)
	}

	// Manage the -test.timeout here, outside of the test. There is a lot
	// of moving parts in an iOS test harness (notably lldb) that can
	// swallow useful stdio or cause its own ruckus.
	if opts.timeout > 1*time.Second {
		s.timedout = make(chan struct{})
		time.AfterFunc(opts.timeout-1*time.Second, func() {
			close(s.timedout)
		})
	}

	go func() {
		s.exited <- s.cmd.Wait()
	}()

	cond := func(out *buf) bool {
		i0 := s.out.LastIndex([]byte("(lldb)"))
		i1 := s.out.LastIndex([]byte("fruitstrap"))
		i2 := s.out.LastIndex([]byte(" connect"))
		return i0 > 0 && i1 > 0 && i2 > 0
	}
	if err := s.wait("lldb start", cond, 15*time.Second); err != nil {
		panic(waitPanic{err})
	}
	return s, nil
}

func (s *lldbSession) do(cmd string) { s.doCmd(cmd, "(lldb)", 0) }

func (s *lldbSession) doCmd(cmd string, waitFor string, extraTimeout time.Duration) {
	startLen := s.out.Len()
	fmt.Fprintln(s.in, cmd)
	cond := func(out *buf) bool {
		i := s.out.LastIndex([]byte(waitFor))
		return i > startLen
	}
	if err := s.wait(fmt.Sprintf("running cmd %q", cmd), cond, extraTimeout); err != nil {
		panic(waitPanic{err})
	}
}

func (s *lldbSession) wait(reason string, cond func(out *buf) bool, extraTimeout time.Duration) error {
	doTimeout := 2*time.Second + extraTimeout
	doTimedout := time.After(doTimeout)
	for {
		select {
		case <-s.timedout:
			if p := s.cmd.Process; p != nil {
				p.Kill()
			}
			return fmt.Errorf("test timeout (%s)", reason)
		case <-doTimedout:
			return fmt.Errorf("command timeout (%s for %v)", reason, doTimeout)
		case err := <-s.exited:
			return fmt.Errorf("exited (%s: %v)", reason, err)
		default:
			if cond(s.out) {
				return nil
			}
			time.Sleep(20 * time.Millisecond)
		}
	}
}

type buf struct {
	mu  sync.Mutex
	buf []byte
}

func (w *buf) Write(in []byte) (n int, err error) {
	w.mu.Lock()
	defer w.mu.Unlock()
	w.buf = append(w.buf, in...)
	return len(in), nil
}

func (w *buf) LastIndex(sep []byte) int {
	w.mu.Lock()
	defer w.mu.Unlock()
	return bytes.LastIndex(w.buf, sep)
}

func (w *buf) Bytes() []byte {
	w.mu.Lock()
	defer w.mu.Unlock()

	b := make([]byte, len(w.buf))
	copy(b, w.buf)
	return b
}

func (w *buf) Len() int {
	w.mu.Lock()
	defer w.mu.Unlock()
	return len(w.buf)
}

type waitPanic struct {
	err error
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
	f.DurationVar(&opts.timeout, "test.timeout", 10*time.Minute, "")
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

func infoPlist(pkgpath string) string {
	return `<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
<key>CFBundleName</key><string>golang.gotest</string>
<key>CFBundleSupportedPlatforms</key><array><string>iPhoneOS</string></array>
<key>CFBundleExecutable</key><string>gotest</string>
<key>CFBundleVersion</key><string>1.0</string>
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
