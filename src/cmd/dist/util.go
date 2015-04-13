// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

// pathf is fmt.Sprintf for generating paths
// (on windows it turns / into \ after the printf).
func pathf(format string, args ...interface{}) string {
	return filepath.Clean(fmt.Sprintf(format, args...))
}

// filter returns a slice containing the elements x from list for which f(x) == true.
func filter(list []string, f func(string) bool) []string {
	var out []string
	for _, x := range list {
		if f(x) {
			out = append(out, x)
		}
	}
	return out
}

// uniq returns a sorted slice containing the unique elements of list.
func uniq(list []string) []string {
	out := make([]string, len(list))
	copy(out, list)
	sort.Strings(out)
	keep := out[:0]
	for _, x := range out {
		if len(keep) == 0 || keep[len(keep)-1] != x {
			keep = append(keep, x)
		}
	}
	return keep
}

// splitlines returns a slice with the result of splitting
// the input p after each \n.
func splitlines(p string) []string {
	return strings.SplitAfter(p, "\n")
}

// splitfields replaces the vector v with the result of splitting
// the input p into non-empty fields containing no spaces.
func splitfields(p string) []string {
	return strings.Fields(p)
}

const (
	CheckExit = 1 << iota
	ShowOutput
	Background
)

var outputLock sync.Mutex

// run runs the command line cmd in dir.
// If mode has ShowOutput set, run collects cmd's output and returns it as a string;
// otherwise, run prints cmd's output to standard output after the command finishes.
// If mode has CheckExit set and the command fails, run calls fatal.
// If mode has Background set, this command is being run as a
// Background job. Only bgrun should use the Background mode,
// not other callers.
func run(dir string, mode int, cmd ...string) string {
	if vflag > 1 {
		errprintf("run: %s\n", strings.Join(cmd, " "))
	}

	xcmd := exec.Command(cmd[0], cmd[1:]...)
	xcmd.Dir = dir
	var data []byte
	var err error

	// If we want to show command output and this is not
	// a background command, assume it's the only thing
	// running, so we can just let it write directly stdout/stderr
	// as it runs without fear of mixing the output with some
	// other command's output. Not buffering lets the output
	// appear as it is printed instead of once the command exits.
	// This is most important for the invocation of 'go1.4 build -v bootstrap/...'.
	if mode&(Background|ShowOutput) == ShowOutput {
		xcmd.Stdout = os.Stdout
		xcmd.Stderr = os.Stderr
		err = xcmd.Run()
	} else {
		data, err = xcmd.CombinedOutput()
	}
	if err != nil && mode&CheckExit != 0 {
		outputLock.Lock()
		if len(data) > 0 {
			xprintf("%s\n", data)
		}
		outputLock.Unlock()
		if mode&Background != 0 {
			bgdied.Done()
		}
		fatal("FAILED: %v: %v", strings.Join(cmd, " "), err)
	}
	if mode&ShowOutput != 0 {
		outputLock.Lock()
		os.Stdout.Write(data)
		outputLock.Unlock()
	}
	if vflag > 2 {
		errprintf("run: %s DONE\n", strings.Join(cmd, " "))
	}
	return string(data)
}

var maxbg = 4 /* maximum number of jobs to run at once */

var (
	bgwork = make(chan func(), 1e5)
	bgdone = make(chan struct{}, 1e5)

	bgdied sync.WaitGroup
	nwork  int32
	ndone  int32

	dying  = make(chan bool)
	nfatal int32
)

func bginit() {
	bgdied.Add(maxbg)
	for i := 0; i < maxbg; i++ {
		go bghelper()
	}
}

func bghelper() {
	for {
		w := <-bgwork
		w()

		// Stop if we're dying.
		if atomic.LoadInt32(&nfatal) > 0 {
			bgdied.Done()
			return
		}
	}
}

// bgrun is like run but runs the command in the background.
// CheckExit|ShowOutput mode is implied (since output cannot be returned).
func bgrun(dir string, cmd ...string) {
	bgwork <- func() {
		run(dir, CheckExit|ShowOutput|Background, cmd...)
	}
}

// bgwait waits for pending bgruns to finish.
// bgwait must be called from only a single goroutine at a time.
func bgwait() {
	var wg sync.WaitGroup
	wg.Add(maxbg)
	done := make(chan bool)
	for i := 0; i < maxbg; i++ {
		bgwork <- func() {
			wg.Done()

			// Hold up bg goroutine until either the wait finishes
			// or the program starts dying due to a call to fatal.
			select {
			case <-dying:
			case <-done:
			}
		}
	}
	wg.Wait()
	close(done)
}

// xgetwd returns the current directory.
func xgetwd() string {
	wd, err := os.Getwd()
	if err != nil {
		fatal("%s", err)
	}
	return wd
}

// xrealwd returns the 'real' name for the given path.
// real is defined as what xgetwd returns in that directory.
func xrealwd(path string) string {
	old := xgetwd()
	if err := os.Chdir(path); err != nil {
		fatal("chdir %s: %v", path, err)
	}
	real := xgetwd()
	if err := os.Chdir(old); err != nil {
		fatal("chdir %s: %v", old, err)
	}
	return real
}

// isdir reports whether p names an existing directory.
func isdir(p string) bool {
	fi, err := os.Stat(p)
	return err == nil && fi.IsDir()
}

// isfile reports whether p names an existing file.
func isfile(p string) bool {
	fi, err := os.Stat(p)
	return err == nil && fi.Mode().IsRegular()
}

// mtime returns the modification time of the file p.
func mtime(p string) time.Time {
	fi, err := os.Stat(p)
	if err != nil {
		return time.Time{}
	}
	return fi.ModTime()
}

// isabs reports whether p is an absolute path.
func isabs(p string) bool {
	return filepath.IsAbs(p)
}

// readfile returns the content of the named file.
func readfile(file string) string {
	data, err := ioutil.ReadFile(file)
	if err != nil {
		fatal("%v", err)
	}
	return string(data)
}

// writefile writes b to the named file, creating it if needed.  if
// exec is non-zero, marks the file as executable.
func writefile(b, file string, exec int) {
	mode := os.FileMode(0666)
	if exec != 0 {
		mode = 0777
	}
	err := ioutil.WriteFile(file, []byte(b), mode)
	if err != nil {
		fatal("%v", err)
	}
}

// xmkdir creates the directory p.
func xmkdir(p string) {
	err := os.Mkdir(p, 0777)
	if err != nil {
		fatal("%v", err)
	}
}

// xmkdirall creates the directory p and its parents, as needed.
func xmkdirall(p string) {
	err := os.MkdirAll(p, 0777)
	if err != nil {
		fatal("%v", err)
	}
}

// xremove removes the file p.
func xremove(p string) {
	if vflag > 2 {
		errprintf("rm %s\n", p)
	}
	os.Remove(p)
}

// xremoveall removes the file or directory tree rooted at p.
func xremoveall(p string) {
	if vflag > 2 {
		errprintf("rm -r %s\n", p)
	}
	os.RemoveAll(p)
}

// xreaddir replaces dst with a list of the names of the files and subdirectories in dir.
// The names are relative to dir; they are not full paths.
func xreaddir(dir string) []string {
	f, err := os.Open(dir)
	if err != nil {
		fatal("%v", err)
	}
	defer f.Close()
	names, err := f.Readdirnames(-1)
	if err != nil {
		fatal("reading %s: %v", dir, err)
	}
	return names
}

// xreaddir replaces dst with a list of the names of the files in dir.
// The names are relative to dir; they are not full paths.
func xreaddirfiles(dir string) []string {
	f, err := os.Open(dir)
	if err != nil {
		fatal("%v", err)
	}
	defer f.Close()
	infos, err := f.Readdir(-1)
	if err != nil {
		fatal("reading %s: %v", dir, err)
	}
	var names []string
	for _, fi := range infos {
		if !fi.IsDir() {
			names = append(names, fi.Name())
		}
	}
	return names
}

// xworkdir creates a new temporary directory to hold object files
// and returns the name of that directory.
func xworkdir() string {
	name, err := ioutil.TempDir("", "go-tool-dist-")
	if err != nil {
		fatal("%v", err)
	}
	return name
}

// fatal prints an error message to standard error and exits.
func fatal(format string, args ...interface{}) {
	fmt.Fprintf(os.Stderr, "go tool dist: %s\n", fmt.Sprintf(format, args...))

	// Wait for background goroutines to finish,
	// so that exit handler that removes the work directory
	// is not fighting with active writes or open files.
	if atomic.AddInt32(&nfatal, 1) == 1 {
		close(dying)
	}
	for i := 0; i < maxbg; i++ {
		bgwork <- func() {} // wake up workers so they notice nfatal > 0
	}
	bgdied.Wait()

	xexit(2)
}

var atexits []func()

// xexit exits the process with return code n.
func xexit(n int) {
	for i := len(atexits) - 1; i >= 0; i-- {
		atexits[i]()
	}
	os.Exit(n)
}

// xatexit schedules the exit-handler f to be run when the program exits.
func xatexit(f func()) {
	atexits = append(atexits, f)
}

// xprintf prints a message to standard output.
func xprintf(format string, args ...interface{}) {
	fmt.Printf(format, args...)
}

// errprintf prints a message to standard output.
func errprintf(format string, args ...interface{}) {
	fmt.Fprintf(os.Stderr, format, args...)
}

// main takes care of OS-specific startup and dispatches to xmain.
func main() {
	os.Setenv("TERM", "dumb") // disable escape codes in clang errors

	slash = string(filepath.Separator)

	gohostos = runtime.GOOS
	switch gohostos {
	case "darwin":
		// Even on 64-bit platform, darwin uname -m prints i386.
		if strings.Contains(run("", CheckExit, "sysctl", "machdep.cpu.extfeatures"), "EM64T") {
			gohostarch = "amd64"
		}
	case "solaris":
		// Even on 64-bit platform, solaris uname -m prints i86pc.
		out := run("", CheckExit, "isainfo", "-n")
		if strings.Contains(out, "amd64") {
			gohostarch = "amd64"
		}
		if strings.Contains(out, "i386") {
			gohostarch = "386"
		}
	case "plan9":
		gohostarch = os.Getenv("objtype")
		if gohostarch == "" {
			fatal("$objtype is unset")
		}
	case "windows":
		exe = ".exe"
	}

	sysinit()

	if gohostarch == "" {
		// Default Unix system.
		out := run("", CheckExit, "uname", "-m")
		switch {
		case strings.Contains(out, "x86_64"), strings.Contains(out, "amd64"):
			gohostarch = "amd64"
		case strings.Contains(out, "86"):
			gohostarch = "386"
		case strings.Contains(out, "arm"):
			gohostarch = "arm"
		case strings.Contains(out, "aarch64"):
			gohostarch = "arm64"
		case strings.Contains(out, "ppc64le"):
			gohostarch = "ppc64le"
		case strings.Contains(out, "ppc64"):
			gohostarch = "ppc64"
		case gohostos == "darwin":
			if strings.Contains(run("", CheckExit, "uname", "-v"), "RELEASE_ARM_") {
				gohostarch = "arm"
			}
		default:
			fatal("unknown architecture: %s", out)
		}
	}

	if gohostarch == "arm" {
		maxbg = min(maxbg, runtime.NumCPU())
	}
	bginit()

	// The OS X 10.6 linker does not support external linking mode.
	// See golang.org/issue/5130.
	//
	// OS X 10.6 does not work with clang either, but OS X 10.9 requires it.
	// It seems to work with OS X 10.8, so we default to clang for 10.8 and later.
	// See golang.org/issue/5822.
	//
	// Roughly, OS X 10.N shows up as uname release (N+4),
	// so OS X 10.6 is uname version 10 and OS X 10.8 is uname version 12.
	if gohostos == "darwin" {
		rel := run("", CheckExit, "uname", "-r")
		if i := strings.Index(rel, "."); i >= 0 {
			rel = rel[:i]
		}
		osx, _ := strconv.Atoi(rel)
		if osx <= 6+4 {
			goextlinkenabled = "0"
		}
		if osx >= 8+4 {
			defaultclang = true
		}
	}

	xinit()
	xmain()
	xexit(0)
}

// xsamefile reports whether f1 and f2 are the same file (or dir)
func xsamefile(f1, f2 string) bool {
	fi1, err1 := os.Stat(f1)
	fi2, err2 := os.Stat(f2)
	if err1 != nil || err2 != nil {
		return f1 == f2
	}
	return os.SameFile(fi1, fi2)
}

func xgetgoarm() string {
	if goos == "nacl" {
		// NaCl guarantees VFPv3 and is always cross-compiled.
		return "7"
	}
	if goos == "darwin" {
		// Assume all darwin/arm devices are have VFPv3. This
		// port is also mostly cross-compiled, so it makes little
		// sense to auto-detect the setting.
		return "7"
	}
	if gohostarch != "arm" || goos != gohostos {
		// Conservative default for cross-compilation.
		return "5"
	}
	if goos == "freebsd" || goos == "openbsd" {
		// FreeBSD has broken VFP support.
		// OpenBSD currently only supports softfloat.
		return "5"
	}
	if goos != "linux" {
		// All other arm platforms that we support
		// require ARMv7.
		return "7"
	}
	cpuinfo := readfile("/proc/cpuinfo")
	goarm := "5"
	for _, line := range splitlines(cpuinfo) {
		line := strings.SplitN(line, ":", 2)
		if len(line) < 2 {
			continue
		}
		if strings.TrimSpace(line[0]) != "Features" {
			continue
		}
		features := splitfields(line[1])
		sort.Strings(features) // so vfpv3 sorts after vfp

		// Infer GOARM value from the vfp features available
		// on this host. Values of GOARM detected are:
		// 5: no vfp support was found
		// 6: vfp (v1) support was detected, but no higher
		// 7: vfpv3 support was detected.
		// This matches the assertions in runtime.checkarm.
		for _, f := range features {
			switch f {
			case "vfp":
				goarm = "6"
			case "vfpv3":
				goarm = "7"
			}
		}
	}
	return goarm
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
