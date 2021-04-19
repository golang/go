// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bytes"
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"sync"
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
// If mode has ShowOutput set and Background unset, run passes cmd's output to
// stdout/stderr directly. Otherwise, run returns cmd's output as a string.
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
			// Prevent fatal from waiting on our own goroutine's
			// bghelper to exit:
			bghelpers.Done()
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

	bghelpers sync.WaitGroup

	dieOnce sync.Once // guards close of dying
	dying   = make(chan struct{})
)

func bginit() {
	bghelpers.Add(maxbg)
	for i := 0; i < maxbg; i++ {
		go bghelper()
	}
}

func bghelper() {
	defer bghelpers.Done()
	for {
		select {
		case <-dying:
			return
		case w := <-bgwork:
			// Dying takes precedence over doing more work.
			select {
			case <-dying:
				return
			default:
				w()
			}
		}
	}
}

// bgrun is like run but runs the command in the background.
// CheckExit|ShowOutput mode is implied (since output cannot be returned).
// bgrun adds 1 to wg immediately, and calls Done when the work completes.
func bgrun(wg *sync.WaitGroup, dir string, cmd ...string) {
	wg.Add(1)
	bgwork <- func() {
		defer wg.Done()
		run(dir, CheckExit|ShowOutput|Background, cmd...)
	}
}

// bgwait waits for pending bgruns to finish.
// bgwait must be called from only a single goroutine at a time.
func bgwait(wg *sync.WaitGroup) {
	done := make(chan struct{})
	go func() {
		wg.Wait()
		close(done)
	}()
	select {
	case <-done:
	case <-dying:
	}
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

const (
	writeExec = 1 << iota
	writeSkipSame
)

// writefile writes b to the named file, creating it if needed.
// if exec is non-zero, marks the file as executable.
// If the file already exists and has the expected content,
// it is not rewritten, to avoid changing the time stamp.
func writefile(b, file string, flag int) {
	new := []byte(b)
	if flag&writeSkipSame != 0 {
		old, err := ioutil.ReadFile(file)
		if err == nil && bytes.Equal(old, new) {
			return
		}
	}
	mode := os.FileMode(0666)
	if flag&writeExec != 0 {
		mode = 0777
	}
	err := ioutil.WriteFile(file, new, mode)
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

	dieOnce.Do(func() { close(dying) })

	// Wait for background goroutines to finish,
	// so that exit handler that removes the work directory
	// is not fighting with active writes or open files.
	bghelpers.Wait()

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

	// provide -check-armv6k first, before checking for $GOROOT so that
	// it is possible to run this check without having $GOROOT available.
	if len(os.Args) > 1 && os.Args[1] == "-check-armv6k" {
		useARMv6K() // might fail with SIGILL
		println("ARMv6K supported.")
		os.Exit(0)
	}

	slash = string(filepath.Separator)

	gohostos = runtime.GOOS
	switch gohostos {
	case "darwin":
		// Even on 64-bit platform, darwin uname -m prints i386.
		// We don't support any of the OS X versions that run on 32-bit-only hardware anymore.
		gohostarch = "amd64"
	case "freebsd":
		// Since FreeBSD 10 gcc is no longer part of the base system.
		defaultclang = true
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
		case strings.Contains(out, "mips64"):
			gohostarch = "mips64"
			if elfIsLittleEndian(os.Args[0]) {
				gohostarch = "mips64le"
			}
		case strings.Contains(out, "mips"):
			gohostarch = "mips"
			if elfIsLittleEndian(os.Args[0]) {
				gohostarch = "mipsle"
			}
		case strings.Contains(out, "s390x"):
			gohostarch = "s390x"
		case gohostos == "darwin":
			if strings.Contains(run("", CheckExit, "uname", "-v"), "RELEASE_ARM_") {
				gohostarch = "arm"
			}
		default:
			fatal("unknown architecture: %s", out)
		}
	}

	if gohostarch == "arm" || gohostarch == "mips64" || gohostarch == "mips64le" {
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

	if len(os.Args) > 1 && os.Args[1] == "-check-goarm" {
		useVFPv1() // might fail with SIGILL
		println("VFPv1 OK.")
		useVFPv3() // might fail with SIGILL
		println("VFPv3 OK.")
		os.Exit(0)
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

	// Try to exec ourselves in a mode to detect VFP support.
	// Seeing how far it gets determines which instructions failed.
	// The test is OS-agnostic.
	out := run("", 0, os.Args[0], "-check-goarm")
	v1ok := strings.Contains(out, "VFPv1 OK.")
	v3ok := strings.Contains(out, "VFPv3 OK.")

	if v1ok && v3ok {
		return "7"
	}
	if v1ok {
		return "6"
	}
	return "5"
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// elfIsLittleEndian detects if the ELF file is little endian.
func elfIsLittleEndian(fn string) bool {
	// read the ELF file header to determine the endianness without using the
	// debug/elf package.
	file, err := os.Open(fn)
	if err != nil {
		fatal("failed to open file to determine endianness: %v", err)
	}
	defer file.Close()
	var hdr [16]byte
	if _, err := io.ReadFull(file, hdr[:]); err != nil {
		fatal("failed to read ELF header to determine endianness: %v", err)
	}
	// hdr[5] is EI_DATA byte, 1 is ELFDATA2LSB and 2 is ELFDATA2MSB
	switch hdr[5] {
	default:
		fatal("unknown ELF endianness of %s: EI_DATA = %d", fn, hdr[5])
	case 1:
		return true
	case 2:
		return false
	}
	panic("unreachable")
}
