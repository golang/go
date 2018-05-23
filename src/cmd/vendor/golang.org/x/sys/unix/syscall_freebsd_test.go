// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build freebsd

package unix_test

import (
	"flag"
	"fmt"
	"io/ioutil"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"runtime"
	"testing"

	"golang.org/x/sys/unix"
)

func TestSysctlUint64(t *testing.T) {
	_, err := unix.SysctlUint64("vm.swap_total")
	if err != nil {
		t.Fatal(err)
	}
}

// FIXME: Infrastructure for launching tests in subprocesses stolen from openbsd_test.go - refactor?
// testCmd generates a proper command that, when executed, runs the test
// corresponding to the given key.

type testProc struct {
	fn      func()                    // should always exit instead of returning
	arg     func(t *testing.T) string // generate argument for test
	cleanup func(arg string) error    // for instance, delete coredumps from testing pledge
	success bool                      // whether zero-exit means success or failure
}

var (
	testProcs = map[string]testProc{}
	procName  = ""
	procArg   = ""
)

const (
	optName = "sys-unix-internal-procname"
	optArg  = "sys-unix-internal-arg"
)

func init() {
	flag.StringVar(&procName, optName, "", "internal use only")
	flag.StringVar(&procArg, optArg, "", "internal use only")

}

func testCmd(procName string, procArg string) (*exec.Cmd, error) {
	exe, err := filepath.Abs(os.Args[0])
	if err != nil {
		return nil, err
	}
	cmd := exec.Command(exe, "-"+optName+"="+procName, "-"+optArg+"="+procArg)
	cmd.Stdout, cmd.Stderr = os.Stdout, os.Stderr
	return cmd, nil
}

// ExitsCorrectly is a comprehensive, one-line-of-use wrapper for testing
// a testProc with a key.
func ExitsCorrectly(t *testing.T, procName string) {
	s := testProcs[procName]
	arg := "-"
	if s.arg != nil {
		arg = s.arg(t)
	}
	c, err := testCmd(procName, arg)
	defer func(arg string) {
		if err := s.cleanup(arg); err != nil {
			t.Fatalf("Failed to run cleanup for %s %s %#v", procName, err, err)
		}
	}(arg)
	if err != nil {
		t.Fatalf("Failed to construct command for %s", procName)
	}
	if (c.Run() == nil) != s.success {
		result := "succeed"
		if !s.success {
			result = "fail"
		}
		t.Fatalf("Process did not %s when it was supposed to", result)
	}
}

func TestMain(m *testing.M) {
	flag.Parse()
	if procName != "" {
		t := testProcs[procName]
		t.fn()
		os.Stderr.WriteString("test function did not exit\n")
		if t.success {
			os.Exit(1)
		} else {
			os.Exit(0)
		}
	}
	os.Exit(m.Run())
}

// end of infrastructure

const testfile = "gocapmodetest"
const testfile2 = testfile + "2"

func CapEnterTest() {
	_, err := os.OpenFile(path.Join(procArg, testfile), os.O_WRONLY|os.O_CREATE|os.O_APPEND, 0666)
	if err != nil {
		panic(fmt.Sprintf("OpenFile: %s", err))
	}

	err = unix.CapEnter()
	if err != nil {
		panic(fmt.Sprintf("CapEnter: %s", err))
	}

	_, err = os.OpenFile(path.Join(procArg, testfile2), os.O_WRONLY|os.O_CREATE|os.O_APPEND, 0666)
	if err == nil {
		panic("OpenFile works!")
	}
	if err.(*os.PathError).Err != unix.ECAPMODE {
		panic(fmt.Sprintf("OpenFile failed wrong: %s %#v", err, err))
	}
	os.Exit(0)
}

func makeTempDir(t *testing.T) string {
	d, err := ioutil.TempDir("", "go_openat_test")
	if err != nil {
		t.Fatalf("TempDir failed: %s", err)
	}
	return d
}

func removeTempDir(arg string) error {
	err := os.RemoveAll(arg)
	if err != nil && err.(*os.PathError).Err == unix.ENOENT {
		return nil
	}
	return err
}

func init() {
	testProcs["cap_enter"] = testProc{
		CapEnterTest,
		makeTempDir,
		removeTempDir,
		true,
	}
}

func TestCapEnter(t *testing.T) {
	if runtime.GOARCH != "amd64" {
		t.Skipf("skipping test on %s", runtime.GOARCH)
	}
	ExitsCorrectly(t, "cap_enter")
}

func OpenatTest() {
	f, err := os.Open(procArg)
	if err != nil {
		panic(err)
	}

	err = unix.CapEnter()
	if err != nil {
		panic(fmt.Sprintf("CapEnter: %s", err))
	}

	fxx, err := unix.Openat(int(f.Fd()), "xx", os.O_WRONLY|os.O_CREATE|os.O_APPEND, 0666)
	if err != nil {
		panic(err)
	}
	unix.Close(fxx)

	// The right to open BASE/xx is not ambient
	_, err = os.OpenFile(procArg+"/xx", os.O_WRONLY|os.O_CREATE|os.O_APPEND, 0666)
	if err == nil {
		panic("OpenFile succeeded")
	}
	if err.(*os.PathError).Err != unix.ECAPMODE {
		panic(fmt.Sprintf("OpenFile failed wrong: %s %#v", err, err))
	}

	// Can't make a new directory either
	err = os.Mkdir(procArg+"2", 0777)
	if err == nil {
		panic("MKdir succeeded")
	}
	if err.(*os.PathError).Err != unix.ECAPMODE {
		panic(fmt.Sprintf("Mkdir failed wrong: %s %#v", err, err))
	}

	// Remove all caps except read and lookup.
	r, err := unix.CapRightsInit([]uint64{unix.CAP_READ, unix.CAP_LOOKUP})
	if err != nil {
		panic(fmt.Sprintf("CapRightsInit failed: %s %#v", err, err))
	}
	err = unix.CapRightsLimit(f.Fd(), r)
	if err != nil {
		panic(fmt.Sprintf("CapRightsLimit failed: %s %#v", err, err))
	}

	// Check we can get the rights back again
	r, err = unix.CapRightsGet(f.Fd())
	if err != nil {
		panic(fmt.Sprintf("CapRightsGet failed: %s %#v", err, err))
	}
	b, err := unix.CapRightsIsSet(r, []uint64{unix.CAP_READ, unix.CAP_LOOKUP})
	if err != nil {
		panic(fmt.Sprintf("CapRightsIsSet failed: %s %#v", err, err))
	}
	if !b {
		panic(fmt.Sprintf("Unexpected rights"))
	}
	b, err = unix.CapRightsIsSet(r, []uint64{unix.CAP_READ, unix.CAP_LOOKUP, unix.CAP_WRITE})
	if err != nil {
		panic(fmt.Sprintf("CapRightsIsSet failed: %s %#v", err, err))
	}
	if b {
		panic(fmt.Sprintf("Unexpected rights (2)"))
	}

	// Can no longer create a file
	_, err = unix.Openat(int(f.Fd()), "xx2", os.O_WRONLY|os.O_CREATE|os.O_APPEND, 0666)
	if err == nil {
		panic("Openat succeeded")
	}
	if err != unix.ENOTCAPABLE {
		panic(fmt.Sprintf("OpenFileAt failed wrong: %s %#v", err, err))
	}

	// But can read an existing one
	_, err = unix.Openat(int(f.Fd()), "xx", os.O_RDONLY, 0666)
	if err != nil {
		panic(fmt.Sprintf("Openat failed: %s %#v", err, err))
	}

	os.Exit(0)
}

func init() {
	testProcs["openat"] = testProc{
		OpenatTest,
		makeTempDir,
		removeTempDir,
		true,
	}
}

func TestOpenat(t *testing.T) {
	if runtime.GOARCH != "amd64" {
		t.Skipf("skipping test on %s", runtime.GOARCH)
	}
	ExitsCorrectly(t, "openat")
}

func TestCapRightsSetAndClear(t *testing.T) {
	r, err := unix.CapRightsInit([]uint64{unix.CAP_READ, unix.CAP_WRITE, unix.CAP_PDWAIT})
	if err != nil {
		t.Fatalf("CapRightsInit failed: %s", err)
	}

	err = unix.CapRightsSet(r, []uint64{unix.CAP_EVENT, unix.CAP_LISTEN})
	if err != nil {
		t.Fatalf("CapRightsSet failed: %s", err)
	}

	b, err := unix.CapRightsIsSet(r, []uint64{unix.CAP_READ, unix.CAP_WRITE, unix.CAP_PDWAIT, unix.CAP_EVENT, unix.CAP_LISTEN})
	if err != nil {
		t.Fatalf("CapRightsIsSet failed: %s", err)
	}
	if !b {
		t.Fatalf("Wrong rights set")
	}

	err = unix.CapRightsClear(r, []uint64{unix.CAP_READ, unix.CAP_PDWAIT})
	if err != nil {
		t.Fatalf("CapRightsClear failed: %s", err)
	}

	b, err = unix.CapRightsIsSet(r, []uint64{unix.CAP_WRITE, unix.CAP_EVENT, unix.CAP_LISTEN})
	if err != nil {
		t.Fatalf("CapRightsIsSet failed: %s", err)
	}
	if !b {
		t.Fatalf("Wrong rights set")
	}
}

// stringsFromByteSlice converts a sequence of attributes to a []string.
// On FreeBSD, each entry consists of a single byte containing the length
// of the attribute name, followed by the attribute name.
// The name is _not_ NULL-terminated.
func stringsFromByteSlice(buf []byte) []string {
	var result []string
	i := 0
	for i < len(buf) {
		next := i + 1 + int(buf[i])
		result = append(result, string(buf[i+1:next]))
		i = next
	}
	return result
}
