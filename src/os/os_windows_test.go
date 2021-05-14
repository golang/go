// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package os_test

import (
	"errors"
	"fmt"
	"internal/poll"
	"internal/syscall/windows"
	"internal/syscall/windows/registry"
	"internal/testenv"
	"io"
	"io/fs"
	"os"
	osexec "os/exec"
	"path/filepath"
	"reflect"
	"runtime"
	"sort"
	"strings"
	"syscall"
	"testing"
	"unicode/utf16"
	"unsafe"
)

// For TestRawConnReadWrite.
type syscallDescriptor = syscall.Handle

// chdir changes the current working directory to the named directory,
// and then restore the original working directory at the end of the test.
func chdir(t *testing.T, dir string) {
	olddir, err := os.Getwd()
	if err != nil {
		t.Fatalf("chdir: %v", err)
	}
	if err := os.Chdir(dir); err != nil {
		t.Fatalf("chdir %s: %v", dir, err)
	}

	t.Cleanup(func() {
		if err := os.Chdir(olddir); err != nil {
			t.Errorf("chdir to original working directory %s: %v", olddir, err)
			os.Exit(1)
		}
	})
}

func TestSameWindowsFile(t *testing.T) {
	temp, err := os.MkdirTemp("", "TestSameWindowsFile")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(temp)
	chdir(t, temp)

	f, err := os.Create("a")
	if err != nil {
		t.Fatal(err)
	}
	f.Close()

	ia1, err := os.Stat("a")
	if err != nil {
		t.Fatal(err)
	}

	path, err := filepath.Abs("a")
	if err != nil {
		t.Fatal(err)
	}
	ia2, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	if !os.SameFile(ia1, ia2) {
		t.Errorf("files should be same")
	}

	p := filepath.VolumeName(path) + filepath.Base(path)
	if err != nil {
		t.Fatal(err)
	}
	ia3, err := os.Stat(p)
	if err != nil {
		t.Fatal(err)
	}
	if !os.SameFile(ia1, ia3) {
		t.Errorf("files should be same")
	}
}

type dirLinkTest struct {
	name    string
	mklink  func(link, target string) error
	issueNo int // correspondent issue number (for broken tests)
}

func testDirLinks(t *testing.T, tests []dirLinkTest) {
	tmpdir, err := os.MkdirTemp("", "testDirLinks")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmpdir)
	chdir(t, tmpdir)

	dir := filepath.Join(tmpdir, "dir")
	err = os.Mkdir(dir, 0777)
	if err != nil {
		t.Fatal(err)
	}
	fi, err := os.Stat(dir)
	if err != nil {
		t.Fatal(err)
	}
	err = os.WriteFile(filepath.Join(dir, "abc"), []byte("abc"), 0644)
	if err != nil {
		t.Fatal(err)
	}
	for _, test := range tests {
		link := filepath.Join(tmpdir, test.name+"_link")
		err := test.mklink(link, dir)
		if err != nil {
			t.Errorf("creating link for %q test failed: %v", test.name, err)
			continue
		}

		data, err := os.ReadFile(filepath.Join(link, "abc"))
		if err != nil {
			t.Errorf("failed to read abc file: %v", err)
			continue
		}
		if string(data) != "abc" {
			t.Errorf(`abc file is expected to have "abc" in it, but has %v`, data)
			continue
		}

		if test.issueNo > 0 {
			t.Logf("skipping broken %q test: see issue %d", test.name, test.issueNo)
			continue
		}

		fi1, err := os.Stat(link)
		if err != nil {
			t.Errorf("failed to stat link %v: %v", link, err)
			continue
		}
		if !fi1.IsDir() {
			t.Errorf("%q should be a directory", link)
			continue
		}
		if fi1.Name() != filepath.Base(link) {
			t.Errorf("Stat(%q).Name() = %q, want %q", link, fi1.Name(), filepath.Base(link))
			continue
		}
		if !os.SameFile(fi, fi1) {
			t.Errorf("%q should point to %q", link, dir)
			continue
		}

		fi2, err := os.Lstat(link)
		if err != nil {
			t.Errorf("failed to lstat link %v: %v", link, err)
			continue
		}
		if m := fi2.Mode(); m&fs.ModeSymlink == 0 {
			t.Errorf("%q should be a link, but is not (mode=0x%x)", link, uint32(m))
			continue
		}
		if m := fi2.Mode(); m&fs.ModeDir != 0 {
			t.Errorf("%q should be a link, not a directory (mode=0x%x)", link, uint32(m))
			continue
		}
	}
}

// reparseData is used to build reparse buffer data required for tests.
type reparseData struct {
	substituteName namePosition
	printName      namePosition
	pathBuf        []uint16
}

type namePosition struct {
	offset uint16
	length uint16
}

func (rd *reparseData) addUTF16s(s []uint16) (offset uint16) {
	off := len(rd.pathBuf) * 2
	rd.pathBuf = append(rd.pathBuf, s...)
	return uint16(off)
}

func (rd *reparseData) addString(s string) (offset, length uint16) {
	p := syscall.StringToUTF16(s)
	return rd.addUTF16s(p), uint16(len(p)-1) * 2 // do not include terminating NUL in the length (as per PrintNameLength and SubstituteNameLength documentation)
}

func (rd *reparseData) addSubstituteName(name string) {
	rd.substituteName.offset, rd.substituteName.length = rd.addString(name)
}

func (rd *reparseData) addPrintName(name string) {
	rd.printName.offset, rd.printName.length = rd.addString(name)
}

func (rd *reparseData) addStringNoNUL(s string) (offset, length uint16) {
	p := syscall.StringToUTF16(s)
	p = p[:len(p)-1]
	return rd.addUTF16s(p), uint16(len(p)) * 2
}

func (rd *reparseData) addSubstituteNameNoNUL(name string) {
	rd.substituteName.offset, rd.substituteName.length = rd.addStringNoNUL(name)
}

func (rd *reparseData) addPrintNameNoNUL(name string) {
	rd.printName.offset, rd.printName.length = rd.addStringNoNUL(name)
}

// pathBuffeLen returns length of rd pathBuf in bytes.
func (rd *reparseData) pathBuffeLen() uint16 {
	return uint16(len(rd.pathBuf)) * 2
}

// Windows REPARSE_DATA_BUFFER contains union member, and cannot be
// translated into Go directly. _REPARSE_DATA_BUFFER type is to help
// construct alternative versions of Windows REPARSE_DATA_BUFFER with
// union part of SymbolicLinkReparseBuffer or MountPointReparseBuffer type.
type _REPARSE_DATA_BUFFER struct {
	header windows.REPARSE_DATA_BUFFER_HEADER
	detail [syscall.MAXIMUM_REPARSE_DATA_BUFFER_SIZE]byte
}

func createDirLink(link string, rdb *_REPARSE_DATA_BUFFER) error {
	err := os.Mkdir(link, 0777)
	if err != nil {
		return err
	}

	linkp := syscall.StringToUTF16(link)
	fd, err := syscall.CreateFile(&linkp[0], syscall.GENERIC_WRITE, 0, nil, syscall.OPEN_EXISTING,
		syscall.FILE_FLAG_OPEN_REPARSE_POINT|syscall.FILE_FLAG_BACKUP_SEMANTICS, 0)
	if err != nil {
		return err
	}
	defer syscall.CloseHandle(fd)

	buflen := uint32(rdb.header.ReparseDataLength) + uint32(unsafe.Sizeof(rdb.header))
	var bytesReturned uint32
	return syscall.DeviceIoControl(fd, windows.FSCTL_SET_REPARSE_POINT,
		(*byte)(unsafe.Pointer(&rdb.header)), buflen, nil, 0, &bytesReturned, nil)
}

func createMountPoint(link string, target *reparseData) error {
	var buf *windows.MountPointReparseBuffer
	buflen := uint16(unsafe.Offsetof(buf.PathBuffer)) + target.pathBuffeLen() // see ReparseDataLength documentation
	byteblob := make([]byte, buflen)
	buf = (*windows.MountPointReparseBuffer)(unsafe.Pointer(&byteblob[0]))
	buf.SubstituteNameOffset = target.substituteName.offset
	buf.SubstituteNameLength = target.substituteName.length
	buf.PrintNameOffset = target.printName.offset
	buf.PrintNameLength = target.printName.length
	pbuflen := len(target.pathBuf)
	copy((*[2048]uint16)(unsafe.Pointer(&buf.PathBuffer[0]))[:pbuflen:pbuflen], target.pathBuf)

	var rdb _REPARSE_DATA_BUFFER
	rdb.header.ReparseTag = windows.IO_REPARSE_TAG_MOUNT_POINT
	rdb.header.ReparseDataLength = buflen
	copy(rdb.detail[:], byteblob)

	return createDirLink(link, &rdb)
}

func TestDirectoryJunction(t *testing.T) {
	var tests = []dirLinkTest{
		{
			// Create link similar to what mklink does, by inserting \??\ at the front of absolute target.
			name: "standard",
			mklink: func(link, target string) error {
				var t reparseData
				t.addSubstituteName(`\??\` + target)
				t.addPrintName(target)
				return createMountPoint(link, &t)
			},
		},
		{
			// Do as junction utility https://technet.microsoft.com/en-au/sysinternals/bb896768.aspx does - set PrintNameLength to 0.
			name: "have_blank_print_name",
			mklink: func(link, target string) error {
				var t reparseData
				t.addSubstituteName(`\??\` + target)
				t.addPrintName("")
				return createMountPoint(link, &t)
			},
		},
	}
	output, _ := osexec.Command("cmd", "/c", "mklink", "/?").Output()
	mklinkSupportsJunctionLinks := strings.Contains(string(output), " /J ")
	if mklinkSupportsJunctionLinks {
		tests = append(tests,
			dirLinkTest{
				name: "use_mklink_cmd",
				mklink: func(link, target string) error {
					output, err := osexec.Command("cmd", "/c", "mklink", "/J", link, target).CombinedOutput()
					if err != nil {
						t.Errorf("failed to run mklink %v %v: %v %q", link, target, err, output)
					}
					return nil
				},
			},
		)
	} else {
		t.Log(`skipping "use_mklink_cmd" test, mklink does not supports directory junctions`)
	}
	testDirLinks(t, tests)
}

func enableCurrentThreadPrivilege(privilegeName string) error {
	ct, err := windows.GetCurrentThread()
	if err != nil {
		return err
	}
	var t syscall.Token
	err = windows.OpenThreadToken(ct, syscall.TOKEN_QUERY|windows.TOKEN_ADJUST_PRIVILEGES, false, &t)
	if err != nil {
		return err
	}
	defer syscall.CloseHandle(syscall.Handle(t))

	var tp windows.TOKEN_PRIVILEGES

	privStr, err := syscall.UTF16PtrFromString(privilegeName)
	if err != nil {
		return err
	}
	err = windows.LookupPrivilegeValue(nil, privStr, &tp.Privileges[0].Luid)
	if err != nil {
		return err
	}
	tp.PrivilegeCount = 1
	tp.Privileges[0].Attributes = windows.SE_PRIVILEGE_ENABLED
	return windows.AdjustTokenPrivileges(t, false, &tp, 0, nil, nil)
}

func createSymbolicLink(link string, target *reparseData, isrelative bool) error {
	var buf *windows.SymbolicLinkReparseBuffer
	buflen := uint16(unsafe.Offsetof(buf.PathBuffer)) + target.pathBuffeLen() // see ReparseDataLength documentation
	byteblob := make([]byte, buflen)
	buf = (*windows.SymbolicLinkReparseBuffer)(unsafe.Pointer(&byteblob[0]))
	buf.SubstituteNameOffset = target.substituteName.offset
	buf.SubstituteNameLength = target.substituteName.length
	buf.PrintNameOffset = target.printName.offset
	buf.PrintNameLength = target.printName.length
	if isrelative {
		buf.Flags = windows.SYMLINK_FLAG_RELATIVE
	}
	pbuflen := len(target.pathBuf)
	copy((*[2048]uint16)(unsafe.Pointer(&buf.PathBuffer[0]))[:pbuflen:pbuflen], target.pathBuf)

	var rdb _REPARSE_DATA_BUFFER
	rdb.header.ReparseTag = syscall.IO_REPARSE_TAG_SYMLINK
	rdb.header.ReparseDataLength = buflen
	copy(rdb.detail[:], byteblob)

	return createDirLink(link, &rdb)
}

func TestDirectorySymbolicLink(t *testing.T) {
	var tests []dirLinkTest
	output, _ := osexec.Command("cmd", "/c", "mklink", "/?").Output()
	mklinkSupportsDirectorySymbolicLinks := strings.Contains(string(output), " /D ")
	if mklinkSupportsDirectorySymbolicLinks {
		tests = append(tests,
			dirLinkTest{
				name: "use_mklink_cmd",
				mklink: func(link, target string) error {
					output, err := osexec.Command("cmd", "/c", "mklink", "/D", link, target).CombinedOutput()
					if err != nil {
						t.Errorf("failed to run mklink %v %v: %v %q", link, target, err, output)
					}
					return nil
				},
			},
		)
	} else {
		t.Log(`skipping "use_mklink_cmd" test, mklink does not supports directory symbolic links`)
	}

	// The rest of these test requires SeCreateSymbolicLinkPrivilege to be held.
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	err := windows.ImpersonateSelf(windows.SecurityImpersonation)
	if err != nil {
		t.Fatal(err)
	}
	defer windows.RevertToSelf()

	err = enableCurrentThreadPrivilege("SeCreateSymbolicLinkPrivilege")
	if err != nil {
		t.Skipf(`skipping some tests, could not enable "SeCreateSymbolicLinkPrivilege": %v`, err)
	}
	tests = append(tests,
		dirLinkTest{
			name: "use_os_pkg",
			mklink: func(link, target string) error {
				return os.Symlink(target, link)
			},
		},
		dirLinkTest{
			// Create link similar to what mklink does, by inserting \??\ at the front of absolute target.
			name: "standard",
			mklink: func(link, target string) error {
				var t reparseData
				t.addPrintName(target)
				t.addSubstituteName(`\??\` + target)
				return createSymbolicLink(link, &t, false)
			},
		},
		dirLinkTest{
			name: "relative",
			mklink: func(link, target string) error {
				var t reparseData
				t.addSubstituteNameNoNUL(filepath.Base(target))
				t.addPrintNameNoNUL(filepath.Base(target))
				return createSymbolicLink(link, &t, true)
			},
		},
	)
	testDirLinks(t, tests)
}

func TestNetworkSymbolicLink(t *testing.T) {
	testenv.MustHaveSymlink(t)

	const _NERR_ServerNotStarted = syscall.Errno(2114)

	dir, err := os.MkdirTemp("", "TestNetworkSymbolicLink")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(dir)

	chdir(t, dir)

	shareName := "GoSymbolicLinkTestShare" // hope no conflictions
	sharePath := filepath.Join(dir, shareName)
	testDir := "TestDir"

	err = os.MkdirAll(filepath.Join(sharePath, testDir), 0777)
	if err != nil {
		t.Fatal(err)
	}

	wShareName, err := syscall.UTF16PtrFromString(shareName)
	if err != nil {
		t.Fatal(err)
	}
	wSharePath, err := syscall.UTF16PtrFromString(sharePath)
	if err != nil {
		t.Fatal(err)
	}

	p := windows.SHARE_INFO_2{
		Netname:     wShareName,
		Type:        windows.STYPE_DISKTREE,
		Remark:      nil,
		Permissions: 0,
		MaxUses:     1,
		CurrentUses: 0,
		Path:        wSharePath,
		Passwd:      nil,
	}

	err = windows.NetShareAdd(nil, 2, (*byte)(unsafe.Pointer(&p)), nil)
	if err != nil {
		if err == syscall.ERROR_ACCESS_DENIED {
			t.Skip("you don't have enough privileges to add network share")
		}
		if err == _NERR_ServerNotStarted {
			t.Skip(_NERR_ServerNotStarted.Error())
		}
		t.Fatal(err)
	}
	defer func() {
		err := windows.NetShareDel(nil, wShareName, 0)
		if err != nil {
			t.Fatal(err)
		}
	}()

	UNCPath := `\\localhost\` + shareName + `\`

	fi1, err := os.Stat(sharePath)
	if err != nil {
		t.Fatal(err)
	}
	fi2, err := os.Stat(UNCPath)
	if err != nil {
		t.Fatal(err)
	}
	if !os.SameFile(fi1, fi2) {
		t.Fatalf("%q and %q should be the same directory, but not", sharePath, UNCPath)
	}

	target := filepath.Join(UNCPath, testDir)
	link := "link"

	err = os.Symlink(target, link)
	if err != nil {
		t.Fatal(err)
	}
	defer os.Remove(link)

	got, err := os.Readlink(link)
	if err != nil {
		t.Fatal(err)
	}
	if got != target {
		t.Errorf(`os.Readlink("%s"): got %v, want %v`, link, got, target)
	}

	got, err = filepath.EvalSymlinks(link)
	if err != nil {
		t.Fatal(err)
	}
	if got != target {
		t.Errorf(`filepath.EvalSymlinks("%s"): got %v, want %v`, link, got, target)
	}
}

func TestStartProcessAttr(t *testing.T) {
	p, err := os.StartProcess(os.Getenv("COMSPEC"), []string{"/c", "cd"}, new(os.ProcAttr))
	if err != nil {
		return
	}
	defer p.Wait()
	t.Fatalf("StartProcess expected to fail, but succeeded.")
}

func TestShareNotExistError(t *testing.T) {
	if testing.Short() {
		t.Skip("slow test that uses network; skipping")
	}
	_, err := os.Stat(`\\no_such_server\no_such_share\no_such_file`)
	if err == nil {
		t.Fatal("stat succeeded, but expected to fail")
	}
	if !os.IsNotExist(err) {
		t.Fatalf("os.Stat failed with %q, but os.IsNotExist(err) is false", err)
	}
}

func TestBadNetPathError(t *testing.T) {
	const ERROR_BAD_NETPATH = syscall.Errno(53)
	if !os.IsNotExist(ERROR_BAD_NETPATH) {
		t.Fatal("os.IsNotExist(syscall.Errno(53)) is false, but want true")
	}
}

func TestStatDir(t *testing.T) {
	defer chtmpdir(t)()

	f, err := os.Open(".")
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()

	fi, err := f.Stat()
	if err != nil {
		t.Fatal(err)
	}

	err = os.Chdir("..")
	if err != nil {
		t.Fatal(err)
	}

	fi2, err := f.Stat()
	if err != nil {
		t.Fatal(err)
	}

	if !os.SameFile(fi, fi2) {
		t.Fatal("race condition occurred")
	}
}

func TestOpenVolumeName(t *testing.T) {
	tmpdir, err := os.MkdirTemp("", "TestOpenVolumeName")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmpdir)
	chdir(t, tmpdir)

	want := []string{"file1", "file2", "file3", "gopher.txt"}
	sort.Strings(want)
	for _, name := range want {
		err := os.WriteFile(filepath.Join(tmpdir, name), nil, 0777)
		if err != nil {
			t.Fatal(err)
		}
	}

	f, err := os.Open(filepath.VolumeName(tmpdir))
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()

	have, err := f.Readdirnames(-1)
	if err != nil {
		t.Fatal(err)
	}
	sort.Strings(have)

	if strings.Join(want, "/") != strings.Join(have, "/") {
		t.Fatalf("unexpected file list %q, want %q", have, want)
	}
}

func TestDeleteReadOnly(t *testing.T) {
	tmpdir := t.TempDir()
	p := filepath.Join(tmpdir, "a")
	// This sets FILE_ATTRIBUTE_READONLY.
	f, err := os.OpenFile(p, os.O_CREATE, 0400)
	if err != nil {
		t.Fatal(err)
	}
	f.Close()

	if err = os.Chmod(p, 0400); err != nil {
		t.Fatal(err)
	}
	if err = os.Remove(p); err != nil {
		t.Fatal(err)
	}
}

func TestStatSymlinkLoop(t *testing.T) {
	testenv.MustHaveSymlink(t)

	defer chtmpdir(t)()

	err := os.Symlink("x", "y")
	if err != nil {
		t.Fatal(err)
	}
	defer os.Remove("y")

	err = os.Symlink("y", "x")
	if err != nil {
		t.Fatal(err)
	}
	defer os.Remove("x")

	_, err = os.Stat("x")
	if _, ok := err.(*fs.PathError); !ok {
		t.Errorf("expected *PathError, got %T: %v\n", err, err)
	}
}

func TestReadStdin(t *testing.T) {
	old := poll.ReadConsole
	defer func() {
		poll.ReadConsole = old
	}()

	p, err := syscall.GetCurrentProcess()
	if err != nil {
		t.Fatalf("Unable to get handle to current process: %v", err)
	}
	var stdinDuplicate syscall.Handle
	err = syscall.DuplicateHandle(p, syscall.Handle(syscall.Stdin), p, &stdinDuplicate, 0, false, syscall.DUPLICATE_SAME_ACCESS)
	if err != nil {
		t.Fatalf("Unable to duplicate stdin: %v", err)
	}
	testConsole := os.NewConsoleFile(stdinDuplicate, "test")

	var tests = []string{
		"abc",
		"äöü",
		"\u3042",
		"“hi”™",
		"hello\x1aworld",
		"\U0001F648\U0001F649\U0001F64A",
	}

	for _, consoleSize := range []int{1, 2, 3, 10, 16, 100, 1000} {
		for _, readSize := range []int{1, 2, 3, 4, 5, 8, 10, 16, 20, 50, 100} {
			for _, s := range tests {
				t.Run(fmt.Sprintf("c%d/r%d/%s", consoleSize, readSize, s), func(t *testing.T) {
					s16 := utf16.Encode([]rune(s))
					poll.ReadConsole = func(h syscall.Handle, buf *uint16, toread uint32, read *uint32, inputControl *byte) error {
						if inputControl != nil {
							t.Fatalf("inputControl not nil")
						}
						n := int(toread)
						if n > consoleSize {
							n = consoleSize
						}
						n = copy((*[10000]uint16)(unsafe.Pointer(buf))[:n:n], s16)
						s16 = s16[n:]
						*read = uint32(n)
						t.Logf("read %d -> %d", toread, *read)
						return nil
					}

					var all []string
					var buf []byte
					chunk := make([]byte, readSize)
					for {
						n, err := testConsole.Read(chunk)
						buf = append(buf, chunk[:n]...)
						if err == io.EOF {
							all = append(all, string(buf))
							if len(all) >= 5 {
								break
							}
							buf = buf[:0]
						} else if err != nil {
							t.Fatalf("reading %q: error: %v", s, err)
						}
						if len(buf) >= 2000 {
							t.Fatalf("reading %q: stuck in loop: %q", s, buf)
						}
					}

					want := strings.Split(s, "\x1a")
					for len(want) < 5 {
						want = append(want, "")
					}
					if !reflect.DeepEqual(all, want) {
						t.Errorf("reading %q:\nhave %x\nwant %x", s, all, want)
					}
				})
			}
		}
	}
}

func TestStatPagefile(t *testing.T) {
	fi, err := os.Stat(`c:\pagefile.sys`)
	if err == nil {
		if fi.Name() == "" {
			t.Fatal(`FileInfo of c:\pagefile.sys has empty name`)
		}
		return
	}
	if os.IsNotExist(err) {
		t.Skip(`skipping because c:\pagefile.sys is not found`)
	}
	t.Fatal(err)
}

// syscallCommandLineToArgv calls syscall.CommandLineToArgv
// and converts returned result into []string.
func syscallCommandLineToArgv(cmd string) ([]string, error) {
	var argc int32
	argv, err := syscall.CommandLineToArgv(&syscall.StringToUTF16(cmd)[0], &argc)
	if err != nil {
		return nil, err
	}
	defer syscall.LocalFree(syscall.Handle(uintptr(unsafe.Pointer(argv))))

	var args []string
	for _, v := range (*argv)[:argc] {
		args = append(args, syscall.UTF16ToString((*v)[:]))
	}
	return args, nil
}

// compareCommandLineToArgvWithSyscall ensures that
// os.CommandLineToArgv(cmd) and syscall.CommandLineToArgv(cmd)
// return the same result.
func compareCommandLineToArgvWithSyscall(t *testing.T, cmd string) {
	syscallArgs, err := syscallCommandLineToArgv(cmd)
	if err != nil {
		t.Fatal(err)
	}
	args := os.CommandLineToArgv(cmd)
	if want, have := fmt.Sprintf("%q", syscallArgs), fmt.Sprintf("%q", args); want != have {
		t.Errorf("testing os.commandLineToArgv(%q) failed: have %q want %q", cmd, args, syscallArgs)
		return
	}
}

func TestCmdArgs(t *testing.T) {
	tmpdir := t.TempDir()

	const prog = `
package main

import (
	"fmt"
	"os"
)

func main() {
	fmt.Printf("%q", os.Args)
}
`
	src := filepath.Join(tmpdir, "main.go")
	if err := os.WriteFile(src, []byte(prog), 0666); err != nil {
		t.Fatal(err)
	}

	exe := filepath.Join(tmpdir, "main.exe")
	cmd := osexec.Command(testenv.GoToolPath(t), "build", "-o", exe, src)
	cmd.Dir = tmpdir
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("building main.exe failed: %v\n%s", err, out)
	}

	var cmds = []string{
		``,
		` a b c`,
		` "`,
		` ""`,
		` """`,
		` "" a`,
		` "123"`,
		` \"123\"`,
		` \"123 456\"`,
		` \\"`,
		` \\\"`,
		` \\\\\"`,
		` \\\"x`,
		` """"\""\\\"`,
		` abc`,
		` \\\\\""x"""y z`,
		"\tb\t\"x\ty\"",
		` "Брад" d e`,
		// examples from https://msdn.microsoft.com/en-us/library/17w5ykft.aspx
		` "abc" d e`,
		` a\\b d"e f"g h`,
		` a\\\"b c d`,
		` a\\\\"b c" d e`,
		// http://daviddeley.com/autohotkey/parameters/parameters.htm#WINARGV
		// from 5.4  Examples
		` CallMeIshmael`,
		` "Call Me Ishmael"`,
		` Cal"l Me I"shmael`,
		` CallMe\"Ishmael`,
		` "CallMe\"Ishmael"`,
		` "Call Me Ishmael\\"`,
		` "CallMe\\\"Ishmael"`,
		` a\\\b`,
		` "a\\\b"`,
		// from 5.5  Some Common Tasks
		` "\"Call Me Ishmael\""`,
		` "C:\TEST A\\"`,
		` "\"C:\TEST A\\\""`,
		// from 5.6  The Microsoft Examples Explained
		` "a b c"  d  e`,
		` "ab\"c"  "\\"  d`,
		` a\\\b d"e f"g h`,
		` a\\\"b c d`,
		` a\\\\"b c" d e`,
		// from 5.7  Double Double Quote Examples (pre 2008)
		` "a b c""`,
		` """CallMeIshmael"""  b  c`,
		` """Call Me Ishmael"""`,
		` """"Call Me Ishmael"" b c`,
	}
	for _, cmd := range cmds {
		compareCommandLineToArgvWithSyscall(t, "test"+cmd)
		compareCommandLineToArgvWithSyscall(t, `"cmd line"`+cmd)
		compareCommandLineToArgvWithSyscall(t, exe+cmd)

		// test both syscall.EscapeArg and os.commandLineToArgv
		args := os.CommandLineToArgv(exe + cmd)
		out, err := osexec.Command(args[0], args[1:]...).CombinedOutput()
		if err != nil {
			t.Fatalf("running %q failed: %v\n%v", args, err, string(out))
		}
		if want, have := fmt.Sprintf("%q", args), string(out); want != have {
			t.Errorf("wrong output of executing %q: have %q want %q", args, have, want)
			continue
		}
	}
}

func findOneDriveDir() (string, error) {
	// as per https://stackoverflow.com/questions/42519624/how-to-determine-location-of-onedrive-on-windows-7-and-8-in-c
	const onedrivekey = `SOFTWARE\Microsoft\OneDrive`
	k, err := registry.OpenKey(registry.CURRENT_USER, onedrivekey, registry.READ)
	if err != nil {
		return "", fmt.Errorf("OpenKey(%q) failed: %v", onedrivekey, err)
	}
	defer k.Close()

	path, _, err := k.GetStringValue("UserFolder")
	if err != nil {
		return "", fmt.Errorf("reading UserFolder failed: %v", err)
	}
	return path, nil
}

// TestOneDrive verifies that OneDrive folder is a directory and not a symlink.
func TestOneDrive(t *testing.T) {
	dir, err := findOneDriveDir()
	if err != nil {
		t.Skipf("Skipping, because we did not find OneDrive directory: %v", err)
	}
	testDirStats(t, dir)
}

func TestWindowsDevNullFile(t *testing.T) {
	testDevNullFile(t, "NUL", true)
	testDevNullFile(t, "nul", true)
	testDevNullFile(t, "Nul", true)

	f1, err := os.Open("NUL")
	if err != nil {
		t.Fatal(err)
	}
	defer f1.Close()

	fi1, err := f1.Stat()
	if err != nil {
		t.Fatal(err)
	}

	f2, err := os.Open("nul")
	if err != nil {
		t.Fatal(err)
	}
	defer f2.Close()

	fi2, err := f2.Stat()
	if err != nil {
		t.Fatal(err)
	}

	if !os.SameFile(fi1, fi2) {
		t.Errorf(`"NUL" and "nul" are not the same file`)
	}
}

// TestSymlinkCreation verifies that creating a symbolic link
// works on Windows when developer mode is active.
// This is supported starting Windows 10 (1703, v10.0.14972).
func TestSymlinkCreation(t *testing.T) {
	if !testenv.HasSymlink() && !isWindowsDeveloperModeActive() {
		t.Skip("Windows developer mode is not active")
	}
	t.Parallel()

	temp := t.TempDir()
	dummyFile := filepath.Join(temp, "file")
	if err := os.WriteFile(dummyFile, []byte(""), 0644); err != nil {
		t.Fatal(err)
	}

	linkFile := filepath.Join(temp, "link")
	if err := os.Symlink(dummyFile, linkFile); err != nil {
		t.Fatal(err)
	}
}

// isWindowsDeveloperModeActive checks whether or not the developer mode is active on Windows 10.
// Returns false for prior Windows versions.
// see https://docs.microsoft.com/en-us/windows/uwp/get-started/enable-your-device-for-development
func isWindowsDeveloperModeActive() bool {
	key, err := registry.OpenKey(registry.LOCAL_MACHINE, "SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\AppModelUnlock", registry.READ)
	if err != nil {
		return false
	}

	val, _, err := key.GetIntegerValue("AllowDevelopmentWithoutDevLicense")
	if err != nil {
		return false
	}

	return val != 0
}

// TestRootRelativeDirSymlink verifies that symlinks to paths relative to the
// drive root (beginning with "\" but no volume name) are created with the
// correct symlink type.
// (See https://golang.org/issue/39183#issuecomment-632175728.)
func TestRootRelativeDirSymlink(t *testing.T) {
	testenv.MustHaveSymlink(t)
	t.Parallel()

	temp := t.TempDir()
	dir := filepath.Join(temp, "dir")
	if err := os.Mkdir(dir, 0755); err != nil {
		t.Fatal(err)
	}

	volumeRelDir := strings.TrimPrefix(dir, filepath.VolumeName(dir)) // leaves leading backslash

	link := filepath.Join(temp, "link")
	err := os.Symlink(volumeRelDir, link)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("Symlink(%#q, %#q)", volumeRelDir, link)

	f, err := os.Open(link)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	if fi, err := f.Stat(); err != nil {
		t.Fatal(err)
	} else if !fi.IsDir() {
		t.Errorf("Open(%#q).Stat().IsDir() = false; want true", f.Name())
	}
}

// TestWorkingDirectoryRelativeSymlink verifies that symlinks to paths relative
// to the current working directory for the drive, such as "C:File.txt", are
// correctly converted to absolute links of the correct symlink type (per
// https://docs.microsoft.com/en-us/windows/win32/fileio/creating-symbolic-links).
func TestWorkingDirectoryRelativeSymlink(t *testing.T) {
	testenv.MustHaveSymlink(t)

	// Construct a directory to be symlinked.
	temp := t.TempDir()
	if v := filepath.VolumeName(temp); len(v) < 2 || v[1] != ':' {
		t.Skipf("Can't test relative symlinks: t.TempDir() (%#q) does not begin with a drive letter.", temp)
	}

	absDir := filepath.Join(temp, `dir\sub`)
	if err := os.MkdirAll(absDir, 0755); err != nil {
		t.Fatal(err)
	}

	// Change to the temporary directory and construct a
	// working-directory-relative symlink.
	oldwd, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := os.Chdir(oldwd); err != nil {
			t.Fatal(err)
		}
	}()
	if err := os.Chdir(temp); err != nil {
		t.Fatal(err)
	}
	t.Logf("Chdir(%#q)", temp)

	wdRelDir := filepath.VolumeName(temp) + `dir\sub` // no backslash after volume.
	absLink := filepath.Join(temp, "link")
	err = os.Symlink(wdRelDir, absLink)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("Symlink(%#q, %#q)", wdRelDir, absLink)

	// Now change back to the original working directory and verify that the
	// symlink still refers to its original path and is correctly marked as a
	// directory.
	if err := os.Chdir(oldwd); err != nil {
		t.Fatal(err)
	}
	t.Logf("Chdir(%#q)", oldwd)

	resolved, err := os.Readlink(absLink)
	if err != nil {
		t.Errorf("Readlink(%#q): %v", absLink, err)
	} else if resolved != absDir {
		t.Errorf("Readlink(%#q) = %#q; want %#q", absLink, resolved, absDir)
	}

	linkFile, err := os.Open(absLink)
	if err != nil {
		t.Fatal(err)
	}
	defer linkFile.Close()

	linkInfo, err := linkFile.Stat()
	if err != nil {
		t.Fatal(err)
	}
	if !linkInfo.IsDir() {
		t.Errorf("Open(%#q).Stat().IsDir() = false; want true", absLink)
	}

	absInfo, err := os.Stat(absDir)
	if err != nil {
		t.Fatal(err)
	}

	if !os.SameFile(absInfo, linkInfo) {
		t.Errorf("SameFile(Stat(%#q), Open(%#q).Stat()) = false; want true", absDir, absLink)
	}
}

// TestStatOfInvalidName is regression test for issue #24999.
func TestStatOfInvalidName(t *testing.T) {
	_, err := os.Stat("*.go")
	if err == nil {
		t.Fatal(`os.Stat("*.go") unexpectedly succeeded`)
	}
}

// findUnusedDriveLetter searches mounted drive list on the system
// (starting from Z: and ending at D:) for unused drive letter.
// It returns path to the found drive root directory (like Z:\) or error.
func findUnusedDriveLetter() (string, error) {
	// Do not use A: and B:, because they are reserved for floppy drive.
	// Do not use C:, because it is normally used for main drive.
	for l := 'Z'; l >= 'D'; l-- {
		p := string(l) + `:\`
		_, err := os.Stat(p)
		if os.IsNotExist(err) {
			return p, nil
		}
	}
	return "", errors.New("Could not find unused drive letter.")
}

func TestRootDirAsTemp(t *testing.T) {
	testenv.MustHaveExec(t)

	if os.Getenv("GO_WANT_HELPER_PROCESS") == "1" {
		fmt.Print(os.TempDir())
		os.Exit(0)
	}

	newtmp, err := findUnusedDriveLetter()
	if err != nil {
		t.Fatal(err)
	}

	cmd := osexec.Command(os.Args[0], "-test.run=TestRootDirAsTemp")
	cmd.Env = os.Environ()
	cmd.Env = append(cmd.Env, "GO_WANT_HELPER_PROCESS=1")
	cmd.Env = append(cmd.Env, "TMP="+newtmp)
	cmd.Env = append(cmd.Env, "TEMP="+newtmp)
	output, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("Failed to spawn child process: %v %q", err, string(output))
	}
	if want, have := newtmp, string(output); have != want {
		t.Fatalf("unexpected child process output %q, want %q", have, want)
	}
}

func testReadlink(t *testing.T, path, want string) {
	got, err := os.Readlink(path)
	if err != nil {
		t.Error(err)
		return
	}
	if got != want {
		t.Errorf(`Readlink(%q): got %q, want %q`, path, got, want)
	}
}

func mklink(t *testing.T, link, target string) {
	output, err := osexec.Command("cmd", "/c", "mklink", link, target).CombinedOutput()
	if err != nil {
		t.Fatalf("failed to run mklink %v %v: %v %q", link, target, err, output)
	}
}

func mklinkj(t *testing.T, link, target string) {
	output, err := osexec.Command("cmd", "/c", "mklink", "/J", link, target).CombinedOutput()
	if err != nil {
		t.Fatalf("failed to run mklink %v %v: %v %q", link, target, err, output)
	}
}

func mklinkd(t *testing.T, link, target string) {
	output, err := osexec.Command("cmd", "/c", "mklink", "/D", link, target).CombinedOutput()
	if err != nil {
		t.Fatalf("failed to run mklink %v %v: %v %q", link, target, err, output)
	}
}

func TestWindowsReadlink(t *testing.T) {
	tmpdir, err := os.MkdirTemp("", "TestWindowsReadlink")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmpdir)

	// Make sure tmpdir is not a symlink, otherwise tests will fail.
	tmpdir, err = filepath.EvalSymlinks(tmpdir)
	if err != nil {
		t.Fatal(err)
	}
	chdir(t, tmpdir)

	vol := filepath.VolumeName(tmpdir)
	output, err := osexec.Command("cmd", "/c", "mountvol", vol, "/L").CombinedOutput()
	if err != nil {
		t.Fatalf("failed to run mountvol %v /L: %v %q", vol, err, output)
	}
	ntvol := strings.Trim(string(output), " \n\r")

	dir := filepath.Join(tmpdir, "dir")
	err = os.MkdirAll(dir, 0777)
	if err != nil {
		t.Fatal(err)
	}

	absdirjlink := filepath.Join(tmpdir, "absdirjlink")
	mklinkj(t, absdirjlink, dir)
	testReadlink(t, absdirjlink, dir)

	ntdirjlink := filepath.Join(tmpdir, "ntdirjlink")
	mklinkj(t, ntdirjlink, ntvol+absdirjlink[len(filepath.VolumeName(absdirjlink)):])
	testReadlink(t, ntdirjlink, absdirjlink)

	ntdirjlinktolink := filepath.Join(tmpdir, "ntdirjlinktolink")
	mklinkj(t, ntdirjlinktolink, ntvol+absdirjlink[len(filepath.VolumeName(absdirjlink)):])
	testReadlink(t, ntdirjlinktolink, absdirjlink)

	mklinkj(t, "reldirjlink", "dir")
	testReadlink(t, "reldirjlink", dir) // relative directory junction resolves to absolute path

	// Make sure we have sufficient privilege to run mklink command.
	testenv.MustHaveSymlink(t)

	absdirlink := filepath.Join(tmpdir, "absdirlink")
	mklinkd(t, absdirlink, dir)
	testReadlink(t, absdirlink, dir)

	ntdirlink := filepath.Join(tmpdir, "ntdirlink")
	mklinkd(t, ntdirlink, ntvol+absdirlink[len(filepath.VolumeName(absdirlink)):])
	testReadlink(t, ntdirlink, absdirlink)

	mklinkd(t, "reldirlink", "dir")
	testReadlink(t, "reldirlink", "dir")

	file := filepath.Join(tmpdir, "file")
	err = os.WriteFile(file, []byte(""), 0666)
	if err != nil {
		t.Fatal(err)
	}

	filelink := filepath.Join(tmpdir, "filelink")
	mklink(t, filelink, file)
	testReadlink(t, filelink, file)

	linktofilelink := filepath.Join(tmpdir, "linktofilelink")
	mklink(t, linktofilelink, ntvol+filelink[len(filepath.VolumeName(filelink)):])
	testReadlink(t, linktofilelink, filelink)

	mklink(t, "relfilelink", "file")
	testReadlink(t, "relfilelink", "file")
}

// os.Mkdir(os.DevNull) fails.
func TestMkdirDevNull(t *testing.T) {
	err := os.Mkdir(os.DevNull, 777)
	oserr, ok := err.(*fs.PathError)
	if !ok {
		t.Fatalf("error (%T) is not *fs.PathError", err)
	}
	errno, ok := oserr.Err.(syscall.Errno)
	if !ok {
		t.Fatalf("error (%T) is not syscall.Errno", oserr)
	}
	if errno != syscall.ENOTDIR {
		t.Fatalf("error %d is not syscall.ENOTDIR", errno)
	}
}
