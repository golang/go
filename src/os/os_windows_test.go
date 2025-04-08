// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package os_test

import (
	"bytes"
	"errors"
	"fmt"
	"internal/godebug"
	"internal/poll"
	"internal/syscall/windows"
	"internal/syscall/windows/registry"
	"internal/testenv"
	"io"
	"io/fs"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"slices"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
	"testing"
	"time"
	"unicode/utf16"
	"unsafe"
)

var winsymlink = godebug.New("winsymlink")
var winreadlinkvolume = godebug.New("winreadlinkvolume")

// For TestRawConnReadWrite.
type syscallDescriptor = syscall.Handle

func TestSameWindowsFile(t *testing.T) {
	t.Chdir(t.TempDir())

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
	name         string
	mklink       func(link, target string) error
	isMountPoint bool
}

func testDirLinks(t *testing.T, tests []dirLinkTest) {
	tmpdir := t.TempDir()
	t.Chdir(tmpdir)

	dir := filepath.Join(tmpdir, "dir")
	err := os.Mkdir(dir, 0777)
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

		fi1, err := os.Stat(link)
		if err != nil {
			t.Errorf("failed to stat link %v: %v", link, err)
			continue
		}
		if tp := fi1.Mode().Type(); tp != fs.ModeDir {
			t.Errorf("Stat(%q) is type %v; want %v", link, tp, fs.ModeDir)
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
		var wantType fs.FileMode
		if test.isMountPoint && winsymlink.Value() != "0" {
			// Mount points are reparse points, and we no longer treat them as symlinks.
			wantType = fs.ModeIrregular
		} else {
			// This is either a real symlink, or a mount point treated as a symlink.
			wantType = fs.ModeSymlink
		}
		if tp := fi2.Mode().Type(); tp != wantType {
			t.Errorf("Lstat(%q) is type %v; want %v", link, tp, wantType)
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
			name:         "standard",
			isMountPoint: true,
			mklink: func(link, target string) error {
				var t reparseData
				t.addSubstituteName(`\??\` + target)
				t.addPrintName(target)
				return createMountPoint(link, &t)
			},
		},
		{
			// Do as junction utility https://learn.microsoft.com/en-us/sysinternals/downloads/junction does - set PrintNameLength to 0.
			name:         "have_blank_print_name",
			isMountPoint: true,
			mklink: func(link, target string) error {
				var t reparseData
				t.addSubstituteName(`\??\` + target)
				t.addPrintName("")
				return createMountPoint(link, &t)
			},
		},
	}
	output, _ := testenv.Command(t, "cmd", "/c", "mklink", "/?").Output()
	mklinkSupportsJunctionLinks := strings.Contains(string(output), " /J ")
	if mklinkSupportsJunctionLinks {
		tests = append(tests,
			dirLinkTest{
				name:         "use_mklink_cmd",
				isMountPoint: true,
				mklink: func(link, target string) error {
					output, err := testenv.Command(t, "cmd", "/c", "mklink", "/J", link, target).CombinedOutput()
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
	output, _ := testenv.Command(t, "cmd", "/c", "mklink", "/?").Output()
	mklinkSupportsDirectorySymbolicLinks := strings.Contains(string(output), " /D ")
	if mklinkSupportsDirectorySymbolicLinks {
		tests = append(tests,
			dirLinkTest{
				name: "use_mklink_cmd",
				mklink: func(link, target string) error {
					output, err := testenv.Command(t, "cmd", "/c", "mklink", "/D", link, target).CombinedOutput()
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

func mustHaveWorkstation(t *testing.T) {
	mar, err := windows.OpenSCManager(nil, nil, windows.SERVICE_QUERY_STATUS)
	if err != nil {
		return
	}
	defer syscall.CloseHandle(mar)
	//LanmanWorkstation is the service name, and Workstation is the display name.
	srv, err := windows.OpenService(mar, syscall.StringToUTF16Ptr("LanmanWorkstation"), windows.SERVICE_QUERY_STATUS)
	if err != nil {
		return
	}
	defer syscall.CloseHandle(srv)
	var state windows.SERVICE_STATUS
	err = windows.QueryServiceStatus(srv, &state)
	if err != nil {
		return
	}
	if state.CurrentState != windows.SERVICE_RUNNING {
		t.Skip("Requires the Windows service Workstation, but it is detected that it is not enabled.")
	}
}

func TestNetworkSymbolicLink(t *testing.T) {
	testenv.MustHaveSymlink(t)

	const _NERR_ServerNotStarted = syscall.Errno(2114)

	dir := t.TempDir()
	t.Chdir(dir)

	pid := os.Getpid()
	shareName := fmt.Sprintf("GoSymbolicLinkTestShare%d", pid)
	sharePath := filepath.Join(dir, shareName)
	testDir := "TestDir"

	err := os.MkdirAll(filepath.Join(sharePath, testDir), 0777)
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

	// Per https://learn.microsoft.com/en-us/windows/win32/api/lmshare/ns-lmshare-share_info_2:
	//
	// “[The shi2_permissions field] indicates the shared resource's permissions
	// for servers running with share-level security. A server running user-level
	// security ignores this member.
	// …
	// Note that Windows does not support share-level security.”
	//
	// So it shouldn't matter what permissions we set here.
	const permissions = 0

	p := windows.SHARE_INFO_2{
		Netname:     wShareName,
		Type:        windows.STYPE_DISKTREE | windows.STYPE_TEMPORARY,
		Remark:      nil,
		Permissions: permissions,
		MaxUses:     1,
		CurrentUses: 0,
		Path:        wSharePath,
		Passwd:      nil,
	}

	err = windows.NetShareAdd(nil, 2, (*byte)(unsafe.Pointer(&p)), nil)
	if err != nil {
		if err == syscall.ERROR_ACCESS_DENIED || err == _NERR_ServerNotStarted {
			t.Skipf("skipping: NetShareAdd: %v", err)
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
		mustHaveWorkstation(t)
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
		t.Errorf(`os.Readlink(%#q): got %v, want %v`, link, got, target)
	}

	got, err = filepath.EvalSymlinks(link)
	if err != nil {
		t.Fatal(err)
	}
	if got != target {
		t.Errorf(`filepath.EvalSymlinks(%#q): got %v, want %v`, link, got, target)
	}
}

func TestStatLxSymLink(t *testing.T) {
	if _, err := exec.LookPath("wsl"); err != nil {
		t.Skip("skipping: WSL not detected")
	}

	t.Chdir(t.TempDir())

	const target = "target"
	const link = "link"

	_, err := testenv.Command(t, "wsl", "/bin/mkdir", target).Output()
	if err != nil {
		// This normally happens when WSL still doesn't have a distro installed to run on.
		t.Skipf("skipping: WSL is not correctly installed: %v", err)
	}

	_, err = testenv.Command(t, "wsl", "/bin/ln", "-s", target, link).Output()
	if err != nil {
		t.Fatal(err)
	}

	fi, err := os.Lstat(link)
	if err != nil {
		t.Fatal(err)
	}
	if m := fi.Mode(); m&fs.ModeSymlink != 0 {
		// This can happen depending on newer WSL versions when running as admin or in developer mode.
		t.Skip("skipping: WSL created reparse tag IO_REPARSE_TAG_SYMLINK instead of an IO_REPARSE_TAG_LX_SYMLINK")
	}
	// Stat'ing a IO_REPARSE_TAG_LX_SYMLINK from outside WSL always return ERROR_CANT_ACCESS_FILE.
	// We check this condition to validate that os.Stat has tried to follow the link.
	_, err = os.Stat(link)
	const ERROR_CANT_ACCESS_FILE = syscall.Errno(1920)
	if err == nil || !errors.Is(err, ERROR_CANT_ACCESS_FILE) {
		t.Fatalf("os.Stat(%q): got %v, want ERROR_CANT_ACCESS_FILE", link, err)
	}
}

func TestStartProcessAttr(t *testing.T) {
	t.Parallel()

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
	t.Parallel()

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
	t.Chdir(t.TempDir())

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
	tmpdir := t.TempDir()
	t.Chdir(tmpdir)

	want := []string{"file1", "file2", "file3", "gopher.txt"}
	slices.Sort(want)
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
	slices.Sort(have)

	if strings.Join(want, "/") != strings.Join(have, "/") {
		t.Fatalf("unexpected file list %q, want %q", have, want)
	}
}

func TestDeleteReadOnly(t *testing.T) {
	t.Parallel()

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
					if !slices.Equal(all, want) {
						t.Errorf("reading %q:\nhave %x\nwant %x", s, all, want)
					}
				})
			}
		}
	}
}

func TestStatPagefile(t *testing.T) {
	t.Parallel()

	const path = `c:\pagefile.sys`
	fi, err := os.Stat(path)
	if err == nil {
		if fi.Name() == "" {
			t.Fatalf("Stat(%q).Name() is empty", path)
		}
		t.Logf("Stat(%q).Size() = %v", path, fi.Size())
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
	if testing.Short() {
		t.Skipf("in short mode; skipping test that builds a binary")
	}
	t.Parallel()

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
	cmd := testenv.Command(t, testenv.GoToolPath(t), "build", "-o", exe, src)
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
		// examples from https://learn.microsoft.com/en-us/cpp/cpp/main-function-command-line-args
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
		out, err := testenv.Command(t, args[0], args[1:]...).CombinedOutput()
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

	path, valtype, err := k.GetStringValue("UserFolder")
	if err != nil {
		return "", fmt.Errorf("reading UserFolder failed: %v", err)
	}

	// REG_SZ values may also contain environment variables that need to be expanded.
	// It's recommended but not required to use REG_EXPAND_SZ for paths that contain environment variables.
	if valtype == registry.EXPAND_SZ || valtype == registry.SZ {
		expanded, err := registry.ExpandString(path)
		if err != nil {
			return "", fmt.Errorf("expanding UserFolder failed: %v", err)
		}
		path = expanded
	}

	return path, nil
}

// TestOneDrive verifies that OneDrive folder is a directory and not a symlink.
func TestOneDrive(t *testing.T) {
	t.Parallel()

	dir, err := findOneDriveDir()
	if err != nil {
		t.Skipf("Skipping, because we did not find OneDrive directory: %v", err)
	}
	testDirStats(t, dir)
}

func TestWindowsDevNullFile(t *testing.T) {
	t.Parallel()

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

func TestFileStatNUL(t *testing.T) {
	t.Parallel()

	f, err := os.Open("NUL")
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()

	fi, err := f.Stat()
	if err != nil {
		t.Fatal(err)
	}
	if got, want := fi.Mode(), os.ModeDevice|os.ModeCharDevice|0666; got != want {
		t.Errorf("Open(%q).Stat().Mode() = %v, want %v", "NUL", got, want)
	}
}

func TestStatNUL(t *testing.T) {
	t.Parallel()

	fi, err := os.Stat("NUL")
	if err != nil {
		t.Fatal(err)
	}
	if got, want := fi.Mode(), os.ModeDevice|os.ModeCharDevice|0666; got != want {
		t.Errorf("Stat(%q).Mode() = %v, want %v", "NUL", got, want)
	}
}

// TestSymlinkCreation verifies that creating a symbolic link
// works on Windows when developer mode is active.
// This is supported starting Windows 10 (1703, v10.0.14972).
func TestSymlinkCreation(t *testing.T) {
	if !testenv.HasSymlink() {
		t.Skip("skipping test; no symlink support")
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
	t.Chdir(temp)
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
	t.Parallel()

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
	if os.Getenv("GO_WANT_HELPER_PROCESS") == "1" {
		fmt.Print(os.TempDir())
		os.Exit(0)
	}

	testenv.MustHaveExec(t)
	t.Parallel()

	exe := testenv.Executable(t)

	newtmp, err := findUnusedDriveLetter()
	if err != nil {
		t.Skip(err)
	}

	cmd := testenv.Command(t, exe, "-test.run=^TestRootDirAsTemp$")
	cmd.Env = cmd.Environ()
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

// replaceDriveWithVolumeID returns path with its volume name replaced with
// the mounted volume ID. E.g. C:\foo -> \\?\Volume{GUID}\foo.
func replaceDriveWithVolumeID(t *testing.T, path string) string {
	t.Helper()
	cmd := testenv.Command(t, "cmd", "/c", "mountvol", filepath.VolumeName(path), "/L")
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("%v: %v\n%s", cmd, err, out)
	}
	vol := strings.Trim(string(out), " \n\r")
	return filepath.Join(vol, path[len(filepath.VolumeName(path)):])
}

func TestReadlink(t *testing.T) {
	tests := []struct {
		junction bool
		dir      bool
		drive    bool
		relative bool
	}{
		{junction: true, dir: true, drive: true, relative: false},
		{junction: true, dir: true, drive: false, relative: false},
		{junction: true, dir: true, drive: false, relative: true},
		{junction: false, dir: true, drive: true, relative: false},
		{junction: false, dir: true, drive: false, relative: false},
		{junction: false, dir: true, drive: false, relative: true},
		{junction: false, dir: false, drive: true, relative: false},
		{junction: false, dir: false, drive: false, relative: false},
		{junction: false, dir: false, drive: false, relative: true},
	}
	for _, tt := range tests {
		tt := tt
		var name string
		if tt.junction {
			name = "junction"
		} else {
			name = "symlink"
		}
		if tt.dir {
			name += "_dir"
		} else {
			name += "_file"
		}
		if tt.drive {
			name += "_drive"
		} else {
			name += "_volume"
		}
		if tt.relative {
			name += "_relative"
		} else {
			name += "_absolute"
		}

		t.Run(name, func(t *testing.T) {
			if !tt.junction {
				testenv.MustHaveSymlink(t)
			}
			if !tt.relative {
				t.Parallel()
			}
			// Make sure tmpdir is not a symlink, otherwise tests will fail.
			tmpdir, err := filepath.EvalSymlinks(t.TempDir())
			if err != nil {
				t.Fatal(err)
			}
			link := filepath.Join(tmpdir, "link")
			target := filepath.Join(tmpdir, "target")
			if tt.dir {
				if err := os.MkdirAll(target, 0777); err != nil {
					t.Fatal(err)
				}
			} else {
				if err := os.WriteFile(target, nil, 0666); err != nil {
					t.Fatal(err)
				}
			}
			var want string
			if tt.relative {
				relTarget := filepath.Base(target)
				if tt.junction {
					want = target // relative directory junction resolves to absolute path
				} else {
					want = relTarget
				}
				t.Chdir(tmpdir)
				link = filepath.Base(link)
				target = relTarget
			} else {
				if tt.drive {
					want = target
				} else {
					volTarget := replaceDriveWithVolumeID(t, target)
					if winreadlinkvolume.Value() == "0" {
						want = target
					} else {
						want = volTarget
					}
					target = volTarget
				}
			}
			if tt.junction {
				cmd := testenv.Command(t, "cmd", "/c", "mklink", "/J", link, target)
				if out, err := cmd.CombinedOutput(); err != nil {
					t.Fatalf("%v: %v\n%s", cmd, err, out)
				}
			} else {
				if err := os.Symlink(target, link); err != nil {
					t.Fatalf("Symlink(%#q, %#q): %v", target, link, err)
				}
			}
			got, err := os.Readlink(link)
			if err != nil {
				t.Fatal(err)
			}
			if got != want {
				t.Fatalf("Readlink(%#q) = %#q; want %#q", target, got, want)
			}
		})
	}
}

func TestOpenDirTOCTOU(t *testing.T) {
	t.Parallel()

	// Check opened directories can't be renamed until the handle is closed.
	// See issue 52747.
	tmpdir := t.TempDir()
	dir := filepath.Join(tmpdir, "dir")
	if err := os.Mkdir(dir, 0777); err != nil {
		t.Fatal(err)
	}
	f, err := os.Open(dir)
	if err != nil {
		t.Fatal(err)
	}
	newpath := filepath.Join(tmpdir, "dir1")
	err = os.Rename(dir, newpath)
	if err == nil || !errors.Is(err, windows.ERROR_SHARING_VIOLATION) {
		f.Close()
		t.Fatalf("Rename(%q, %q) = %v; want windows.ERROR_SHARING_VIOLATION", dir, newpath, err)
	}
	f.Close()
	err = os.Rename(dir, newpath)
	if err != nil {
		t.Error(err)
	}
}

func TestAppExecLinkStat(t *testing.T) {
	// We expect executables installed to %LOCALAPPDATA%\Microsoft\WindowsApps to
	// be reparse points with tag IO_REPARSE_TAG_APPEXECLINK. Here we check that
	// such reparse points are treated as irregular (but executable) files, not
	// broken symlinks.
	appdata := os.Getenv("LOCALAPPDATA")
	if appdata == "" {
		t.Skipf("skipping: LOCALAPPDATA not set")
	}

	pythonExeName := "python3.exe"
	pythonPath := filepath.Join(appdata, `Microsoft\WindowsApps`, pythonExeName)

	lfi, err := os.Lstat(pythonPath)
	if err != nil {
		t.Skip("skipping test, because Python 3 is not installed via the Windows App Store on this system; see https://golang.org/issue/42919")
	}

	// An APPEXECLINK reparse point is not a symlink, so os.Readlink should return
	// a non-nil error for it, and Stat should return results identical to Lstat.
	linkName, err := os.Readlink(pythonPath)
	if err == nil {
		t.Errorf("os.Readlink(%q) = %q, but expected an error\n(should be an APPEXECLINK reparse point, not a symlink)", pythonPath, linkName)
	}

	sfi, err := os.Stat(pythonPath)
	if err != nil {
		t.Fatalf("Stat %s: %v", pythonPath, err)
	}

	if lfi.Name() != sfi.Name() {
		t.Logf("os.Lstat(%q) = %+v", pythonPath, lfi)
		t.Logf("os.Stat(%q)  = %+v", pythonPath, sfi)
		t.Errorf("files should be same")
	}

	if lfi.Name() != pythonExeName {
		t.Errorf("Stat %s: got %q, but wanted %q", pythonPath, lfi.Name(), pythonExeName)
	}
	if tp := lfi.Mode().Type(); tp != fs.ModeIrregular {
		// A reparse point is not a regular file, but we don't have a more appropriate
		// ModeType bit for it, so it should be marked as irregular.
		t.Errorf("%q should not be a an irregular file (mode=0x%x)", pythonPath, uint32(tp))
	}

	if sfi.Name() != pythonExeName {
		t.Errorf("Stat %s: got %q, but wanted %q", pythonPath, sfi.Name(), pythonExeName)
	}
	if m := sfi.Mode(); m&fs.ModeSymlink != 0 {
		t.Errorf("%q should be a file, not a link (mode=0x%x)", pythonPath, uint32(m))
	}
	if m := sfi.Mode(); m&fs.ModeDir != 0 {
		t.Errorf("%q should be a file, not a directory (mode=0x%x)", pythonPath, uint32(m))
	}
	if m := sfi.Mode(); m&fs.ModeIrregular == 0 {
		// A reparse point is not a regular file, but we don't have a more appropriate
		// ModeType bit for it, so it should be marked as irregular.
		t.Errorf("%q should not be a regular file (mode=0x%x)", pythonPath, uint32(m))
	}

	p, err := exec.LookPath(pythonPath)
	if err != nil {
		t.Errorf("exec.LookPath(%q): %v", pythonPath, err)
	}
	if p != pythonPath {
		t.Errorf("exec.LookPath(%q) = %q; want %q", pythonPath, p, pythonPath)
	}
}

func TestIllformedUTF16FileName(t *testing.T) {
	dir := t.TempDir()
	const sep = string(os.PathSeparator)
	if !strings.HasSuffix(dir, sep) {
		dir += sep
	}

	// This UTF-16 file name is ill-formed as it contains low surrogates that are not preceded by high surrogates ([1:5]).
	namew := []uint16{0x2e, 0xdc6d, 0xdc73, 0xdc79, 0xdc73, 0x30, 0x30, 0x30, 0x31, 0}

	// Create a file whose name contains unpaired surrogates.
	// Use syscall.CreateFile instead of os.Create to simulate a file that is created by
	// a non-Go program so the file name hasn't gone through syscall.UTF16FromString.
	dirw := utf16.Encode([]rune(dir))
	pathw := append(dirw, namew...)
	fd, err := syscall.CreateFile(&pathw[0], syscall.GENERIC_ALL, 0, nil, syscall.CREATE_NEW, 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	syscall.CloseHandle(fd)

	name := syscall.UTF16ToString(namew)
	path := filepath.Join(dir, name)
	// Verify that os.Lstat can query the file.
	fi, err := os.Lstat(path)
	if err != nil {
		t.Fatal(err)
	}
	if got := fi.Name(); got != name {
		t.Errorf("got %q, want %q", got, name)
	}
	// Verify that File.Readdirnames lists the file.
	f, err := os.Open(dir)
	if err != nil {
		t.Fatal(err)
	}
	files, err := f.Readdirnames(0)
	f.Close()
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Contains(files, name) {
		t.Error("file not listed")
	}
	// Verify that os.RemoveAll can remove the directory
	// and that it doesn't hang.
	err = os.RemoveAll(dir)
	if err != nil {
		t.Error(err)
	}
}

func TestUTF16Alloc(t *testing.T) {
	allowsPerRun := func(want int, f func()) {
		t.Helper()
		got := int(testing.AllocsPerRun(5, f))
		if got != want {
			t.Errorf("got %d allocs, want %d", got, want)
		}
	}
	allowsPerRun(1, func() {
		syscall.UTF16ToString([]uint16{'a', 'b', 'c'})
	})
	allowsPerRun(1, func() {
		syscall.UTF16FromString("abc")
	})
}

func TestNewFileInvalid(t *testing.T) {
	t.Parallel()
	if f := os.NewFile(uintptr(syscall.InvalidHandle), "invalid"); f != nil {
		t.Errorf("NewFile(InvalidHandle) got %v want nil", f)
	}
}

func TestReadDirPipe(t *testing.T) {
	dir := `\\.\pipe\`
	fi, err := os.Stat(dir)
	if err != nil || !fi.IsDir() {
		t.Skipf("%s is not a directory", dir)
	}
	_, err = os.ReadDir(dir)
	if err != nil {
		t.Errorf("ReadDir(%q) = %v", dir, err)
	}
}

func TestReadDirNoFileID(t *testing.T) {
	*os.AllowReadDirFileID = false
	defer func() { *os.AllowReadDirFileID = true }()

	dir := t.TempDir()
	pathA := filepath.Join(dir, "a")
	pathB := filepath.Join(dir, "b")
	if err := os.WriteFile(pathA, nil, 0666); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(pathB, nil, 0666); err != nil {
		t.Fatal(err)
	}

	files, err := os.ReadDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	if len(files) != 2 {
		t.Fatalf("ReadDir(%q) = %v; want 2 files", dir, files)
	}

	// Check that os.SameFile works with files returned by os.ReadDir.
	f1, err := files[0].Info()
	if err != nil {
		t.Fatal(err)
	}
	f2, err := files[1].Info()
	if err != nil {
		t.Fatal(err)
	}
	if !os.SameFile(f1, f1) {
		t.Errorf("SameFile(%v, %v) = false; want true", f1, f1)
	}
	if !os.SameFile(f2, f2) {
		t.Errorf("SameFile(%v, %v) = false; want true", f2, f2)
	}
	if os.SameFile(f1, f2) {
		t.Errorf("SameFile(%v, %v) = true; want false", f1, f2)
	}

	// Check that os.SameFile works with a mix of os.ReadDir and os.Stat files.
	f1s, err := os.Stat(pathA)
	if err != nil {
		t.Fatal(err)
	}
	f2s, err := os.Stat(pathB)
	if err != nil {
		t.Fatal(err)
	}
	if !os.SameFile(f1, f1s) {
		t.Errorf("SameFile(%v, %v) = false; want true", f1, f1s)
	}
	if !os.SameFile(f2, f2s) {
		t.Errorf("SameFile(%v, %v) = false; want true", f2, f2s)
	}
}

func TestReadWriteFileOverlapped(t *testing.T) {
	// See https://go.dev/issue/15388.
	t.Parallel()

	name := filepath.Join(t.TempDir(), "test.txt")
	wname, err := syscall.UTF16PtrFromString(name)
	if err != nil {
		t.Fatal(err)
	}
	h, err := syscall.CreateFile(wname, syscall.GENERIC_ALL, 0, nil, syscall.CREATE_NEW, syscall.FILE_ATTRIBUTE_NORMAL|syscall.FILE_FLAG_OVERLAPPED, 0)
	if err != nil {
		t.Fatal(err)
	}
	f := os.NewFile(uintptr(h), name)
	defer f.Close()

	data := []byte("test")
	n, err := f.Write(data)
	if err != nil {
		t.Fatal(err)
	}
	if n != len(data) {
		t.Fatalf("Write = %d; want %d", n, len(data))
	}

	if _, err := f.Seek(0, io.SeekStart); err != nil {
		t.Fatal(err)
	}

	got, err := io.ReadAll(f)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(got, data) {
		t.Fatalf("Read = %q; want %q", got, data)
	}
}

func TestStdinOverlappedPipe(t *testing.T) {
	// Test that we can read from a named pipe open with FILE_FLAG_OVERLAPPED.
	// See https://go.dev/issue/15388.
	if os.Getenv("GO_WANT_HELPER_PROCESS") == "1" {
		var buf string
		_, err := fmt.Scanln(&buf)
		if err != nil {
			fmt.Print(err)
			os.Exit(1)
		}
		fmt.Println(buf)
		os.Exit(0)
	}

	t.Parallel()
	name := pipeName()

	// Create the read handle inherited by the child process.
	r := newPipe(t, name, false, true)
	defer r.Close()

	// Create a write handle.
	w, err := os.OpenFile(name, os.O_WRONLY, 0666)
	if err != nil {
		t.Fatal(err)
	}
	defer w.Close()

	// Write some data to the pipe. The child process will read it.
	want := []byte("test\n")
	if _, err := w.Write(want); err != nil {
		t.Fatal(err)
	}

	// Create a child process that will read from the pipe
	// and write the data to stdout.
	cmd := testenv.Command(t, testenv.Executable(t), fmt.Sprintf("-test.run=^%s$", t.Name()), "-test.v")
	cmd = testenv.CleanCmdEnv(cmd)
	cmd.Env = append(cmd.Env, "GO_WANT_HELPER_PROCESS=1")
	cmd.Stdin = r
	got, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("running %q failed: %v\n%s", cmd, err, got)
	}

	if !bytes.Contains(got, want) {
		t.Fatalf("output %q does not contain %q", got, want)
	}
}

func newFileOverlapped(t testing.TB, name string, overlapped bool) *os.File {
	namep, err := syscall.UTF16PtrFromString(name)
	if err != nil {
		t.Fatal(err)
	}
	flags := syscall.FILE_ATTRIBUTE_NORMAL
	if overlapped {
		flags |= syscall.FILE_FLAG_OVERLAPPED
	}
	h, err := syscall.CreateFile(namep,
		syscall.GENERIC_READ|syscall.GENERIC_WRITE,
		syscall.FILE_SHARE_WRITE|syscall.FILE_SHARE_READ,
		nil, syscall.OPEN_ALWAYS, uint32(flags), 0)
	if err != nil {
		t.Fatal(err)
	}
	f := os.NewFile(uintptr(h), name)
	t.Cleanup(func() {
		if err := f.Close(); err != nil && !errors.Is(err, os.ErrClosed) {
			t.Fatal(err)
		}
	})
	return f
}

var currentProcess = sync.OnceValue(func() string {
	// Convert the process ID to a string.
	return strconv.FormatUint(uint64(os.Getpid()), 10)
})

var pipeCounter atomic.Uint64

func newBytePipe(t testing.TB, name string, overlapped bool) *os.File {
	return newPipe(t, name, false, overlapped)
}

func newMessagePipe(t testing.TB, name string, overlapped bool) *os.File {
	return newPipe(t, name, true, overlapped)
}

func pipeName() string {
	return `\\.\pipe\go-os-test-` + currentProcess() + `-` + strconv.FormatUint(pipeCounter.Add(1), 10)
}

func newPipe(t testing.TB, name string, message, overlapped bool) *os.File {
	wname, err := syscall.UTF16PtrFromString(name)
	if err != nil {
		t.Fatal(err)
	}
	// Create the read handle.
	flags := windows.PIPE_ACCESS_DUPLEX
	if overlapped {
		flags |= syscall.FILE_FLAG_OVERLAPPED
	}
	typ := windows.PIPE_TYPE_BYTE | windows.PIPE_READMODE_BYTE
	if message {
		typ = windows.PIPE_TYPE_MESSAGE | windows.PIPE_READMODE_MESSAGE
	}
	h, err := windows.CreateNamedPipe(wname, uint32(flags), uint32(typ), 1, 4096, 4096, 0, nil)
	if err != nil {
		t.Fatal(err)
	}
	f := os.NewFile(uintptr(h), name)
	t.Cleanup(func() {
		if err := f.Close(); err != nil && !errors.Is(err, os.ErrClosed) {
			t.Fatal(err)
		}
	})
	return f
}

func testReadWrite(t *testing.T, fdr, fdw *os.File) {
	write := make(chan string, 1)
	read := make(chan struct{}, 1)
	go func() {
		for s := range write {
			n, err := fdw.Write([]byte(s))
			read <- struct{}{}
			if err != nil {
				t.Error(err)
			}
			if n != len(s) {
				t.Errorf("expected to write %d bytes, got %d", len(s), n)
			}
		}
	}()
	for i := range 10 {
		s := strconv.Itoa(i)
		write <- s
		<-read
		buf := make([]byte, len(s))
		_, err := io.ReadFull(fdr, buf)
		if err != nil {
			t.Fatalf("read failed: %v", err)
		}
		if !bytes.Equal(buf, []byte(s)) {
			t.Fatalf("expected %q, got %q", s, buf)
		}
	}
	close(read)
	close(write)
}

func testPreadPwrite(t *testing.T, fdr, fdw *os.File) {
	type op struct {
		s   string
		off int64
	}
	write := make(chan op, 1)
	read := make(chan struct{}, 1)
	go func() {
		for o := range write {
			n, err := fdw.WriteAt([]byte(o.s), o.off)
			read <- struct{}{}
			if err != nil {
				t.Error(err)
			}
			if n != len(o.s) {
				t.Errorf("expected to write %d bytes, got %d", len(o.s), n)
			}
		}
	}()
	for i := range 10 {
		off := int64(i % 3) // exercise some back and forth
		s := strconv.Itoa(i)
		write <- op{s, off}
		<-read
		buf := make([]byte, len(s))
		n, err := fdr.ReadAt(buf, off)
		if err != nil {
			t.Fatal(err)
		}
		if n != len(s) {
			t.Fatalf("expected to read %d bytes, got %d", len(s), n)
		}
		if !bytes.Equal(buf, []byte(s)) {
			t.Fatalf("expected %q, got %q", s, buf)
		}
	}
	close(read)
	close(write)
}

func testFileReadEOF(t *testing.T, f *os.File) {
	end, err := f.Seek(0, io.SeekEnd)
	if err != nil {
		t.Fatal(err)
	}
	var buf [1]byte
	n, err := f.Read(buf[:])
	if err != nil && err != io.EOF {
		t.Errorf("expected EOF, got %v", err)
	}
	if n != 0 {
		t.Errorf("expected 0 bytes, got %d", n)
	}

	n, err = f.ReadAt(buf[:], end)
	if err != nil && err != io.EOF {
		t.Errorf("expected EOF, got %v", err)
	}
	if n != 0 {
		t.Errorf("expected 0 bytes, got %d", n)
	}
}

func TestFile(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name            string
		overlappedRead  bool
		overlappedWrite bool
	}{
		{"overlapped", true, true},
		{"overlapped-read", true, false},
		{"overlapped-write", false, true},
		{"sync", false, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			name := filepath.Join(t.TempDir(), "foo")
			rh := newFileOverlapped(t, name, tt.overlappedRead)
			wh := newFileOverlapped(t, name, tt.overlappedWrite)
			testReadWrite(t, rh, wh)
			testPreadPwrite(t, rh, wh)
			testFileReadEOF(t, rh)
		})
	}
}

func TestPipe(t *testing.T) {
	t.Parallel()
	r, w, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := r.Close(); err != nil {
			t.Fatal(err)
		}
		if err := w.Close(); err != nil {
			t.Fatal(err)
		}
	}()
	testReadWrite(t, r, w)
}

func TestNamedPipe(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name            string
		overlappedRead  bool
		overlappedWrite bool
	}{
		{"overlapped", true, true},
		{"overlapped-write", false, true},
		{"overlapped-read", true, false},
		{"sync", false, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			name := pipeName()
			pipe := newBytePipe(t, name, tt.overlappedWrite)
			file := newFileOverlapped(t, name, tt.overlappedRead)
			testReadWrite(t, pipe, file)
		})
	}
}

func TestPipeMessageReadEOF(t *testing.T) {
	t.Parallel()
	name := pipeName()
	pipe := newMessagePipe(t, name, true)
	file := newFileOverlapped(t, name, true)

	_, err := pipe.Write(nil)
	if err != nil {
		t.Error(err)
	}

	var buf [10]byte
	n, err := file.Read(buf[:])
	if err != io.EOF {
		t.Errorf("expected EOF, got %v", err)
	}
	if n != 0 {
		t.Errorf("expected 0 bytes, got %d", n)
	}
}

func TestPipeClosedEOF(t *testing.T) {
	t.Parallel()
	name := pipeName()
	pipe := newBytePipe(t, name, true)
	file := newFileOverlapped(t, name, true)

	pipe.Close()

	var buf [10]byte
	n, err := file.Read(buf[:])
	if err != io.EOF {
		t.Errorf("expected EOF, got %v", err)
	}
	if n != 0 {
		t.Errorf("expected 0 bytes, got %d", n)
	}
}

func TestPipeReadTimeout(t *testing.T) {
	t.Parallel()
	name := pipeName()
	_ = newBytePipe(t, name, true)
	file := newFileOverlapped(t, name, true)

	err := file.SetReadDeadline(time.Now().Add(time.Millisecond))
	if err != nil {
		t.Fatal(err)
	}

	var buf [10]byte
	_, err = file.Read(buf[:])
	if !errors.Is(err, os.ErrDeadlineExceeded) {
		t.Errorf("expected deadline exceeded, got %v", err)
	}
}

func TestPipeCanceled(t *testing.T) {
	t.Parallel()
	name := pipeName()
	_ = newBytePipe(t, name, true)
	file := newFileOverlapped(t, name, true)
	ch := make(chan struct{}, 1)
	go func() {
		for {
			select {
			case <-ch:
				return
			default:
				sc, err := file.SyscallConn()
				if err != nil {
					t.Error(err)
					return
				}
				if err := sc.Control(func(fd uintptr) {
					syscall.CancelIoEx(syscall.Handle(fd), nil)
				}); err != nil {
					t.Error(err)
					return
				}
				time.Sleep(100 * time.Millisecond)
			}
		}
	}()
	var tmp [1]byte
	// Read will block until the cancel is complete.
	_, err := file.Read(tmp[:])
	ch <- struct{}{}
	if errors.Is(err, os.ErrDeadlineExceeded) {
		t.Skip("took too long to cancel")
	}
	if !errors.Is(err, syscall.ERROR_OPERATION_ABORTED) {
		t.Errorf("expected ERROR_OPERATION_ABORTED, got %v", err)
	}
}

func TestPipeExternalIOCP(t *testing.T) {
	// Test that a caller can associate an overlapped handle to an external IOCP
	// even when the handle is also associated to a poll.FD. Also test that
	// the FD can still perform I/O after the association.
	t.Parallel()
	name := pipeName()
	pipe := newMessagePipe(t, name, true)
	_ = newFileOverlapped(t, name, true) // Just open a pipe client

	sc, err := pipe.SyscallConn()
	if err != nil {
		t.Error(err)
		return
	}
	if err := sc.Control(func(fd uintptr) {
		_, err := windows.CreateIoCompletionPort(syscall.Handle(fd), 0, 0, 1)
		if err != nil {
			t.Fatal(err)
		}
	}); err != nil {
		t.Error(err)
	}

	_, err = pipe.Write([]byte("hello"))
	if err != nil {
		t.Fatal(err)
	}
}
