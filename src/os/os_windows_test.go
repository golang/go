// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package os_test

import (
	"fmt"
	"internal/syscall/windows"
	"internal/testenv"
	"io"
	"io/ioutil"
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

func TestSameWindowsFile(t *testing.T) {
	temp, err := ioutil.TempDir("", "TestSameWindowsFile")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(temp)

	wd, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	err = os.Chdir(temp)
	if err != nil {
		t.Fatal(err)
	}
	defer os.Chdir(wd)

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
	tmpdir, err := ioutil.TempDir("", "testDirLinks")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmpdir)

	oldwd, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	err = os.Chdir(tmpdir)
	if err != nil {
		t.Fatal(err)
	}
	defer os.Chdir(oldwd)

	dir := filepath.Join(tmpdir, "dir")
	err = os.Mkdir(dir, 0777)
	if err != nil {
		t.Fatal(err)
	}
	err = ioutil.WriteFile(filepath.Join(dir, "abc"), []byte("abc"), 0644)
	if err != nil {
		t.Fatal(err)
	}
	for _, test := range tests {
		link := filepath.Join(tmpdir, test.name+"_link")
		err := test.mklink(link, dir)
		if err != nil {
			t.Errorf("creating link for %s test failed: %v", test.name, err)
			continue
		}

		data, err := ioutil.ReadFile(filepath.Join(link, "abc"))
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

		fi, err := os.Stat(link)
		if err != nil {
			t.Errorf("failed to stat link %v: %v", link, err)
			continue
		}
		expected := filepath.Base(dir)
		got := fi.Name()
		if !fi.IsDir() || expected != got {
			t.Errorf("link should point to %v but points to %v instead", expected, got)
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
	return rd.addUTF16s(p), uint16(len(p)-1) * 2 // do not include terminating NUL in the legth (as per PrintNameLength and SubstituteNameLength documentation)
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
	copy((*[2048]uint16)(unsafe.Pointer(&buf.PathBuffer[0]))[:], target.pathBuf)

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
	copy((*[2048]uint16)(unsafe.Pointer(&buf.PathBuffer[0]))[:], target.pathBuf)

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

	dir, err := ioutil.TempDir("", "TestNetworkSymbolicLink")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(dir)

	oldwd, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	err = os.Chdir(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer os.Chdir(oldwd)

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
	tmpdir, err := ioutil.TempDir("", "TestOpenVolumeName")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmpdir)

	wd, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	err = os.Chdir(tmpdir)
	if err != nil {
		t.Fatal(err)
	}
	defer os.Chdir(wd)

	want := []string{"file1", "file2", "file3", "gopher.txt"}
	sort.Strings(want)
	for _, name := range want {
		err := ioutil.WriteFile(filepath.Join(tmpdir, name), nil, 0777)
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
	tmpdir, err := ioutil.TempDir("", "TestDeleteReadOnly")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmpdir)
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
	if perr, ok := err.(*os.PathError); !ok || perr.Err != syscall.ELOOP {
		t.Errorf("expected *PathError with ELOOP, got %T: %v\n", err, err)
	}
}

func TestReadStdin(t *testing.T) {
	old := *os.ReadConsoleFunc
	defer func() {
		*os.ReadConsoleFunc = old
	}()

	testConsole := os.NewConsoleFile(syscall.Stdin, "test")

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
					*os.ReadConsoleFunc = func(h syscall.Handle, buf *uint16, toread uint32, read *uint32, inputControl *byte) error {
						if inputControl != nil {
							t.Fatalf("inputControl not nil")
						}
						n := int(toread)
						if n > consoleSize {
							n = consoleSize
						}
						n = copy((*[10000]uint16)(unsafe.Pointer(buf))[:n], s16)
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
	_, err := os.Stat(`c:\pagefile.sys`)
	if err == nil {
		return
	}
	if os.IsNotExist(err) {
		t.Skip(`skipping because c:\pagefile.sys is not found`)
	}
	t.Fatal(err)
}
