// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime_test

import (
	"bytes"
	"fmt"
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"syscall"
	"testing"
	"unsafe"
)

type DLL struct {
	*syscall.DLL
	t *testing.T
}

func GetDLL(t *testing.T, name string) *DLL {
	d, e := syscall.LoadDLL(name)
	if e != nil {
		t.Fatal(e)
	}
	return &DLL{DLL: d, t: t}
}

func (d *DLL) Proc(name string) *syscall.Proc {
	p, e := d.FindProc(name)
	if e != nil {
		d.t.Fatal(e)
	}
	return p
}

func TestStdCall(t *testing.T) {
	type Rect struct {
		left, top, right, bottom int32
	}
	res := Rect{}
	expected := Rect{1, 1, 40, 60}
	a, _, _ := GetDLL(t, "user32.dll").Proc("UnionRect").Call(
		uintptr(unsafe.Pointer(&res)),
		uintptr(unsafe.Pointer(&Rect{10, 1, 14, 60})),
		uintptr(unsafe.Pointer(&Rect{1, 2, 40, 50})))
	if a != 1 || res.left != expected.left ||
		res.top != expected.top ||
		res.right != expected.right ||
		res.bottom != expected.bottom {
		t.Error("stdcall USER32.UnionRect returns", a, "res=", res)
	}
}

func Test64BitReturnStdCall(t *testing.T) {

	const (
		VER_BUILDNUMBER      = 0x0000004
		VER_MAJORVERSION     = 0x0000002
		VER_MINORVERSION     = 0x0000001
		VER_PLATFORMID       = 0x0000008
		VER_PRODUCT_TYPE     = 0x0000080
		VER_SERVICEPACKMAJOR = 0x0000020
		VER_SERVICEPACKMINOR = 0x0000010
		VER_SUITENAME        = 0x0000040

		VER_EQUAL         = 1
		VER_GREATER       = 2
		VER_GREATER_EQUAL = 3
		VER_LESS          = 4
		VER_LESS_EQUAL    = 5

		ERROR_OLD_WIN_VERSION syscall.Errno = 1150
	)

	type OSVersionInfoEx struct {
		OSVersionInfoSize uint32
		MajorVersion      uint32
		MinorVersion      uint32
		BuildNumber       uint32
		PlatformId        uint32
		CSDVersion        [128]uint16
		ServicePackMajor  uint16
		ServicePackMinor  uint16
		SuiteMask         uint16
		ProductType       byte
		Reserve           byte
	}

	d := GetDLL(t, "kernel32.dll")

	var m1, m2 uintptr
	VerSetConditionMask := d.Proc("VerSetConditionMask")
	m1, m2, _ = VerSetConditionMask.Call(m1, m2, VER_MAJORVERSION, VER_GREATER_EQUAL)
	m1, m2, _ = VerSetConditionMask.Call(m1, m2, VER_MINORVERSION, VER_GREATER_EQUAL)
	m1, m2, _ = VerSetConditionMask.Call(m1, m2, VER_SERVICEPACKMAJOR, VER_GREATER_EQUAL)
	m1, m2, _ = VerSetConditionMask.Call(m1, m2, VER_SERVICEPACKMINOR, VER_GREATER_EQUAL)

	vi := OSVersionInfoEx{
		MajorVersion:     5,
		MinorVersion:     1,
		ServicePackMajor: 2,
		ServicePackMinor: 0,
	}
	vi.OSVersionInfoSize = uint32(unsafe.Sizeof(vi))
	r, _, e2 := d.Proc("VerifyVersionInfoW").Call(
		uintptr(unsafe.Pointer(&vi)),
		VER_MAJORVERSION|VER_MINORVERSION|VER_SERVICEPACKMAJOR|VER_SERVICEPACKMINOR,
		m1, m2)
	if r == 0 && e2 != ERROR_OLD_WIN_VERSION {
		t.Errorf("VerifyVersionInfo failed: %s", e2)
	}
}

func TestCDecl(t *testing.T) {
	var buf [50]byte
	fmtp, _ := syscall.BytePtrFromString("%d %d %d")
	a, _, _ := GetDLL(t, "user32.dll").Proc("wsprintfA").Call(
		uintptr(unsafe.Pointer(&buf[0])),
		uintptr(unsafe.Pointer(fmtp)),
		1000, 2000, 3000)
	if string(buf[:a]) != "1000 2000 3000" {
		t.Error("cdecl USER32.wsprintfA returns", a, "buf=", buf[:a])
	}
}

func TestEnumWindows(t *testing.T) {
	d := GetDLL(t, "user32.dll")
	isWindows := d.Proc("IsWindow")
	counter := 0
	cb := syscall.NewCallback(func(hwnd syscall.Handle, lparam uintptr) uintptr {
		if lparam != 888 {
			t.Error("lparam was not passed to callback")
		}
		b, _, _ := isWindows.Call(uintptr(hwnd))
		if b == 0 {
			t.Error("USER32.IsWindow returns FALSE")
		}
		counter++
		return 1 // continue enumeration
	})
	a, _, _ := d.Proc("EnumWindows").Call(cb, 888)
	if a == 0 {
		t.Error("USER32.EnumWindows returns FALSE")
	}
	if counter == 0 {
		t.Error("Callback has been never called or your have no windows")
	}
}

func callback(hwnd syscall.Handle, lparam uintptr) uintptr {
	(*(*func())(unsafe.Pointer(&lparam)))()
	return 0 // stop enumeration
}

// nestedCall calls into Windows, back into Go, and finally to f.
func nestedCall(t *testing.T, f func()) {
	c := syscall.NewCallback(callback)
	d := GetDLL(t, "user32.dll")
	defer d.Release()
	d.Proc("EnumWindows").Call(c, uintptr(*(*unsafe.Pointer)(unsafe.Pointer(&f))))
}

func TestCallback(t *testing.T) {
	var x = false
	nestedCall(t, func() { x = true })
	if !x {
		t.Fatal("nestedCall did not call func")
	}
}

func TestCallbackGC(t *testing.T) {
	nestedCall(t, runtime.GC)
}

func TestCallbackPanicLocked(t *testing.T) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	if !runtime.LockedOSThread() {
		t.Fatal("runtime.LockOSThread didn't")
	}
	defer func() {
		s := recover()
		if s == nil {
			t.Fatal("did not panic")
		}
		if s.(string) != "callback panic" {
			t.Fatal("wrong panic:", s)
		}
		if !runtime.LockedOSThread() {
			t.Fatal("lost lock on OS thread after panic")
		}
	}()
	nestedCall(t, func() { panic("callback panic") })
	panic("nestedCall returned")
}

func TestCallbackPanic(t *testing.T) {
	// Make sure panic during callback unwinds properly.
	if runtime.LockedOSThread() {
		t.Fatal("locked OS thread on entry to TestCallbackPanic")
	}
	defer func() {
		s := recover()
		if s == nil {
			t.Fatal("did not panic")
		}
		if s.(string) != "callback panic" {
			t.Fatal("wrong panic:", s)
		}
		if runtime.LockedOSThread() {
			t.Fatal("locked OS thread on exit from TestCallbackPanic")
		}
	}()
	nestedCall(t, func() { panic("callback panic") })
	panic("nestedCall returned")
}

func TestCallbackPanicLoop(t *testing.T) {
	// Make sure we don't blow out m->g0 stack.
	for i := 0; i < 100000; i++ {
		TestCallbackPanic(t)
	}
}

func TestBlockingCallback(t *testing.T) {
	c := make(chan int)
	go func() {
		for i := 0; i < 10; i++ {
			c <- <-c
		}
	}()
	nestedCall(t, func() {
		for i := 0; i < 10; i++ {
			c <- i
			if j := <-c; j != i {
				t.Errorf("out of sync %d != %d", j, i)
			}
		}
	})
}

func TestCallbackInAnotherThread(t *testing.T) {
	// TODO: test a function which calls back in another thread: QueueUserAPC() or CreateThread()
}

type cbDLLFunc int // int determines number of callback parameters

func (f cbDLLFunc) stdcallName() string {
	return fmt.Sprintf("stdcall%d", f)
}

func (f cbDLLFunc) cdeclName() string {
	return fmt.Sprintf("cdecl%d", f)
}

func (f cbDLLFunc) buildOne(stdcall bool) string {
	var funcname, attr string
	if stdcall {
		funcname = f.stdcallName()
		attr = "__stdcall"
	} else {
		funcname = f.cdeclName()
		attr = "__cdecl"
	}
	typename := "t" + funcname
	p := make([]string, f)
	for i := range p {
		p[i] = "uintptr_t"
	}
	params := strings.Join(p, ",")
	for i := range p {
		p[i] = fmt.Sprintf("%d", i+1)
	}
	args := strings.Join(p, ",")
	return fmt.Sprintf(`
typedef void %s (*%s)(%s);
void %s(%s f, uintptr_t n) {
	uintptr_t i;
	for(i=0;i<n;i++){
		f(%s);
	}
}
	`, attr, typename, params, funcname, typename, args)
}

func (f cbDLLFunc) build() string {
	return "#include <stdint.h>\n\n" + f.buildOne(false) + f.buildOne(true)
}

var cbFuncs = [...]interface{}{
	2: func(i1, i2 uintptr) uintptr {
		if i1+i2 != 3 {
			panic("bad input")
		}
		return 0
	},
	3: func(i1, i2, i3 uintptr) uintptr {
		if i1+i2+i3 != 6 {
			panic("bad input")
		}
		return 0
	},
	4: func(i1, i2, i3, i4 uintptr) uintptr {
		if i1+i2+i3+i4 != 10 {
			panic("bad input")
		}
		return 0
	},
	5: func(i1, i2, i3, i4, i5 uintptr) uintptr {
		if i1+i2+i3+i4+i5 != 15 {
			panic("bad input")
		}
		return 0
	},
	6: func(i1, i2, i3, i4, i5, i6 uintptr) uintptr {
		if i1+i2+i3+i4+i5+i6 != 21 {
			panic("bad input")
		}
		return 0
	},
	7: func(i1, i2, i3, i4, i5, i6, i7 uintptr) uintptr {
		if i1+i2+i3+i4+i5+i6+i7 != 28 {
			panic("bad input")
		}
		return 0
	},
	8: func(i1, i2, i3, i4, i5, i6, i7, i8 uintptr) uintptr {
		if i1+i2+i3+i4+i5+i6+i7+i8 != 36 {
			panic("bad input")
		}
		return 0
	},
	9: func(i1, i2, i3, i4, i5, i6, i7, i8, i9 uintptr) uintptr {
		if i1+i2+i3+i4+i5+i6+i7+i8+i9 != 45 {
			panic("bad input")
		}
		return 0
	},
}

type cbDLL struct {
	name      string
	buildArgs func(out, src string) []string
}

func (d *cbDLL) buildSrc(t *testing.T, path string) {
	f, err := os.Create(path)
	if err != nil {
		t.Fatalf("failed to create source file: %v", err)
	}
	defer f.Close()

	for i := 2; i < 10; i++ {
		fmt.Fprint(f, cbDLLFunc(i).build())
	}
}

func (d *cbDLL) build(t *testing.T, dir string) string {
	srcname := d.name + ".c"
	d.buildSrc(t, filepath.Join(dir, srcname))
	outname := d.name + ".dll"
	args := d.buildArgs(outname, srcname)
	cmd := exec.Command(args[0], args[1:]...)
	cmd.Dir = dir
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("failed to build dll: %v - %v", err, string(out))
	}
	return filepath.Join(dir, outname)
}

var cbDLLs = []cbDLL{
	{
		"test",
		func(out, src string) []string {
			return []string{"gcc", "-shared", "-s", "-Werror", "-o", out, src}
		},
	},
	{
		"testO2",
		func(out, src string) []string {
			return []string{"gcc", "-shared", "-s", "-Werror", "-o", out, "-O2", src}
		},
	},
}

type cbTest struct {
	n     int     // number of callback parameters
	param uintptr // dll function parameter
}

func (test *cbTest) run(t *testing.T, dllpath string) {
	dll := syscall.MustLoadDLL(dllpath)
	defer dll.Release()
	cb := cbFuncs[test.n]
	stdcall := syscall.NewCallback(cb)
	f := cbDLLFunc(test.n)
	test.runOne(t, dll, f.stdcallName(), stdcall)
	cdecl := syscall.NewCallbackCDecl(cb)
	test.runOne(t, dll, f.cdeclName(), cdecl)
}

func (test *cbTest) runOne(t *testing.T, dll *syscall.DLL, proc string, cb uintptr) {
	defer func() {
		if r := recover(); r != nil {
			t.Errorf("dll call %v(..., %d) failed: %v", proc, test.param, r)
		}
	}()
	dll.MustFindProc(proc).Call(cb, test.param)
}

var cbTests = []cbTest{
	{2, 1},
	{2, 10000},
	{3, 3},
	{4, 5},
	{4, 6},
	{5, 2},
	{6, 7},
	{6, 8},
	{7, 6},
	{8, 1},
	{9, 8},
	{9, 10000},
	{3, 4},
	{5, 3},
	{7, 7},
	{8, 2},
	{9, 9},
}

func TestStdcallAndCDeclCallbacks(t *testing.T) {
	if _, err := exec.LookPath("gcc"); err != nil {
		t.Skip("skipping test: gcc is missing")
	}
	tmp, err := ioutil.TempDir("", "TestCDeclCallback")
	if err != nil {
		t.Fatal("TempDir failed: ", err)
	}
	defer os.RemoveAll(tmp)

	for _, dll := range cbDLLs {
		dllPath := dll.build(t, tmp)
		for _, test := range cbTests {
			test.run(t, dllPath)
		}
	}
}

func TestRegisterClass(t *testing.T) {
	kernel32 := GetDLL(t, "kernel32.dll")
	user32 := GetDLL(t, "user32.dll")
	mh, _, _ := kernel32.Proc("GetModuleHandleW").Call(0)
	cb := syscall.NewCallback(func(hwnd syscall.Handle, msg uint32, wparam, lparam uintptr) (rc uintptr) {
		t.Fatal("callback should never get called")
		return 0
	})
	type Wndclassex struct {
		Size       uint32
		Style      uint32
		WndProc    uintptr
		ClsExtra   int32
		WndExtra   int32
		Instance   syscall.Handle
		Icon       syscall.Handle
		Cursor     syscall.Handle
		Background syscall.Handle
		MenuName   *uint16
		ClassName  *uint16
		IconSm     syscall.Handle
	}
	name := syscall.StringToUTF16Ptr("test_window")
	wc := Wndclassex{
		WndProc:   cb,
		Instance:  syscall.Handle(mh),
		ClassName: name,
	}
	wc.Size = uint32(unsafe.Sizeof(wc))
	a, _, err := user32.Proc("RegisterClassExW").Call(uintptr(unsafe.Pointer(&wc)))
	if a == 0 {
		t.Fatalf("RegisterClassEx failed: %v", err)
	}
	r, _, err := user32.Proc("UnregisterClassW").Call(uintptr(unsafe.Pointer(name)), 0)
	if r == 0 {
		t.Fatalf("UnregisterClass failed: %v", err)
	}
}

func TestOutputDebugString(t *testing.T) {
	d := GetDLL(t, "kernel32.dll")
	p := syscall.StringToUTF16Ptr("testing OutputDebugString")
	d.Proc("OutputDebugStringW").Call(uintptr(unsafe.Pointer(p)))
}

func TestRaiseException(t *testing.T) {
	o := runTestProg(t, "testprog", "RaiseException")
	if strings.Contains(o, "RaiseException should not return") {
		t.Fatalf("RaiseException did not crash program: %v", o)
	}
	if !strings.Contains(o, "Exception 0xbad") {
		t.Fatalf("No stack trace: %v", o)
	}
}

func TestZeroDivisionException(t *testing.T) {
	o := runTestProg(t, "testprog", "ZeroDivisionException")
	if !strings.Contains(o, "panic: runtime error: integer divide by zero") {
		t.Fatalf("No stack trace: %v", o)
	}
}

func TestWERDialogue(t *testing.T) {
	if os.Getenv("TESTING_WER_DIALOGUE") == "1" {
		defer os.Exit(0)

		*runtime.TestingWER = true
		const EXCEPTION_NONCONTINUABLE = 1
		mod := syscall.MustLoadDLL("kernel32.dll")
		proc := mod.MustFindProc("RaiseException")
		proc.Call(0xbad, EXCEPTION_NONCONTINUABLE, 0, 0)
		println("RaiseException should not return")
		return
	}
	cmd := exec.Command(os.Args[0], "-test.run=TestWERDialogue")
	cmd.Env = []string{"TESTING_WER_DIALOGUE=1"}
	// Child process should not open WER dialogue, but return immediately instead.
	cmd.CombinedOutput()
}

var used byte

func use(buf []byte) {
	for _, c := range buf {
		used += c
	}
}

func forceStackCopy() (r int) {
	var f func(int) int
	f = func(i int) int {
		var buf [256]byte
		use(buf[:])
		if i == 0 {
			return 0
		}
		return i + f(i-1)
	}
	r = f(128)
	return
}

func TestReturnAfterStackGrowInCallback(t *testing.T) {
	if _, err := exec.LookPath("gcc"); err != nil {
		t.Skip("skipping test: gcc is missing")
	}

	const src = `
#include <stdint.h>
#include <windows.h>

typedef uintptr_t __stdcall (*callback)(uintptr_t);

uintptr_t cfunc(callback f, uintptr_t n) {
   uintptr_t r;
   r = f(n);
   SetLastError(333);
   return r;
}
`
	tmpdir, err := ioutil.TempDir("", "TestReturnAfterStackGrowInCallback")
	if err != nil {
		t.Fatal("TempDir failed: ", err)
	}
	defer os.RemoveAll(tmpdir)

	srcname := "mydll.c"
	err = ioutil.WriteFile(filepath.Join(tmpdir, srcname), []byte(src), 0)
	if err != nil {
		t.Fatal(err)
	}
	outname := "mydll.dll"
	cmd := exec.Command("gcc", "-shared", "-s", "-Werror", "-o", outname, srcname)
	cmd.Dir = tmpdir
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("failed to build dll: %v - %v", err, string(out))
	}
	dllpath := filepath.Join(tmpdir, outname)

	dll := syscall.MustLoadDLL(dllpath)
	defer dll.Release()

	proc := dll.MustFindProc("cfunc")

	cb := syscall.NewCallback(func(n uintptr) uintptr {
		forceStackCopy()
		return n
	})

	// Use a new goroutine so that we get a small stack.
	type result struct {
		r   uintptr
		err syscall.Errno
	}
	c := make(chan result)
	go func() {
		r, _, err := proc.Call(cb, 100)
		c <- result{r, err.(syscall.Errno)}
	}()
	want := result{r: 100, err: 333}
	if got := <-c; got != want {
		t.Errorf("got %d want %d", got, want)
	}
}

// removeOneCPU removes one (any) cpu from affinity mask.
// It returns new affinity mask.
func removeOneCPU(mask uintptr) (uintptr, error) {
	if mask == 0 {
		return 0, fmt.Errorf("cpu affinity mask is empty")
	}
	maskbits := int(unsafe.Sizeof(mask) * 8)
	for i := 0; i < maskbits; i++ {
		newmask := mask & ^(1 << uint(i))
		if newmask != mask {
			return newmask, nil
		}

	}
	panic("not reached")
}

func resumeChildThread(kernel32 *syscall.DLL, childpid int) error {
	_OpenThread := kernel32.MustFindProc("OpenThread")
	_ResumeThread := kernel32.MustFindProc("ResumeThread")
	_Thread32First := kernel32.MustFindProc("Thread32First")
	_Thread32Next := kernel32.MustFindProc("Thread32Next")

	snapshot, err := syscall.CreateToolhelp32Snapshot(syscall.TH32CS_SNAPTHREAD, 0)
	if err != nil {
		return err
	}
	defer syscall.CloseHandle(snapshot)

	const _THREAD_SUSPEND_RESUME = 0x0002

	type ThreadEntry32 struct {
		Size           uint32
		tUsage         uint32
		ThreadID       uint32
		OwnerProcessID uint32
		BasePri        int32
		DeltaPri       int32
		Flags          uint32
	}

	var te ThreadEntry32
	te.Size = uint32(unsafe.Sizeof(te))
	ret, _, err := _Thread32First.Call(uintptr(snapshot), uintptr(unsafe.Pointer(&te)))
	if ret == 0 {
		return err
	}
	for te.OwnerProcessID != uint32(childpid) {
		ret, _, err = _Thread32Next.Call(uintptr(snapshot), uintptr(unsafe.Pointer(&te)))
		if ret == 0 {
			return err
		}
	}
	h, _, err := _OpenThread.Call(_THREAD_SUSPEND_RESUME, 1, uintptr(te.ThreadID))
	if h == 0 {
		return err
	}
	defer syscall.Close(syscall.Handle(h))

	ret, _, err = _ResumeThread.Call(h)
	if ret == 0xffffffff {
		return err
	}
	return nil
}

func TestNumCPU(t *testing.T) {
	if os.Getenv("GO_WANT_HELPER_PROCESS") == "1" {
		// in child process
		fmt.Fprintf(os.Stderr, "%d", runtime.NumCPU())
		os.Exit(0)
	}

	switch n := runtime.NumberOfProcessors(); {
	case n < 1:
		t.Fatalf("system cannot have %d cpu(s)", n)
	case n == 1:
		if runtime.NumCPU() != 1 {
			t.Fatalf("runtime.NumCPU() returns %d on single cpu system", runtime.NumCPU())
		}
		return
	}

	const (
		_CREATE_SUSPENDED   = 0x00000004
		_PROCESS_ALL_ACCESS = syscall.STANDARD_RIGHTS_REQUIRED | syscall.SYNCHRONIZE | 0xfff
	)

	kernel32 := syscall.MustLoadDLL("kernel32.dll")
	_GetProcessAffinityMask := kernel32.MustFindProc("GetProcessAffinityMask")
	_SetProcessAffinityMask := kernel32.MustFindProc("SetProcessAffinityMask")

	cmd := exec.Command(os.Args[0], "-test.run=TestNumCPU")
	cmd.Env = append(os.Environ(), "GO_WANT_HELPER_PROCESS=1")
	var buf bytes.Buffer
	cmd.Stdout = &buf
	cmd.Stderr = &buf
	cmd.SysProcAttr = &syscall.SysProcAttr{CreationFlags: _CREATE_SUSPENDED}
	err := cmd.Start()
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		err = cmd.Wait()
		childOutput := string(buf.Bytes())
		if err != nil {
			t.Fatalf("child failed: %v: %v", err, childOutput)
		}
		// removeOneCPU should have decreased child cpu count by 1
		want := fmt.Sprintf("%d", runtime.NumCPU()-1)
		if childOutput != want {
			t.Fatalf("child output: want %q, got %q", want, childOutput)
		}
	}()

	defer func() {
		err = resumeChildThread(kernel32, cmd.Process.Pid)
		if err != nil {
			t.Fatal(err)
		}
	}()

	ph, err := syscall.OpenProcess(_PROCESS_ALL_ACCESS, false, uint32(cmd.Process.Pid))
	if err != nil {
		t.Fatal(err)
	}
	defer syscall.CloseHandle(ph)

	var mask, sysmask uintptr
	ret, _, err := _GetProcessAffinityMask.Call(uintptr(ph), uintptr(unsafe.Pointer(&mask)), uintptr(unsafe.Pointer(&sysmask)))
	if ret == 0 {
		t.Fatal(err)
	}

	newmask, err := removeOneCPU(mask)
	if err != nil {
		t.Fatal(err)
	}

	ret, _, err = _SetProcessAffinityMask.Call(uintptr(ph), newmask)
	if ret == 0 {
		t.Fatal(err)
	}
	ret, _, err = _GetProcessAffinityMask.Call(uintptr(ph), uintptr(unsafe.Pointer(&mask)), uintptr(unsafe.Pointer(&sysmask)))
	if ret == 0 {
		t.Fatal(err)
	}
	if newmask != mask {
		t.Fatalf("SetProcessAffinityMask didn't set newmask of 0x%x. Current mask is 0x%x.", newmask, mask)
	}
}
