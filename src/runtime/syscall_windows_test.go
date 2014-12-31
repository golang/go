// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime_test

import (
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
	o := executeTest(t, raiseExceptionSource, nil)
	if strings.Contains(o, "RaiseException should not return") {
		t.Fatalf("RaiseException did not crash program: %v", o)
	}
	if !strings.Contains(o, "Exception 0xbad") {
		t.Fatalf("No stack trace: %v", o)
	}
}

const raiseExceptionSource = `
package main
import "syscall"
func main() {
	const EXCEPTION_NONCONTINUABLE = 1
	mod := syscall.MustLoadDLL("kernel32.dll")
	proc := mod.MustFindProc("RaiseException")
	proc.Call(0xbad, EXCEPTION_NONCONTINUABLE, 0, 0)
	println("RaiseException should not return")
}
`

func TestZeroDivisionException(t *testing.T) {
	o := executeTest(t, zeroDivisionExceptionSource, nil)
	if !strings.Contains(o, "panic: runtime error: integer divide by zero") {
		t.Fatalf("No stack trace: %v", o)
	}
}

const zeroDivisionExceptionSource = `
package main
func main() {
	x := 1
	y := 0
	z := x / y
	println(z)
}
`

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
