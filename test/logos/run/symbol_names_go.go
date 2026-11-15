// run

package main

import (
	"fmt"
	"os"
	"os/exec"
	"reflect"
	"runtime"
	"strings"
)

func fail(format string, args ...any) {
	fmt.Printf("FAIL "+format+"\n", args...)
	os.Exit(1)
}

func checkFuncForPC(want string, v any) {
	fn := runtime.FuncForPC(reflect.ValueOf(v).Pointer())
	if fn == nil {
		fail("FuncForPC nil for %s", want)
	}
	if got := fn.Name(); got != want {
		fail("FuncForPC = %q want %q", got, want)
	}
}

func checkFuncForPCContains(substr string, v any) {
	fn := runtime.FuncForPC(reflect.ValueOf(v).Pointer())
	if fn == nil {
		fail("FuncForPC nil for %s", substr)
	}
	if got := fn.Name(); !strings.Contains(got, substr) {
		fail("FuncForPC = %q missing %q", got, substr)
	}
}

func checkFramesEqual(want string) {
	var pcs [16]uintptr
	n := runtime.Callers(0, pcs[:])
	frames := runtime.CallersFrames(pcs[:n])
	for {
		fr, more := frames.Next()
		if fr.Function == want {
			return
		}
		if !more {
			break
		}
	}
	fail("CallersFrames missing %q", want)
}

func checkFramesContain(substr string) {
	var pcs [16]uintptr
	n := runtime.Callers(0, pcs[:])
	frames := runtime.CallersFrames(pcs[:n])
	for {
		fr, more := frames.Next()
		if strings.Contains(fr.Function, substr) {
			return
		}
		if !more {
			break
		}
	}
	fail("CallersFrames missing function containing %q", substr)
}

//go:noinline
func f_logosct1() {
	if os.Getenv("CHILD") == "1" {
		panic("boom")
	}
	checkFramesEqual("main.f_logosct1")
}

//go:noinline
func f_logosct007() {
	checkFramesEqual("main.f_logosct007")
}

//go:noinline
func g_logosct1_logosct2() {
	checkFramesEqual("main.g_logosct1_logosct2")
}

//go:noinline
func _logosct9() {
	checkFramesEqual("main._logosct9")
}

type T struct{}

//go:noinline
func (T) m_logosct3() {
	checkFramesEqual("main.T.m_logosct3")
}

//go:noinline
func h_logosct4() func() {
	return func() {
		checkFramesContain("h_logosct4")
	}
}

//go:noinline
func k_logosct5[T any](v T) T {
	checkFramesContain("k_logosct5")
	return v
}

func checkPanicTrace() {
	exe, err := os.Executable()
	if err != nil {
		fail("Executable: %v", err)
	}
	cmd := exec.Command(exe)
	cmd.Env = append(os.Environ(), "CHILD=1")
	out, err := cmd.CombinedOutput()
	if err == nil {
		fail("child exit 0 want panic")
	}
	if !strings.Contains(string(out), "panic: boom") {
		fail("child stderr missing panic header:\n%s", out)
	}
	if !strings.Contains(string(out), "main.f_logosct1(") {
		fail("panic traceback mangled, want main.f_logosct1:\n%s", out)
	}
}

func main() {
	if os.Getenv("CHILD") == "1" {
		f_logosct1()
		return
	}
	checkFuncForPC("main.f_logosct1", f_logosct1)
	checkFuncForPC("main.f_logosct007", f_logosct007)
	checkFuncForPC("main.g_logosct1_logosct2", g_logosct1_logosct2)
	checkFuncForPC("main._logosct9", _logosct9)
	checkFuncForPCContains("T.m_logosct3", T{}.m_logosct3)

	f_logosct1()
	f_logosct007()
	g_logosct1_logosct2()
	_logosct9()
	T{}.m_logosct3()
	h_logosct4()()
	if k_logosct5[int](41) != 41 {
		fail("generic call wrong value")
	}
	if k_logosct5[string]("s") != "s" {
		fail("generic call wrong value")
	}
	checkPanicTrace()
	fmt.Println("ok symbols")
}
