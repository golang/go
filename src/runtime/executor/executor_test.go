// Test scaffolding shared by behavioral tests.

package executor_test

import (
	"runtime"
	"testing"
	"unsafe"
	_ "unsafe" // for go:linkname
)

// currentTID returns an opaque uintptr that identifies the OS
// thread the calling goroutine is currently running on. It uses
// the runtime's *m pointer (one *m per OS thread) so the value is
// stable across goroutine switches that stay on the same thread —
// in particular, across coroswitch transfers between an executor
// task and the goroutine that switched into it.
//
// The caller should hold the goroutine to its thread via
// runtime.LockOSThread for cross-frame comparisons to be
// meaningful.
func currentTID() uintptr {
	return uintptr(runtime_test_execCurM())
}

//go:linkname runtime_test_execCurM runtime.execCurM
func runtime_test_execCurM() unsafe.Pointer

// assertNumGoroutineUnchanged runs body and fails the test if
// runtime.NumGoroutine() differs before and after.
func assertNumGoroutineUnchanged(t *testing.T, body func()) {
	t.Helper()
	runtime.GC()
	before := runtime.NumGoroutine()
	body()
	runtime.GC()
	after := runtime.NumGoroutine()
	if delta := after - before; delta != 0 {
		t.Fatalf("runtime.NumGoroutine() changed by %d (before=%d after=%d); want 0", delta, before, after)
	}
}
