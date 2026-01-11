package tests

import (
	"strings"
	"testing"
	errorpkg "unit_test/error"
)

// Function that does not use errpkg.Log (should never panic from paniconcall).
func testnoLog() {}

func TestUsesLog(t *testing.T) {
	defer func() {
		r := recover()
		if r == nil {
			t.Skip("panic-on-call not enabled; run with -gcflags=all='-paniconcall=unit_test/error.Log'")
			return
		}
		// Verify it's the panic-on-call panic
		msg := r.(error).Error()
		if strings.Contains(msg, "panic-on-call: unit_test/error.Log") {
			t.Logf("✓ Successfully triggered panic-on-call for unit_test/error.Log: %s", msg)
		} else {
			t.Fatalf("Unexpected panic: %v", r)
		}
	}()
	errorpkg.Log()
}

func TestNoLogNeverPanics(t *testing.T) {
	defer func() {
		if r := recover(); r != nil {
			t.Fatalf("Unexpected panic in testnoLog: %v", r)
		}
	}()
	testnoLog()
}
