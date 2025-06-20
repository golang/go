package main

import (
	"runtime"
	"testing"
)

// Test that proper truncation triggers a panic
func TestTruncationPanic(t *testing.T) {
	defer func() {
		if r := recover(); r != nil {
			// Check if the panic message contains "integer truncation"
			if err, ok := r.(error); ok {
				if err.Error() == "integer truncation" {
					t.Log("✓ Truncation detected correctly:", err.Error())
					return
				}
			}
			t.Fatalf("Expected 'integer truncation' panic, got: %v", r)
		} else {
			t.Fatal("Expected panic due to truncation, but no panic occurred")
		}
	}()

	// This should trigger a truncation panic
	var u16 uint16 = 256
	_ = uint8(u16) // This conversion truncates 256 to 0
}

// Test that safe conversions don't panic
func TestSafeConversion(t *testing.T) {
	defer func() {
		if r := recover(); r != nil {
			t.Fatalf("Unexpected panic during safe conversion: %v", r)
		}
	}()

	// These should NOT panic (values fit in target type)
	var u16 uint16 = 255
	u8 := uint8(u16)
	if u8 != 255 {
		t.Errorf("Expected u8=255, got u8=%d", u8)
	}

	var i32 int32 = 127
	i8 := int8(i32)
	if i8 != 127 {
		t.Errorf("Expected i8=127, got i8=%d", i8)
	}

	var i64 int64 = 32767
	i16 := int16(i64)
	if i16 != 32767 {
		t.Errorf("Expected i16=32767, got i16=%d", i16)
	}
}

// Test multiple truncation scenarios
func TestMultipleTruncationScenarios(t *testing.T) {
	scenarios := []struct {
		name     string
		testFunc func() interface{}
		expected string
	}{
		{
			name: "uint32 to uint16",
			testFunc: func() interface{} {
				var u32 uint32 = 65536
				return uint16(u32)
			},
			expected: "integer truncation",
		},
		{
			name: "int64 to int32",
			testFunc: func() interface{} {
				var i64 int64 = 2147483648
				return int32(i64)
			},
			expected: "integer truncation",
		},
		{
			name: "uint64 to uint8",
			testFunc: func() interface{} {
				var u64 uint64 = 256
				return uint8(u64)
			},
			expected: "integer truncation",
		},
	}

	for _, scenario := range scenarios {
		t.Run(scenario.name, func(t *testing.T) {
			defer func() {
				if r := recover(); r != nil {
					if err, ok := r.(error); ok {
						if err.Error() == scenario.expected {
							t.Logf("✓ %s: Truncation detected correctly", scenario.name)
							return
						}
					}
					t.Fatalf("%s: Expected '%s' panic, got: %v", scenario.name, scenario.expected, r)
				} else {
					t.Fatalf("%s: Expected panic, but no panic occurred", scenario.name)
				}
			}()

			scenario.testFunc()
		})
	}
}

// Benchmark to test performance impact
func BenchmarkTruncationCheck(b *testing.B) {
	defer func() {
		if r := recover(); r != nil {
			// Expected to panic, just continue benchmarking
		}
	}()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		func() {
			defer func() {
				recover() // Catch the panic to continue benchmarking
			}()
			var u16 uint16 = 256
			_ = uint8(u16)
		}()
	}
}

// Test stack trace contains useful information
func TestStackTrace(t *testing.T) {
	defer func() {
		if r := recover(); r != nil {
			// Get stack trace
			stackBuf := make([]byte, 1024)
			stack := runtime.Stack(stackBuf, false)
			stackStr := string(stack)

			// Verify stack trace contains our test function
			if contains(stackStr, "TestStackTrace") {
				t.Log("✓ Stack trace contains test function name")
			} else {
				t.Error("Stack trace missing test function name")
			}

			t.Logf("Stack trace:\n%s", stackStr)
			return
		}
		t.Fatal("Expected panic for stack trace test")
	}()

	var u32 uint32 = 65536
	_ = uint16(u32)
}

// Helper function to check if string contains substring
func contains(s, substr string) bool {
	return len(s) >= len(substr) && (s == substr || len(s) > len(substr) && (s[:len(substr)] == substr || s[len(s)-len(substr):] == substr || containsHelper(s, substr)))
}

func containsHelper(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}
