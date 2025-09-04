package tests

import (
	"testing"
)

// isTruncationDetectionEnabled checks if truncation detection is enabled by attempting 
// a truncation that should panic if detection is enabled
func isTruncationDetectionEnabled() bool {
	panicked := false
	func() {
		defer func() {
			if recover() != nil {
				panicked = true
			}
		}()
		
		// Try a simple truncation that should trigger detection if enabled
		var test uint16 = 256
		_ = uint8(test) // This should panic if truncation detection is on
	}()
	
	return panicked
}

// skipIfTruncationDisabled skips the test if truncation detection is disabled
func skipIfTruncationDisabled(t *testing.T) {
	if !isTruncationDetectionEnabled() {
		t.Skip("Skipping truncation test - truncation detection is disabled")
	}
}

func TestInt64ToInt32Overflow(t *testing.T) {
	skipIfTruncationDisabled(t)
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("Expected panic for int64 to int32 overflow")
		}
	}()
	var large int64 = 0x100000000
	_ = int32(large)
}

func TestInt64ToInt32Underflow(t *testing.T) {
	skipIfTruncationDisabled(t)
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("Expected panic for int64 to int32 underflow")
		}
	}()
	var large int64 = -0x100000000
	_ = int32(large)
}

func TestInt32ToInt16Overflow(t *testing.T) {
	skipIfTruncationDisabled(t)
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("Expected panic for int32 to int16 overflow")
		}
	}()
	var large int32 = 0x10000
	_ = int16(large)
}

func TestInt32ToInt16Underflow(t *testing.T) {
	skipIfTruncationDisabled(t)
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("Expected panic for int32 to int16 underflow")
		}
	}()
	var large int32 = -0x10000
	_ = int16(large)
}

func TestInt16ToInt8Overflow(t *testing.T) {
	skipIfTruncationDisabled(t)
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("Expected panic for int16 to int8 overflow")
		}
	}()
	var large int16 = 0x100
	_ = int8(large)
}

func TestInt16ToInt8Underflow(t *testing.T) {
	skipIfTruncationDisabled(t)
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("Expected panic for int16 to int8 underflow")
		}
	}()
	var large int16 = -0x100
	_ = int8(large)
}

func TestUint64ToUint32Overflow(t *testing.T) {
	skipIfTruncationDisabled(t)
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("Expected panic for uint64 to uint32 overflow")
		}
	}()
	var large uint64 = 0x100000000
	_ = uint32(large)
}

func TestUint32ToUint16Overflow(t *testing.T) {
	skipIfTruncationDisabled(t)
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("Expected panic for uint32 to uint16 overflow")
		}
	}()
	var large uint32 = 0x10000
	_ = uint16(large)
}

func TestUint16ToUint8Overflow(t *testing.T) {
	skipIfTruncationDisabled(t)
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("Expected panic for uint16 to uint8 overflow")
		}
	}()
	var large uint16 = 0x100
	_ = uint8(large)
}

func TestIntToInt32OnLargeValues(t *testing.T) {
	skipIfTruncationDisabled(t)
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("Expected panic for int to int32 on large values")
		}
	}()
	var large int = 0x100000000
	_ = int32(large)
}

func TestIntToInt16OnLargeValues(t *testing.T) {
	skipIfTruncationDisabled(t)
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("Expected panic for int to int16 on large values")
		}
	}()
	var large int = 0x10000
	_ = int16(large)
}

func TestIntToInt8OnLargeValues(t *testing.T) {
	skipIfTruncationDisabled(t)
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("Expected panic for int to int8 on large values")
		}
	}()
	var large int = 0x100
	_ = int8(large)
}

func TestUintToUint32OnLargeValues(t *testing.T) {
	skipIfTruncationDisabled(t)
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("Expected panic for uint to uint32 on large values")
		}
	}()
	var large uint = 0x100000000
	_ = uint32(large)
}

func TestUintToUint16OnLargeValues(t *testing.T) {
	skipIfTruncationDisabled(t)
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("Expected panic for uint to uint16 on large values")
		}
	}()
	var large uint = 0x10000
	_ = uint16(large)
}

func TestUintToUint8OnLargeValues(t *testing.T) {
	skipIfTruncationDisabled(t)
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("Expected panic for uint to uint8 on large values")
		}
	}()
	var large uint = 0x100
	_ = uint8(large)
}

func TestSignedToUnsignedNegative(t *testing.T) {
	skipIfTruncationDisabled(t)
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("Expected panic for signed to unsigned with negative values")
		}
	}()
	var negative int32 = -1
	_ = uint32(negative)
}

func TestUnsignedToSigned(t *testing.T) {
	skipIfTruncationDisabled(t)
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("Expected panic for unsigned to signed with large values")
		}
	}()
	var unsigned uint32 = 0xFFFFFFFF
	_ = int32(unsigned)
}

func TestInt16ToUint16Negative(t *testing.T) {
	skipIfTruncationDisabled(t)
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("Expected panic for int16 to uint16 with negative values")
		}
	}()
	var negative int16 = -1
	_ = uint16(negative)
}

func TestInt8ToUint8Negative(t *testing.T) {
	skipIfTruncationDisabled(t)
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("Expected panic for int8 to uint8 with negative values")
		}
	}()
	var negative int8 = -1
	_ = uint8(negative)
}

func TestComplexTruncationChain(t *testing.T) {
	skipIfTruncationDisabled(t)
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("Expected panic for complex truncation chain")
		}
	}()
	var start int64 = 0x123456789ABCDEF
	var step1 int32 = int32(start)
	var step2 int16 = int16(step1)
	_ = int8(step2)
}

func TestRuntimeComputedTruncation(t *testing.T) {
	skipIfTruncationDisabled(t)
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("Expected panic for runtime computed truncation")
		}
	}()
	var base int64 = 1000
	for i := 0; i < 10; i++ {
		base = base * 10
	}
	_ = int32(base)
}

func TestBufferSizeVulnerability(t *testing.T) {
	skipIfTruncationDisabled(t)
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("Expected panic for buffer size vulnerability")
		}
	}()
	var requestedSize int64 = 0x200000000
	var actualSize int32 = int32(requestedSize)
	_ = actualSize
}

func TestArrayIndexTruncation(t *testing.T) {
	skipIfTruncationDisabled(t)
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("Expected panic for array index truncation")
		}
	}()
	var largeIndex int64 = 0x80000000
	var truncatedIndex int32 = int32(largeIndex)
	_ = truncatedIndex
}

func TestMemoryOffsetTruncation(t *testing.T) {
	skipIfTruncationDisabled(t)
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("Expected panic for memory offset truncation")
		}
	}()
	var offset int64 = 0x180000000
	var truncatedOffset int32 = int32(offset)
	_ = truncatedOffset
}

func TestSecurityBoundaryTruncation(t *testing.T) {
	skipIfTruncationDisabled(t)
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("Expected panic for security boundary truncation")
		}
	}()
	var securityLimit int64 = 0x7FFFFFFF + 1000
	var checkedLimit int32 = int32(securityLimit)
	_ = checkedLimit
}

// Platform-dependent truncation edge cases
func TestPlatformDependentIntTruncation(t *testing.T) {
	skipIfTruncationDisabled(t)
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("Expected panic for platform-dependent int truncation")
		}
	}()
	var a int = 0x80000000
	_ = int32(a)
}

func TestBoundaryTruncationInt32MaxPlusOne(t *testing.T) {
	skipIfTruncationDisabled(t)
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("Expected panic for int32 max+1 boundary truncation")
		}
	}()
	var c int64 = 0x80000000
	_ = int32(c)
}

func TestBoundaryTruncationInt32MinMinusOne(t *testing.T) {
	skipIfTruncationDisabled(t)
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("Expected panic for int32 min-1 boundary truncation")
		}
	}()
	var e int64 = -0x80000001
	_ = int32(e)
}

func TestBoundaryTruncationInt16MaxPlusOne(t *testing.T) {
	skipIfTruncationDisabled(t)
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("Expected panic for int16 max+1 boundary truncation")
		}
	}()
	var g int32 = 0x8000
	_ = int16(g)
}

func TestBoundaryTruncationInt8MaxPlusOne(t *testing.T) {
	skipIfTruncationDisabled(t)
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("Expected panic for int8 max+1 boundary truncation")
		}
	}()
	var i int16 = 0x80
	_ = int8(i)
}

func TestBoundaryTruncationUint32MaxPlusOne(t *testing.T) {
	skipIfTruncationDisabled(t)
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("Expected panic for uint32 max+1 boundary truncation")
		}
	}()
	var k uint64 = 0x100000000
	_ = uint32(k)
}

func TestBoundaryTruncationUint16MaxPlusOne(t *testing.T) {
	skipIfTruncationDisabled(t)
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("Expected panic for uint16 max+1 boundary truncation")
		}
	}()
	var m uint32 = 0x10000
	_ = uint16(m)
}

func TestBoundaryTruncationUint8MaxPlusOne(t *testing.T) {
	skipIfTruncationDisabled(t)
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("Expected panic for uint8 max+1 boundary truncation")
		}
	}()
	var o uint16 = 0x100
	_ = uint8(o)
}

// Additional edge case truncation tests
func TestBitOperationTruncation(t *testing.T) {
	skipIfTruncationDisabled(t)
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("Expected panic for bit operation truncation")
		}
	}()
	var value int64 = 0x123456789ABCDEF0
	var truncated int32 = int32(value)
	_ = truncated
}

func TestChainedTruncationWithBitOps(t *testing.T) {
	skipIfTruncationDisabled(t)
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("Expected panic for chained truncation with bit ops")
		}
	}()
	var value int64 = 0x7FFFFFFFFFFFFFFF
	var step1 int32 = int32(value >> 16)
	_ = step1
}

func TestUintToIntLargeValue(t *testing.T) {
	// This test may not trigger panic depending on implementation
	var large uint64 = 0x8000000000000000
	result := int64(large)
	_ = result
}

func TestIntToUintNegativeEdgeCase(t *testing.T) {
	// This test may not trigger panic depending on implementation
	var negative int64 = -1
	result := uint64(negative)
	_ = result
}

func TestSafeTruncation(t *testing.T) {
	// These conversions should not panic
	var small int64 = 100
	result := int32(small)
	if result != 100 {
		t.Fatalf("Expected 100, got %d", result)
	}

	var smallUint uint64 = 200
	result2 := uint32(smallUint)
	if result2 != 200 {
		t.Fatalf("Expected 200, got %d", result2)
	}
}

// Suppression directive tests for truncation
func TestTruncationSuppression_LineAbove(t *testing.T) {
    // Expect no panic due to suppression marker on previous line
    var big uint16 = 300
    // truncation_false_positive
    _ = uint8(big)
}

func TestTruncationSuppression_SameLine(t *testing.T) {
    // Expect no panic due to suppression marker on same line
    var big uint16 = 300
    _ = uint8(big) // truncation_false_positive
}
