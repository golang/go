package tests

import (
	"testing"
)

func TestInt64ToInt32Overflow(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("Expected panic for int64 to int32 overflow")
		}
	}()
	var large int64 = 0x100000000
	_ = int32(large)
}

func TestInt64ToInt32Underflow(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("Expected panic for int64 to int32 underflow")
		}
	}()
	var large int64 = -0x100000000
	_ = int32(large)
}

func TestInt32ToInt16Overflow(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("Expected panic for int32 to int16 overflow")
		}
	}()
	var large int32 = 0x10000
	_ = int16(large)
}

func TestInt32ToInt16Underflow(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("Expected panic for int32 to int16 underflow")
		}
	}()
	var large int32 = -0x10000
	_ = int16(large)
}

func TestInt16ToInt8Overflow(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("Expected panic for int16 to int8 overflow")
		}
	}()
	var large int16 = 0x100
	_ = int8(large)
}

func TestInt16ToInt8Underflow(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("Expected panic for int16 to int8 underflow")
		}
	}()
	var large int16 = -0x100
	_ = int8(large)
}

func TestUint64ToUint32Overflow(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("Expected panic for uint64 to uint32 overflow")
		}
	}()
	var large uint64 = 0x100000000
	_ = uint32(large)
}

func TestUint32ToUint16Overflow(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("Expected panic for uint32 to uint16 overflow")
		}
	}()
	var large uint32 = 0x10000
	_ = uint16(large)
}

func TestUint16ToUint8Overflow(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("Expected panic for uint16 to uint8 overflow")
		}
	}()
	var large uint16 = 0x100
	_ = uint8(large)
}

func TestIntToInt32OnLargeValues(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("Expected panic for int to int32 on large values")
		}
	}()
	var large int = 0x100000000
	_ = int32(large)
}

func TestIntToInt16OnLargeValues(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("Expected panic for int to int16 on large values")
		}
	}()
	var large int = 0x10000
	_ = int16(large)
}

func TestIntToInt8OnLargeValues(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("Expected panic for int to int8 on large values")
		}
	}()
	var large int = 0x100
	_ = int8(large)
}

func TestUintToUint32OnLargeValues(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("Expected panic for uint to uint32 on large values")
		}
	}()
	var large uint = 0x100000000
	_ = uint32(large)
}

func TestUintToUint16OnLargeValues(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("Expected panic for uint to uint16 on large values")
		}
	}()
	var large uint = 0x10000
	_ = uint16(large)
}

func TestUintToUint8OnLargeValues(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("Expected panic for uint to uint8 on large values")
		}
	}()
	var large uint = 0x100
	_ = uint8(large)
}

func TestSignedToUnsignedNegative(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("Expected panic for signed to unsigned with negative values")
		}
	}()
	var negative int32 = -1
	_ = uint32(negative)
}

func TestInt16ToUint16Negative(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("Expected panic for int16 to uint16 with negative values")
		}
	}()
	var negative int16 = -1
	_ = uint16(negative)
}

func TestInt8ToUint8Negative(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("Expected panic for int8 to uint8 with negative values")
		}
	}()
	var negative int8 = -1
	_ = uint8(negative)
}

func TestComplexTruncationChain(t *testing.T) {
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
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("Expected panic for security boundary truncation")
		}
	}()
	var securityLimit int64 = 0x7FFFFFFF + 1000
	var checkedLimit int32 = int32(securityLimit)
	_ = checkedLimit
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