package tests

import (
	"testing"
)

func TestSignedInt8Overflow(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("Expected panic for int8 overflow")
		}
	}()
	var a int8 = 127
	var b int8 = 1
	_ = a + b
}

func TestSignedInt8Underflow(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("Expected panic for int8 underflow")
		}
	}()
	var a int8 = -128
	var b int8 = 1
	_ = a - b
}

func TestSignedInt16Overflow(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("Expected panic for int16 overflow")
		}
	}()
	var a int16 = 32767
	var b int16 = 1
	_ = a + b
}

func TestSignedInt16Underflow(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("Expected panic for int16 underflow")
		}
	}()
	var a int16 = -32768
	var b int16 = 1
	_ = a - b
}

func TestSignedInt32Overflow(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("Expected panic for int32 overflow")
		}
	}()
	var a int32 = 2147483647
	var b int32 = 1
	_ = a + b
}

func TestSignedInt32Underflow(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("Expected panic for int32 underflow")
		}
	}()
	var a int32 = -2147483648
	var b int32 = 1
	_ = a - b
}

func TestUnsignedUint8Overflow(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("Expected panic for uint8 overflow")
		}
	}()
	var a uint8 = 255
	var b uint8 = 1
	_ = a + b
}

func TestUnsignedUint8Underflow(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("Expected panic for uint8 underflow")
		}
	}()
	var a uint8 = 0
	var b uint8 = 1
	_ = a - b
}

func TestUnsignedUint16Overflow(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("Expected panic for uint16 overflow")
		}
	}()
	var a uint16 = 65535
	var b uint16 = 1
	_ = a + b
}

func TestUnsignedUint16Underflow(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("Expected panic for uint16 underflow")
		}
	}()
	var a uint16 = 0
	var b uint16 = 1
	_ = a - b
}

func TestUnsignedUint32Overflow(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("Expected panic for uint32 overflow")
		}
	}()
	var a uint32 = 4294967295
	var b uint32 = 1
	_ = a + b
}

func TestUnsignedUint32Underflow(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("Expected panic for uint32 underflow")
		}
	}()
	var a uint32 = 0
	var b uint32 = 1
	_ = a - b
}

func TestUnsignedUint64Overflow(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("Expected panic for uint64 overflow")
		}
	}()
	var a uint64 = 18446744073709551615
	var b uint64 = 1
	_ = a + b
}

func TestUnsignedUint64Underflow(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("Expected panic for uint64 underflow")
		}
	}()
	var a uint64 = 0
	var b uint64 = 1
	_ = a - b
}


func TestSignedDivisionOverflow(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("Expected panic for signed division overflow (MIN_INT / -1)")
		}
	}()
	var a int8 = -128
	var b int8 = -1
	_ = a / b
}

// Boundary value arithmetic tests - near overflow boundaries
func TestBoundaryArithmeticInt8(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("Expected panic for int8 boundary arithmetic")
		}
	}()
	var a int8 = 126
	var b int8 = 2
	_ = a + b
}

func TestBoundaryArithmeticInt16(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("Expected panic for int16 boundary arithmetic")
		}
	}()
	var a int16 = 32766
	var b int16 = 2
	_ = a + b
}

func TestBoundaryArithmeticInt32(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("Expected panic for int32 boundary arithmetic")
		}
	}()
	var a int32 = 2147483646
	var b int32 = 2
	_ = a + b
}

func TestBoundaryArithmeticUint8(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("Expected panic for uint8 boundary arithmetic")
		}
	}()
	var a uint8 = 254
	var b uint8 = 2
	_ = a + b
}

func TestBoundaryArithmeticUint16(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("Expected panic for uint16 boundary arithmetic")
		}
	}()
	var a uint16 = 65534
	var b uint16 = 2
	_ = a + b
}

func TestBoundaryArithmeticUint32(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("Expected panic for uint32 boundary arithmetic")
		}
	}()
	var a uint32 = 4294967294
	var b uint32 = 2
	_ = a + b
}

func TestBoundaryArithmeticUint64(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("Expected panic for uint64 boundary arithmetic")
		}
	}()
	var a uint64 = 18446744073709551614
	var b uint64 = 2
	_ = a + b
}

func TestNearNegativeBoundaryInt8(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("Expected panic for int8 near negative boundary")
		}
	}()
	var a int8 = -127
	var b int8 = 2
	_ = a - b
}

func TestNearNegativeBoundaryInt16(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("Expected panic for int16 near negative boundary")
		}
	}()
	var a int16 = -32767
	var b int16 = 2
	_ = a - b
}

func TestNearNegativeBoundaryInt32(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("Expected panic for int32 near negative boundary")
		}
	}()
	var a int32 = -2147483647
	var b int32 = 2
	_ = a - b
}

// Zero operand tests
func TestZeroOperandAddition(t *testing.T) {
	// Adding zero to max value should not panic
	var a int8 = 127
	var b int8 = 0
	result := a + b
	if result != 127 {
		t.Fatalf("Expected 127, got %d", result)
	}
}

func TestZeroOperandSubtraction(t *testing.T) {
	// Subtracting zero from min value should not panic
	var a int8 = -128
	var b int8 = 0
	result := a - b
	if result != -128 {
		t.Fatalf("Expected -128, got %d", result)
	}
}

func TestZeroOperandMultiplication(t *testing.T) {
	// Multiplying by zero should not panic
	var a int8 = 127
	var b int8 = 0
	result := a * b
	if result != 0 {
		t.Fatalf("Expected 0, got %d", result)
	}
}

func TestSubtractFromZeroUnsigned(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("Expected panic for subtracting from zero unsigned")
		}
	}()
	var a uint8 = 0
	var b uint8 = 255
	_ = a - b
}

// Division edge cases for int16 and int32
func TestInt16DivisionOverflow(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("Expected panic for int16 division overflow (MIN_INT / -1)")
		}
	}()
	var a int16 = -32768
	var b int16 = -1
	_ = a / b
}

func TestInt32DivisionOverflow(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("Expected panic for int32 division overflow (MIN_INT / -1)")
		}
	}()
	var a int32 = -2147483648
	var b int32 = -1
	_ = a / b
}

// Safe multiplication test
func TestSafeMultiplication(t *testing.T) {
	// These operations should not panic
	var a int8 = 10
	var b int8 = 10
	result := a * b
	if result != 100 {
		t.Fatalf("Expected 100, got %d", result)
	}
}

func TestSafeArithmetic(t *testing.T) {
	// These operations should not panic
	var a int8 = 50
	var b int8 = 30
	result := a + b
	if result != 80 {
		t.Fatalf("Expected 80, got %d", result)
	}

	var c uint8 = 100
	var d uint8 = 50
	result2 := c + d
	if result2 != 150 {
		t.Fatalf("Expected 150, got %d", result2)
	}
}

// Suppression directive tests for overflow/underflow
func TestOverflowSuppression_LineAbove(t *testing.T) {
    // Expect no panic due to suppression marker on previous line
    var a int8 = 120
    var b int8 = 10
    // overflow_false_positive
    _ = a + b
}

func TestOverflowSuppression_SameLine(t *testing.T) {
    // Expect no panic due to suppression marker on same line
    var a int8 = 120
    var b int8 = 10
    _ = a + b // overflow_false_positive
}