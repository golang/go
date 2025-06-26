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

func TestSignedMultiplicationOverflow(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("Expected panic for signed multiplication overflow")
		}
	}()
	var a int8 = 127
	var b int8 = 2
	_ = a * b
}

func TestUnsignedMultiplicationOverflow(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("Expected panic for unsigned multiplication overflow")
		}
	}()
	var a uint8 = 255
	var b uint8 = 2
	_ = a * b
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