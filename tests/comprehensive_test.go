package tests

import (
	"testing"
)

// Fuzzing tests for comprehensive coverage
func FuzzIntegerOverflow(f *testing.F) {
	f.Add(int8(126), int8(1))
	f.Add(int8(127), int8(1))
	f.Add(int8(-127), int8(-2))
	f.Add(int8(-128), int8(1))
	
	f.Fuzz(func(t *testing.T, a, b int8) {
		defer func() {
			recover() // Catch any panics during fuzzing
		}()
		_ = a + b
		_ = a - b
		_ = a * b
		if b != 0 {
			_ = a / b
		}
	})
}

func FuzzUnsignedIntegerOverflow(f *testing.F) {
	f.Add(uint8(254), uint8(1))
	f.Add(uint8(255), uint8(1))
	f.Add(uint8(0), uint8(1))
	f.Add(uint8(1), uint8(255))
	
	f.Fuzz(func(t *testing.T, a, b uint8) {
		defer func() {
			recover() // Catch any panics during fuzzing
		}()
		_ = a + b
		_ = a - b
		_ = a * b
		if b != 0 {
			_ = a / b
		}
	})
}

func FuzzIntegerTruncation(f *testing.F) {
	f.Add(int64(0x7FFFFFFF + 1))
	f.Add(int64(-0x80000000 - 1))
	f.Add(int64(0x123456789ABCDEF))
	f.Add(int64(-0x123456789ABCDEF))
	
	f.Fuzz(func(t *testing.T, val int64) {
		defer func() {
			recover() // Catch any panics during fuzzing
		}()
		_ = int32(val)
		_ = int16(val)
		_ = int8(val)
	})
}

// Edge case tests
func TestEdgeCaseDivisionByZero(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("Expected panic for division by zero")
		}
	}()
	var a int8 = 10
	var b int8 = 0
	_ = a / b
}

func TestEdgeCaseMinIntDivisionByMinusOne(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("Expected panic for MIN_INT / -1")
		}
	}()
	var a int8 = -128
	var b int8 = -1
	_ = a / b
}

func TestEdgeCaseMaxValueOperations(t *testing.T) {
	t.Run("Int8MaxAddition", func(t *testing.T) {
		defer func() {
			if r := recover(); r == nil {
				t.Fatal("Expected panic for int8 max + 1")
			}
		}()
		var a int8 = 127
		var b int8 = 1
		_ = a + b
	})
	
	t.Run("Uint8MaxAddition", func(t *testing.T) {
		defer func() {
			if r := recover(); r == nil {
				t.Fatal("Expected panic for uint8 max + 1")
			}
		}()
		var a uint8 = 255
		var b uint8 = 1
		_ = a + b
	})
	
	t.Run("Int16MaxAddition", func(t *testing.T) {
		defer func() {
			if r := recover(); r == nil {
				t.Fatal("Expected panic for int16 max + 1")
			}
		}()
		var a int16 = 32767
		var b int16 = 1
		_ = a + b
	})
	
	t.Run("Uint16MaxAddition", func(t *testing.T) {
		defer func() {
			if r := recover(); r == nil {
				t.Fatal("Expected panic for uint16 max + 1")
			}
		}()
		var a uint16 = 65535
		var b uint16 = 1
		_ = a + b
	})
	
	t.Run("Int32MaxAddition", func(t *testing.T) {
		defer func() {
			if r := recover(); r == nil {
				t.Fatal("Expected panic for int32 max + 1")
			}
		}()
		var a int32 = 2147483647
		var b int32 = 1
		_ = a + b
	})
	
	t.Run("Uint32MaxAddition", func(t *testing.T) {
		defer func() {
			if r := recover(); r == nil {
				t.Fatal("Expected panic for uint32 max + 1")
			}
		}()
		var a uint32 = 4294967295
		var b uint32 = 1
		_ = a + b
	})
	
	t.Run("Uint64MaxAddition", func(t *testing.T) {
		defer func() {
			if r := recover(); r == nil {
				t.Fatal("Expected panic for uint64 max + 1")
			}
		}()
		var a uint64 = 18446744073709551615
		var b uint64 = 1
		_ = a + b
	})
}

func TestEdgeCaseMinValueOperations(t *testing.T) {
	t.Run("Int8MinSubtraction", func(t *testing.T) {
		defer func() {
			if r := recover(); r == nil {
				t.Fatal("Expected panic for int8 min - 1")
			}
		}()
		var a int8 = -128
		var b int8 = 1
		_ = a - b
	})
	
	t.Run("Uint8MinSubtraction", func(t *testing.T) {
		defer func() {
			if r := recover(); r == nil {
				t.Fatal("Expected panic for uint8 min - 1")
			}
		}()
		var a uint8 = 0
		var b uint8 = 1
		_ = a - b
	})
	
	t.Run("Int16MinSubtraction", func(t *testing.T) {
		defer func() {
			if r := recover(); r == nil {
				t.Fatal("Expected panic for int16 min - 1")
			}
		}()
		var a int16 = -32768
		var b int16 = 1
		_ = a - b
	})
	
	t.Run("Uint16MinSubtraction", func(t *testing.T) {
		defer func() {
			if r := recover(); r == nil {
				t.Fatal("Expected panic for uint16 min - 1")
			}
		}()
		var a uint16 = 0
		var b uint16 = 1
		_ = a - b
	})
	
	t.Run("Int32MinSubtraction", func(t *testing.T) {
		defer func() {
			if r := recover(); r == nil {
				t.Fatal("Expected panic for int32 min - 1")
			}
		}()
		var a int32 = -2147483648
		var b int32 = 1
		_ = a - b
	})
	
	t.Run("Uint32MinSubtraction", func(t *testing.T) {
		defer func() {
			if r := recover(); r == nil {
				t.Fatal("Expected panic for uint32 min - 1")
			}
		}()
		var a uint32 = 0
		var b uint32 = 1
		_ = a - b
	})
	
	t.Run("Uint64MinSubtraction", func(t *testing.T) {
		defer func() {
			if r := recover(); r == nil {
				t.Fatal("Expected panic for uint64 min - 1")
			}
		}()
		var a uint64 = 0
		var b uint64 = 1
		_ = a - b
	})
}

// Combined arithmetic and truncation tests
func TestCombinedArithmeticAndTruncation(t *testing.T) {
	t.Run("OverflowThenTruncate", func(t *testing.T) {
		defer func() {
			if r := recover(); r == nil {
				t.Fatal("Expected panic for overflow or truncation")
			}
		}()
		var a int32 = 2147483647
		var b int32 = 1
		result := a + b // Should panic here
		_ = int16(result) // Won't reach if first operation panics
	})
	
	t.Run("TruncateThenOverflow", func(t *testing.T) {
		defer func() {
			if r := recover(); r == nil {
				t.Fatal("Expected panic for truncation")
			}
		}()
		var large int64 = 0x100000000
		var truncated int32 = int32(large) // Should panic here
		_ = truncated + 1 // Won't reach if first operation panics
	})
}

// Real-world scenario tests
func TestRealWorldScenarios(t *testing.T) {
	t.Run("BufferAllocation", func(t *testing.T) {
		defer func() {
			if r := recover(); r == nil {
				t.Fatal("Expected panic for buffer size truncation")
			}
		}()
		var requestedSize int64 = 0x100000000 // 4GB
		var bufferSize int32 = int32(requestedSize) // Should panic
		_ = bufferSize
	})
	
	t.Run("ArrayIndexing", func(t *testing.T) {
		defer func() {
			if r := recover(); r == nil {
				t.Fatal("Expected panic for array index truncation")
			}
		}()
		var largeIndex int64 = 0x80000000
		var arrayIndex int32 = int32(largeIndex) // Should panic
		_ = arrayIndex
	})
	
	t.Run("NetworkPacketSize", func(t *testing.T) {
		defer func() {
			if r := recover(); r == nil {
				t.Fatal("Expected panic for packet size calculation")
			}
		}()
		var headerSize uint16 = 65535
		var payloadSize uint16 = 1
		var totalSize uint16 = headerSize + payloadSize // Should panic
		_ = totalSize
	})
}