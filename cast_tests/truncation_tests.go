package main

import (
	"fmt"
	"os"
)

func testInt64ToInt32Overflow() {
	fmt.Println("Testing int64 to int32 overflow...")
	var large int64 = 0x100000000 // 2^32, exceeds int32 range
	_ = int32(large)              // Should panic - truncation loses data
}

func testInt64ToInt32Underflow() {
	fmt.Println("Testing int64 to int32 underflow...")
	var large int64 = -0x100000000 // -2^32, exceeds int32 range
	_ = int32(large)               // Should panic - truncation loses data
}

func testInt32ToInt16Overflow() {
	fmt.Println("Testing int32 to int16 overflow...")
	var large int32 = 0x10000 // 2^16, exceeds int16 range
	_ = int16(large)          // Should panic - truncation loses data
}

func testInt32ToInt16Underflow() {
	fmt.Println("Testing int32 to int16 underflow...")
	var large int32 = -0x10000 // -2^16, exceeds int16 range
	_ = int16(large)           // Should panic - truncation loses data
}

func testInt16ToInt8Overflow() {
	fmt.Println("Testing int16 to int8 overflow...")
	var large int16 = 0x100 // 256, exceeds int8 range
	_ = int8(large)         // Should panic - truncation loses data
}

func testInt16ToInt8Underflow() {
	fmt.Println("Testing int16 to int8 underflow...")
	var large int16 = -0x100 // -256, exceeds int8 range
	_ = int8(large)          // Should panic - truncation loses data
}

func testUint64ToUint32Overflow() {
	fmt.Println("Testing uint64 to uint32 overflow...")
	var large uint64 = 0x100000000 // 2^32, exceeds uint32 range
	_ = uint32(large)              // Should panic - truncation loses data
}

func testUint32ToUint16Overflow() {
	fmt.Println("Testing uint32 to uint16 overflow...")
	var large uint32 = 0x10000 // 2^16, exceeds uint16 range
	_ = uint16(large)          // Should panic - truncation loses data
}

func testUint16ToUint8Overflow() {
	fmt.Println("Testing uint16 to uint8 overflow...")
	var large uint16 = 0x100 // 256, exceeds uint8 range
	_ = uint8(large)         // Should panic - truncation loses data
}

func testIntToInt32OnLargeValues() {
	fmt.Println("Testing int to int32 on large values...")
	var large int = 0x100000000 // On 64-bit systems, this exceeds int32 range
	_ = int32(large)            // Should panic - truncation loses data
}

func testIntToInt16OnLargeValues() {
	fmt.Println("Testing int to int16 on large values...")
	var large int = 0x10000 // Exceeds int16 range
	_ = int16(large)        // Should panic - truncation loses data
}

func testIntToInt8OnLargeValues() {
	fmt.Println("Testing int to int8 on large values...")
	var large int = 0x100 // Exceeds int8 range
	_ = int8(large)       // Should panic - truncation loses data
}

func testUintToUint32OnLargeValues() {
	fmt.Println("Testing uint to uint32 on large values...")
	var large uint = 0x100000000 // On 64-bit systems, this exceeds uint32 range
	_ = uint32(large)            // Should panic - truncation loses data
}

func testUintToUint16OnLargeValues() {
	fmt.Println("Testing uint to uint16 on large values...")
	var large uint = 0x10000 // Exceeds uint16 range
	_ = uint16(large)        // Should panic - truncation loses data
}

func testUintToUint8OnLargeValues() {
	fmt.Println("Testing uint to uint8 on large values...")
	var large uint = 0x100 // Exceeds uint8 range
	_ = uint8(large)       // Should panic - truncation loses data
}

func testSignedToUnsignedNegative() {
	fmt.Println("Testing signed to unsigned with negative values...")
	var negative int32 = -1
	_ = uint32(negative) // Should panic - negative values to unsigned is unsafe
}

func testSignedToUnsignedNegativeInt16() {
	fmt.Println("Testing int16 to uint16 with negative values...")
	var negative int16 = -1
	_ = uint16(negative) // Should panic - negative values to unsigned is unsafe
}

func testSignedToUnsignedNegativeInt8() {
	fmt.Println("Testing int8 to uint8 with negative values...")
	var negative int8 = -1
	_ = uint8(negative) // Should panic - negative values to unsigned is unsafe
}

func testComplexTruncationChain() {
	fmt.Println("Testing complex truncation chain...")
	var start int64 = 0x123456789ABCDEF // Large 64-bit value
	var step1 int32 = int32(start)      // First truncation - should panic
	var step2 int16 = int16(step1)      // Second truncation (won't reach if first panics)
	_ = int8(step2)                     // Third truncation (won't reach if first panics)
}

func testRuntimeComputedTruncation() {
	fmt.Println("Testing runtime computed truncation...")
	var base int64 = 1000
	for i := 0; i < 10; i++ {
		base = base * 10 // Grows exponentially
	}
	_ = int32(base) // Should panic when base exceeds int32 range
}

func testBufferSizeVulnerability() {
	fmt.Println("Testing buffer size vulnerability simulation...")
	// Simulate a common security vulnerability where buffer size is truncated
	var requestedSize int64 = 0x200000000       // Large allocation request (8GB)
	var actualSize int32 = int32(requestedSize) // Truncated to small size - should panic
	_ = actualSize
}

func testArrayIndexTruncation() {
	fmt.Println("Testing array index truncation...")
	var largeIndex int64 = 0x80000000            // Large index value
	var truncatedIndex int32 = int32(largeIndex) // Should panic - index truncation
	_ = truncatedIndex
}

func testMemoryOffsetTruncation() {
	fmt.Println("Testing memory offset truncation...")
	var offset int64 = 0x180000000            // Large memory offset
	var truncatedOffset int32 = int32(offset) // Should panic - offset truncation
	_ = truncatedOffset
}

func testSecurityBoundaryTruncation() {
	fmt.Println("Testing security boundary truncation...")
	// Simulate truncation that could bypass security checks
	var securityLimit int64 = 0x7FFFFFFF + 1000   // Just above int32 max
	var checkedLimit int32 = int32(securityLimit) // Should panic - security boundary violated
	_ = checkedLimit
}

func runTest(testName string, testFunc func()) {
	defer func() {
		if r := recover(); r != nil {
			fmt.Printf("PASS %s: Correctly panicked with: %v\n", testName, r)
		} else {
			fmt.Printf("FAIL %s: Should have panicked but didn't\n", testName)
			os.Exit(1)
		}
	}()
	testFunc()
}

func main() {
	fmt.Println("Running integer truncation security tests...")
	fmt.Println("Testing critical truncation patterns that must be prevented for security.")
	fmt.Println()

	// Core integer truncation tests
	runTest("Int64 to Int32 Overflow", testInt64ToInt32Overflow)
	runTest("Int64 to Int32 Underflow", testInt64ToInt32Underflow)
	runTest("Int32 to Int16 Overflow", testInt32ToInt16Overflow)
	runTest("Int32 to Int16 Underflow", testInt32ToInt16Underflow)
	runTest("Int16 to Int8 Overflow", testInt16ToInt8Overflow)
	runTest("Int16 to Int8 Underflow", testInt16ToInt8Underflow)
	runTest("Uint64 to Uint32 Overflow", testUint64ToUint32Overflow)
	runTest("Uint32 to Uint16 Overflow", testUint32ToUint16Overflow)
	runTest("Uint16 to Uint8 Overflow", testUint16ToUint8Overflow)
	runTest("Int to Int32 Large Values", testIntToInt32OnLargeValues)
	runTest("Int to Int16 Large Values", testIntToInt16OnLargeValues)
	runTest("Int to Int8 Large Values", testIntToInt8OnLargeValues)
	runTest("Uint to Uint32 Large Values", testUintToUint32OnLargeValues)
	runTest("Uint to Uint16 Large Values", testUintToUint16OnLargeValues)
	runTest("Uint to Uint8 Large Values", testUintToUint8OnLargeValues)
	runTest("Signed to Unsigned Negative", testSignedToUnsignedNegative)
	runTest("Int16 to Uint16 Negative", testSignedToUnsignedNegativeInt16)
	runTest("Int8 to Uint8 Negative", testSignedToUnsignedNegativeInt8)
	runTest("Complex Truncation Chain", testComplexTruncationChain)
	runTest("Runtime Computed Truncation", testRuntimeComputedTruncation)
	runTest("Buffer Size Vulnerability", testBufferSizeVulnerability)
	runTest("Array Index Truncation", testArrayIndexTruncation)
	runTest("Memory Offset Truncation", testMemoryOffsetTruncation)
	runTest("Security Boundary Truncation", testSecurityBoundaryTruncation)

	fmt.Println("The compiler correctly detected and prevented all critical integer truncations.")
}
