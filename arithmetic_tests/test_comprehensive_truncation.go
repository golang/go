package main

import "fmt"

func main() {
	fmt.Println("Testing comprehensive truncation detection...")

	// Test different type conversion scenarios

	// 1. int32 to int16 truncation (should panic)
	var i32 int32 = 32768
	fmt.Printf("Test 1 - Before: i32=%d\n", i32)
	i16 := int16(i32) // Should panic
	fmt.Printf("Test 1 - After: i16=%d\n", i16)

	// 2. int64 to int32 truncation (should panic)
	var i64 int64 = 2147483648
	fmt.Printf("Test 2 - Before: i64=%d\n", i64)
	i32_2 := int32(i64) // Should panic
	fmt.Printf("Test 2 - After: i32=%d\n", i32_2)

	// 3. uint64 to uint32 truncation (should panic)
	var u64 uint64 = 4294967296
	fmt.Printf("Test 3 - Before: u64=%d\n", u64)
	u32 := uint32(u64) // Should panic
	fmt.Printf("Test 3 - After: u32=%d\n", u32)

	// 4. Mixed signed/unsigned truncation (should panic)
	var i32_3 int32 = 256
	fmt.Printf("Test 4 - Before: i32=%d\n", i32_3)
	u8 := uint8(i32_3) // Should panic
	fmt.Printf("Test 4 - After: u8=%d\n", u8)

	fmt.Println("All tests completed (should not reach here)")
}