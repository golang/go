package main

import "fmt"

func testDivisionByZero() {
	fmt.Println("Testing division by zero (should panic with divide by zero):")
	defer func() {
		if r := recover(); r != nil {
			fmt.Printf("OK -- Division by zero correctly panicked: %v\n", r)
		} else {
			fmt.Println("FAILED -- Division by zero did not panic")
		}
	}()

	var a int32 = 10
	var b int32 = 0
	result := a / b
	fmt.Printf("Division by zero did not panic, result: %d\n", result)
}

func testNormalDivisions() {
	fmt.Println("Testing normal divisions (should not panic):")

	// Various normal division cases
	cases := []struct {
		a, b     int32
		expected int32
	}{
		{10, 2, 5},
		{-10, 2, -5},
		{10, -2, -5},
		{-10, -2, 5},
		{-2147483647, -1, 2147483647}, // MAX_INT / -1 = -MAX_INT (valid)
		{-2147483648, 2, -1073741824}, // MIN_INT / 2 (valid)
	}

	for _, tc := range cases {
		func() {
			defer func() {
				if r := recover(); r != nil {
					fmt.Printf("Normal division %d / %d panicked unexpectedly: %v\n", tc.a, tc.b, r)
				}
			}()

			result := tc.a / tc.b
			if result == tc.expected {
				fmt.Printf("OK-- %d / %d = %d (correct)\n", tc.a, tc.b, result)
			} else {
				fmt.Printf("FAIL-- %d / %d = %d (expected %d)\n", tc.a, tc.b, result, tc.expected)
			}
		}()
	}
}

func main() {
	fmt.Println("Testing division edge cases...")

	testDivisionByZero()
	fmt.Println()
	testNormalDivisions()

	fmt.Println("\nAll edge case tests completed!")
}
