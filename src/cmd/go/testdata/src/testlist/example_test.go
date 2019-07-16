package testlist

import (
	"fmt"
)

func ExampleSimple() {
	fmt.Println("Test with Output.")

	// Output: Test with Output.
}

func ExampleWithEmptyOutput() {
	fmt.Println("")

	// Output:
}

func ExampleNoOutput() {
	_ = fmt.Sprint("Test with no output")
}
