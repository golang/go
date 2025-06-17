package lib

func UnsafeOperation() int8 {
	// This should NOT trigger overflow detection because it's in vendor/
	var a int8 = 127
	var b int8 = 1
	return a + b  // Would overflow but should be ignored
}