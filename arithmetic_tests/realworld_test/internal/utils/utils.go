package utils

func ProcessData(a, b int8) int8 {
	return a - b  // Should this trigger overflow detection?
}