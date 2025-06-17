package overflow

func TestOverflow() int8 {
	var a, b int8 = 127, 1
	return a + b // Should trigger overflow
}