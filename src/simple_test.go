package main

func testAdd() int8 {
	var a, b int8 = 127, 1
	return a + b // Should overflow: 127 + 1 = -128 (wraps around)
}

func testSub() int8 {
	var a, b int8 = -128, 1
	return a - b // Should overflow: -128 - 1 = 127 (wraps around)
}

func testMul() int8 {
	var a, b int8 = 64, 2
	return a * b // Should overflow: 64 * 2 = -128 (wraps around)
}

func main() {
	_ = testAdd()
	_ = testSub()
	_ = testMul()
}