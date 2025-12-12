// Final test of decorator functionality

package main

import "fmt"

// 装饰器定义
func logger(f func()) func() {
	fmt.Println("[Decorator] logger called during program init")
	return func() {
		fmt.Println("[Decorator] Before function call")
		f()
		fmt.Println("[Decorator] After function call")
	}
}

func loggerWithParams(f func(string)) func(string) {
	fmt.Println("[Decorator] loggerWithParams called during program init")
	return func(name string) {
		fmt.Println("[Decorator] Before function call")
		f(name)
		fmt.Println("[Decorator] After function call")
	}
}

// 带返回值的装饰器
func loggerWithReturn(f func(int) int) func(int) int {
	fmt.Println("[Decorator] loggerWithReturn called during program init")
	return func(x int) int {
		fmt.Println("[Decorator] Before function call")
		result := f(x)
		fmt.Println("[Decorator] After function call")
		return result
	}
}

// 装饰无参数、无返回值函数
func greet1() {
	fmt.Println("Hello from greet1!")
}

// 装饰带参数、无返回值函数（参数会被闭包捕获）
func greet2(name string) {
	fmt.Println("Hello from greet2,", name)
}

// 装饰带参数、带返回值的函数
func calculate(x int) int {
	fmt.Printf("Calculating square of %d\n", x)
	return x * x
}

func main() {
	fmt.Println("=== Program Started ===")
	fmt.Println()

	fmt.Println("--- 测试 1: 无参数函数 ---")
	greet1_w := logger(greet1)
	greet1_w()
	fmt.Println()

	fmt.Println("--- 测试 2: 带参数函数 ---")
	greet2_w := loggerWithParams(greet2)
	greet2_w("Alice")
	fmt.Println()

	fmt.Println("--- 测试 3: 带返回值函数 ---")
	calculate_w := loggerWithReturn(calculate)
	result := calculate_w(5)
	fmt.Printf("Result: %d\n", result)
	fmt.Println()

	fmt.Println("=== Program Ended ===")
}
