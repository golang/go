// Test of decorator functionality - with params

package main

import "fmt"

// 装饰器定义 - 带参数
func loggerWithParams(f func(string)) func(string) {
	fmt.Println("[Decorator] loggerWithParams called during program init")
	return func(name string) {
		fmt.Println("[Decorator] Before function call")
		f(name)
		fmt.Println("[Decorator] After function call")
	}
}

// 装饰带参数、无返回值函数
@loggerWithParams
func greet2(name string) {
	fmt.Println("Hello from greet2,", name)
}

func main() {
	fmt.Println("=== Program Started ===")
	fmt.Println()
	
	fmt.Println("--- 测试: 带参数函数 ---")
	greet2("Alice")
	fmt.Println()
	
	fmt.Println("=== Program Ended ===")
}

