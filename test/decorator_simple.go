// Test of decorator functionality - no params

package main

import "fmt"

// 装饰器定义 - 无参数
func logger(f func()) func() {
	fmt.Println("[Decorator] logger called during program init")
	return func() {
		fmt.Println("[Decorator] Before function call")
		f()
		fmt.Println("[Decorator] After function call")
	}
}

// 装饰无参数、无返回值函数
@logger
func greet1() {
	fmt.Println("Hello from greet1!")
}

func main() {
	fmt.Println("=== Program Started ===")
	fmt.Println()
	
	fmt.Println("--- 测试 1: 无参数函数 ---")
	greet1()
	fmt.Println()
	
	fmt.Println("=== Program Ended ===")
}

