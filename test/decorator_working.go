// 当前能工作的装饰器示例

package main

import "fmt"

// 装饰器定义（func() func() 签名）
func logger(f func()) func() {
	fmt.Println("[Decorator] logger called during program init")
	return func() {
		fmt.Println("[Decorator] Before function call")
		f()
		fmt.Println("[Decorator] After function call")
	}
}

func trace(f func()) func() {
	fmt.Println("[Decorator] trace called during program init")
	return func() {
		fmt.Println("[Trace] → Entering function")
		f()
		fmt.Println("[Trace] ← Exiting function")
	}
}

// 装饰无参数、无返回值的函数
@logger
func greet() {
	fmt.Println("Hello, World!")
}

@trace
func doWork() {
	fmt.Println("Working...")
}

@logger
func sayGoodbye() {
	fmt.Println("Goodbye!")
}

func main() {
	fmt.Println("=== Program Started ===")
	fmt.Println()
	
	greet()
	fmt.Println()
	
	doWork()
	fmt.Println()
	
	sayGoodbye()
	fmt.Println()
	
	fmt.Println("=== Program Ended ===")
}
