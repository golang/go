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


func logger2(f func()) func() {
	fmt.Println("[Decorator] logger called during program init")
	return func() {
		fmt.Println("[Decorator] Before function call 2")
		f()
		fmt.Println("[Decorator] After function call 2")
	}
}


func logger3(f func(), message string) func() {
	fmt.Println("[Decorator] logger called during program init")
	return func() {
		fmt.Println("[Decorator] Before function call 2", message)
		f()
		fmt.Println("[Decorator] After function call 2", message)
	}
}

func loggerWithParams(f func(string, string)) func(string, string) {
	fmt.Println("[Decorator] loggerWithParams called during program init")
	return func(name string, message string) {
		fmt.Println("[Decorator] Before function call", name)
		f(name, message)
		fmt.Println("[Decorator] After function call", message)
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


func loggerWithReturnAndParams(f func(int) int, logName string, logMessage string) func(int) int {
	fmt.Println("[Decorator] loggerWithReturn called during program init")
	return func(x int) int {
		fmt.Println("[Decorator] Before function call", logName)
		result := f(x)
		fmt.Println("[Decorator] After function call", logName)
		return result
	}
}


// 装饰无参数、无返回值函数
@logger3("Hello from greet1!")
@logger2
@logger
func greet1() {
	fmt.Println("Hello from greet1!")
}

// 装饰带参数、无返回值函数（参数会被闭包捕获）
@loggerWithParams
func greet2(name string, message string) {
	fmt.Println("Hello from greet2,", name, message)
}

// 装饰带参数、带返回值的函数
@loggerWithReturn
func calculate(x int) int {
	fmt.Printf("Calculating square of %d\n", x)
	return x * x
}

@loggerWithReturnAndParams("calculate2", "Calculating square of %d")
func calculate2(x int) int {
	fmt.Printf("Calculating square of %d\n", x)
	return x * x
}

func main() {
	fmt.Println("=== Program Started ===")
	fmt.Println()
	
	fmt.Println("--- 测试 1: 无参数函数 ---")
	greet1()
	fmt.Println()
	
	fmt.Println("--- 测试 2: 带参数函数 ---")
	greet2("Alice", "Hello from greet2, Alice")
	fmt.Println()
	
	fmt.Println("--- 测试 3: 带返回值函数 ---")
	result := calculate(5)
	fmt.Printf("Result: %d\n", result)
	fmt.Println()
	
	fmt.Println("=== Program Ended ===")
}
