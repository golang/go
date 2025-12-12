// Test带参数装饰器

package main

import "fmt"

// 带参数的装饰器
func validator(f func(int) int) func(int) int {
	fmt.Println("[Init] validator decorator created")
	return func(x int) int {
		fmt.Printf("[Validator] 检查输入: %d\n", x)
		if x < 0 {
			fmt.Println("[Validator] 拒绝负数!")
			return 0
		}
		result := f(x)
		fmt.Printf("[Validator] 返回结果: %d\n", result)
		return result
	}
}

@validator
func square(n int) int {
	fmt.Printf("[Function] 计算 %d 的平方\n", n)
	return n * n
}

func main() {
	fmt.Println("=== 测试开始 ===")
	
	result1 := square(5)
	fmt.Printf("square(5) = %d\n", result1)
	fmt.Println()
	
	result2 := square(-3)
	fmt.Printf("square(-3) = %d\n", result2)
	
	fmt.Println("=== 测试结束 ===")
}
