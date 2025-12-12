// compile

// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// 综合测试装饰器语法

package main

// 各种装饰器函数
func noopDecorator(f func()) func() {
	return f
}

func intDecorator(f func(int) int) func(int) int {
	return f
}

func multiArgDecorator(f func(int, string) bool) func(int, string) bool {
	return f
}

func returnsMultiple(f func() (int, string)) func() (int, string) {
	return f
}

// 使用装饰器的函数
@noopDecorator func simpleFunc() {
}

@intDecorator func addOne(x int) int {
	return x + 1
}

@multiArgDecorator func checkValue(n int, s string) bool {
	return n > 0 && len(s) > 0
}

@returnsMultiple func getPair() (int, string) {
	return 42, "answer"
}

// 无装饰器的普通函数
func normalFunc() {
}

func main() {
	simpleFunc()
	_ = addOne(10)
	_ = checkValue(5, "test")
	_, _ = getPair()
	normalFunc()
}
