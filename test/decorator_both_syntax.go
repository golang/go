// compile

// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// 测试两种装饰器语法：换行和同行

package main

func decorator1(f func()) func() {
	return f
}

func decorator2(f func(int) int) func(int) int {
	return f
}

func decorator3(f func(int, string) bool) func(int, string) bool {
	return f
}

// ==========================================
// 换行语法测试
// ==========================================

@decorator1
func newlineFunc1() {
}

@decorator2
func newlineFunc2(x int) int {
	return x
}

@decorator3
func newlineFunc3(n int, s string) bool {
	return true
}

// ==========================================
// 同行语法测试
// ==========================================

@decorator1 func samelineFunc1() {
}

@decorator2 func samelineFunc2(x int) int {
	return x
}

@decorator3 func samelineFunc3(n int, s string) bool {
	return true
}

// ==========================================
// 混合使用
// ==========================================

@decorator1
func mixedFunc1() {
}

@decorator2 func mixedFunc2(x int) int {
	return x
}

@decorator3
func mixedFunc3(n int, s string) bool {
	return true
}

func main() {
	// 换行语法
	newlineFunc1()
	_ = newlineFunc2(1)
	_ = newlineFunc3(1, "a")
	
	// 同行语法
	samelineFunc1()
	_ = samelineFunc2(2)
	_ = samelineFunc3(2, "b")
	
	// 混合使用
	mixedFunc1()
	_ = mixedFunc2(3)
	_ = mixedFunc3(3, "c")
}
