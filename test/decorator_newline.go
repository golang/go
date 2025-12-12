// compile

// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// 测试装饰器换行语法

package main

func wrapper1(f func()) func() {
	return f
}

func wrapper2(f func(int) int) func(int) int {
	return func(x int) int {
		return f(x) * 2
	}
}

// 装饰器和 func 在不同行（换行语法）
@wrapper1
func noArgs() {
}

@wrapper2
func withArgs(x int) int {
	return x + 1
}

// 也支持在同一行（向后兼容）
@wrapper1 func sameLineFunc() {
}

func main() {
	noArgs()
	_ = withArgs(10)
	sameLineFunc()
}
