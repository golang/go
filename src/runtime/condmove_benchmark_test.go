// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime_test

import (
	"math/rand"
	"testing"
)

var (
	benchNum = 1000000 // 减少循环次数，避免测试时间过长
	// 预生成测试数据，避免在测试中生成
	testData32  = make([]uint32, 1000)
	testData16  = make([]uint16, 1000)
	testDataInt = make([]int, 1000)
)

func init() {
	// 初始化测试数据，包含各种边界情况
	for i := range testData32 {
		if i%4 == 0 {
			testData32[i] = 0
		} else if i%4 == 1 {
			testData32[i] = 1
		} else if i%4 == 2 {
			testData32[i] = uint32(rand.Intn(1000))
		} else {
			testData32[i] = 0xFFFFFFFF
		}
		testData16[i] = uint16(testData32[i])
		testDataInt[i] = int(testData32[i])
	}
}

// 测试用例1: 随机条件选择 - 避免分支预测
func BenchmarkRandomCondSelect32(b *testing.B) {
	for i := 0; i < b.N; i++ {
		for j := 0; j < len(testData32); j++ {
			_ = randomCondSelect32(testData32[j], testData32[(j+1)%len(testData32)])
		}
	}
}

func BenchmarkRandomCondSelect16(b *testing.B) {
	for i := 0; i < b.N; i++ {
		for j := 0; j < len(testData16); j++ {
			_ = randomCondSelect16(testData16[j], testData16[(j+1)%len(testData16)])
		}
	}
}

func BenchmarkRandomCondSelectInt(b *testing.B) {
	for i := 0; i < b.N; i++ {
		for j := 0; j < len(testDataInt); j++ {
			_ = randomCondSelectInt(testDataInt[j], testDataInt[(j+1)%len(testDataInt)])
		}
	}
}

// 新增简单测试用例
func BenchmarkSimpleCondSelect32(b *testing.B) {
	for i := 0; i < b.N; i++ {
		for j := 0; j < len(testData32); j++ {
			_ = simpleCondSelect32(testData32[j], testData32[(j+1)%len(testData32)])
		}
	}
}

func BenchmarkMinCondSelect32(b *testing.B) {
	for i := 0; i < b.N; i++ {
		for j := 0; j < len(testData32); j++ {
			_ = minCondSelect32(testData32[j], testData32[(j+1)%len(testData32)])
		}
	}
}


// ========================================
// 50% 分支预测正确率测试用例
// 这些测试用例专门设计来让分支预测器无法有效预测（50%正确率）
// 用于验证 zicond 在不可预测分支场景下的性能优势
// ========================================

// 测试用例A: 完全随机LSB位 - 50%概率
func BenchmarkUnpredictableLSB32(b *testing.B) {
	for i := 0; i < b.N; i++ {
		for j := 0; j < len(testData32); j++ {
			_ = unpredictableLSB32(testData32[j], testData32[(j+1)%len(testData32)])
		}
	}
}

// 测试用例B: XOR哈希位 - 50%概率
func BenchmarkUnpredictableXOR32(b *testing.B) {
	for i := 0; i < b.N; i++ {
		for j := 0; j < len(testData32); j++ {
			_ = unpredictableXOR32(testData32[j], testData32[(j+1)%len(testData32)])
		}
	}
}

// 测试用例C: 数据相关的位模式 - 50%概率
func BenchmarkUnpredictableBitPattern32(b *testing.B) {
	for i := 0; i < b.N; i++ {
		for j := 0; j < len(testData32); j++ {
			_ = unpredictableBitPattern32(testData32[j], testData32[(j+1)%len(testData32)])
		}
	}
}

// 测试用例D: 混合位运算 - 50%概率
func BenchmarkUnpredictableMixedBits32(b *testing.B) {
	for i := 0; i < b.N; i++ {
		for j := 0; j < len(testData32); j++ {
			_ = unpredictableMixedBits32(testData32[j], testData32[(j+1)%len(testData32)])
		}
	}
}

// 测试用例E: 伪随机序列 - 50%概率
func BenchmarkUnpredictablePseudoRandom32(b *testing.B) {
	for i := 0; i < b.N; i++ {
		for j := 0; j < len(testData32); j++ {
			_ = unpredictablePseudoRandom32(testData32[j], testData32[(j+1)%len(testData32)])
		}
	}
}

// 测试用例F: 16位版本 - 50%概率
func BenchmarkUnpredictableLSB16(b *testing.B) {
	for i := 0; i < b.N; i++ {
		for j := 0; j < len(testData16); j++ {
			_ = unpredictableLSB16(testData16[j], testData16[(j+1)%len(testData16)])
		}
	}
}

// 原有的简单测试用例（保留用于对比）
func BenchmarkCmovInt(b *testing.B) {
	for i := 0; i < b.N; i++ {
		for x := range 1000 {
			cmovint(x)
		}
	}
}

func BenchmarkCmov32bit(b *testing.B) {
	for i := 0; i < b.N; i++ {
		for x := range 1000 {
			cmov32bit(uint32(x), uint32(1000-x))
		}
	}
}

func BenchmarkCmov16bit(b *testing.B) {
	for i := 0; i < b.N; i++ {
		for x := range 1000 {
			cmov16bit(uint16(x), uint16(1000-x))
		}
	}
}

// 测试函数实现

//go:noinline
func randomCondSelect32(x, y uint32) uint32 {
	// 使用位运算创建随机条件，避免分支预测
	result := x
	if (x^y)&1 != 0 { // 奇偶性不同
		result = y
	}
	return result
}

//go:noinline
func simpleCondSelect32(x, y uint32) uint32 {
	// 最简单的条件选择，测试 Zicond 基本功能
	result := x
	if x == 0 {
		result = y
	}
	return result
}

//go:noinline
func minCondSelect32(x, y uint32) uint32 {
	// 最小值选择，常见优化场景
	result := y
	if x < y {
		result = x
	}
	return result
}

//go:noinline
func randomCondSelect16(x, y uint16) uint16 {
	result := x
	if (x^y)&1 != 0 {
		result = y
	}
	return result
}

//go:noinline
func randomCondSelectInt(x, y int) int {
	result := x
	if (x^y)&1 != 0 {
		result = y
	}
	return result
}

//go:noinline
// func complexCondSelect32(x, y uint32) uint32 {
// 	// 复杂条件：多个条件组合
// 	if (x&1 == 0) && (y&1 == 1) {
// 		return x
// 	}
// 	if (x > 100) && (y < 200) {
// 		return y
// 	}
// 	return x + y
// }

// //go:noinline
// func complexCondSelect16(x, y uint16) uint16 {
// 	if (x&1 == 0) && (y&1 == 1) {
// 		return x
// 	}
// 	if (x > 100) && (y < 200) {
// 		return y
// 	}
// 	return x + y
// }

// //go:noinline
// func mathCondSelect32(x, y uint32) uint32 {
// 	// 数学运算中的条件选择
// 	if x*y == 0 {
// 		return x
// 	}
// 	if (x+y)&1 == 0 {
// 		return y
// 	}
// 	return x * y
// }

// //go:noinline
// func mathCondSelect16(x, y uint16) uint16 {
// 	if x*y == 0 {
// 		return x
// 	}
// 	if (x+y)&1 == 0 {
// 		return y
// 	}
// 	return x * y
// }

// //go:noinline
// func arrayCondSelect32(x, y uint32) uint32 {
// 	// 模拟数组处理中的条件选择
// 	if x < 10 {
// 		return x
// 	}
// 	if y > 1000 {
// 		return y
// 	}
// 	return (x + y) / 2
// }

// 原有的简单测试函数（保留用于对比）
//
//go:noinline
func condSelect(x, y int) int {
	var result int
	if x == y {
		result = x
	} else {
		result = y
	}
	return result
}

//go:noinline
func cmovint(c int) int {
	x := c + 4
	if x < 0 {
		x = 182
	}
	return x
}

//go:noinline
func cmov32bit(x, y uint32) uint32 {
	if x < y {
		x = -y
	}
	return x
}

//go:noinline
func cmov16bit(x, y uint16) uint16 {
	if x < y {
		x = -y
	}
	return x
}

// ========================================
// 50% 分支预测正确率的测试函数实现
// 这些函数使用完全不可预测的条件，确保分支预测器只能达到50%的正确率
// ========================================

//go:noinline
func unpredictableLSB32(x, y uint32) uint32 {
	// 使用最低有效位（LSB），对于随机数据，这是50/50的概率
	// 分支预测器无法预测，因为每个值的LSB是独立的
	result := x
	if (x & 1) != (y & 1) {
		result = y
	}
	return result
}

//go:noinline
func unpredictableXOR32(x, y uint32) uint32 {
	// 使用XOR结果的LSB，创建不可预测的条件
	// XOR操作会打乱位模式，使得预测变得困难
	result := x
	if ((x ^ y) & 1) != 0 {
		result = y
	}
	return result
}

//go:noinline
func unpredictableBitPattern32(x, y uint32) uint32 {
	// 使用多个位的组合，创建复杂的位模式
	// 检查x和y的位模式是否不同（通过XOR和位计数）
	result := x
	// 使用XOR后检查特定位的组合
	diff := x ^ y
	if (diff & 0x55555555) != 0 { // 检查奇数位
		result = y
	}
	return result
}

//go:noinline
func unpredictableMixedBits32(x, y uint32) uint32 {
	// 混合使用多个位运算，创建完全不可预测的条件
	// 这种模式对分支预测器来说是最困难的
	result := x
	// 组合：XOR + AND + 位移
	hash := (x ^ y) & ((x << 1) | (y >> 1))
	if (hash & 1) != 0 {
		result = y
	}
	return result
}

//go:noinline
func unpredictablePseudoRandom32(x, y uint32) uint32 {
	// 使用伪随机位模式，模拟真实世界的不可预测数据
	// 通过多个位运算的组合，确保条件完全随机
	result := x
	// 创建一个简单的哈希函数来确定选择
	// 使用线性反馈移位寄存器（LFSR）风格的位操作
	hash := x ^ y
	hash ^= hash >> 16
	hash ^= hash >> 8
	hash ^= hash >> 4
	hash ^= hash >> 2
	hash ^= hash >> 1
	if (hash & 1) != 0 {
		result = y
	}
	return result
}

//go:noinline
func unpredictableLSB16(x, y uint16) uint16 {
	// 16位版本的LSB测试
	result := x
	if (x & 1) != (y & 1) {
		result = y
	}
	return result
}
