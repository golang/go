// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// hash 包提供了 hash 函数的接口。
package hash

import "io"

// Hash 是一个被所有 hash 函数实现的公共接口。
//
// 标准库的 Hash实现（比如 hash\crc32、crypto\sha256） 实现了 encoding.BinaryMarshaler 和
// encoding.BinaryUnmarshaler 接口。 封装 hash实现 可以将其内部的状态保存下来，并在以后用于其他处理，
// 而不必重新写入先前写入 hash 的数据。
// hash 状态可能包含其原始形式的输入部分，用户应该对任何可能出现的安全隐含做处理。
//
// 兼容性：将来对 hash 或者 crpto 包的任何更改都将努力保持和使用之前版本的编码的状态的兼容性。
// 也就是说，软件包的任何发行版都能够解码已之前的任何发行版写入的输入，但要考虑到诸如安全修补之类的问题。
// 有关背景信息，请参阅 Go兼容性 文档：https://golang.org/doc/go1compat。
type Hash interface {
	// 通过嵌入匿名的 io.Writer 接口向 hash 中添加更多的数据。
	// 永远不会返回错误
	io.Writer

	// Sum 将当前 hash 附加到 b 并返回结果切片。
	// 它不会改变底层的 hash 状态。
	Sum(b []byte) []byte

	// Rest 将 hash 重置为初始状态
	Reset()

	// Size 返回 Sum 切片的数量
	Size() int

	// BlockSize 返回 hash 底层块大小。
	// Write 方法可以接收任意大小的数据， 但是提供的数据是块大小的倍数是效率更高。
	BlockSize() int
}

// Hash32 是一个被所有 32 位 hash 函数实现的公共接口。
type Hash32 interface {
	Hash
	Sum32() uint32
}

// Hash64 是一个被所有 64 位 hash 函数实现的公共接口。
type Hash64 interface {
	Hash
	Sum64() uint64
}
