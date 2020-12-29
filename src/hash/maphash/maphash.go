// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// maphash 包在字节序列上提供哈希函数。
// 这些哈希函数旨在用于实现哈希表或者其他数据结构，这些hash表或者数据结构需要将任意字符串或者字节序列映射到无符号 64 位整数的均匀分布。
//
// 哈希函数具有抗碰撞性，但不是加密安全的。（有关密码的使用请参考 crypto/sha256 和 crypto/sha512）
//
// 给定字节序列的哈希值在单个流程中是一致的，而在不同流程中会有不同。
package maphash

import "unsafe"

// Seed 是一个随机值，用于选择由哈希计算的指定的哈希函数。如果两个哈希使用相同的 Seed，它们将为任意给定的输入计算相同的哈希值。
// 如果两个哈希使用不同的 Seed， 它们很可能为任何给定输入计算不同的哈希值。
//
// 必须通过调用 MakeSeed 来初始化一个 Seed。zero seed 是指未初始化的，且不能用于哈希的 SetSeed 方法。
//
// 每个 Seed 对于单个流程来说都是本地的，无法序列化或以其他方式在其他流程中重新创建。
type Seed struct {
	s uint64
}

// Hash 计算字节序列的种子哈希
//
// zero Hash 是一个可以使用的有效哈希。
// zero Hash 在第一次调用 Reset、Write、Seed、Sum64 或 Seed 方法期间为其自身选择一个随机的种子。
// 要控制种子， 请使用 SetSeed。
//
// 计算得出的哈希值仅取决于初始种子和提供给哈希对象的字节序列，而不取决于提供字节的方式。
// 例如，这三个序列都具有相同的效果：
//
//     h.Write([]byte{'f','o','o'})
//     h.WriteByte('f'); h.WriteByte('o'); h.WriteByte('o')
//     h.WriteString("foo")
//
// Hash 具有抗冲击性。
// Hash 不能安全的被多个 goroutines 并发使用，但是种子可以。
// 如果多个 goroutines 必须计算相同的种子哈希，每个 goroutine 可以声明自己的 Hash 和使用公共的种子来调用 SetSeed。
type Hash struct {
	_     [0]func() // not comparable
	seed  Seed      // initial seed used for this hash
	state Seed      // current hash of all flushed bytes
	buf   [64]byte  // unflushed byte buffer
	n     int       // number of unflushed bytes
}

// initSeed seeds the hash if necessary.
// initSeed is called lazily before any operation that actually uses h.seed/h.state.
// Note that this does not include Write/WriteByte/WriteString in the case
// where they only add to h.buf. (If they write too much, they call h.flush,
// which does call h.initSeed.)
func (h *Hash) initSeed() {
	if h.seed.s == 0 {
		h.setSeed(MakeSeed())
	}
}

// WriteByte 将 b 添加到由 h 散列的字节序列中。
// 它永远不会失败；错误结果由 io.ByteWriter 的实现者决定。
func (h *Hash) WriteByte(b byte) error {
	if h.n == len(h.buf) {
		h.flush()
	}
	h.buf[h.n] = b
	h.n++
	return nil
}

// Write 将 b 添加到由 h 散列的字符序列中。
// 它总是写入所有的 b 且不会失败。 数量和错误结果由 io.ByteWriter 的实现者决定。
func (h *Hash) Write(b []byte) (int, error) {
	size := len(b)
	for h.n+len(b) > len(h.buf) {
		k := copy(h.buf[h.n:], b)
		h.n = len(h.buf)
		b = b[k:]
		h.flush()
	}
	h.n += copy(h.buf[h.n:], b)
	return size, nil
}

// WriteString 将 s 的字节添加到由 h 散列的字符序列中。
// 它总是写入所有的 s 且不会失败。 数量和错误结果由 io.ByteWriter 的实现者决定。
func (h *Hash) WriteString(s string) (int, error) {
	size := len(s)
	for h.n+len(s) > len(h.buf) {
		k := copy(h.buf[h.n:], s)
		h.n = len(h.buf)
		s = s[k:]
		h.flush()
	}
	h.n += copy(h.buf[h.n:], s)
	return size, nil
}

// Seed 返回 h 的种子值。
func (h *Hash) Seed() Seed {
	h.initSeed()
	return h.seed
}

// SetSeed 将 h 设置为使用种子，该种子必须是由 MakeSeed 或者 其他 Hash's Seed 方法返回的。
// 具有相同种子的两个哈希对象行为也相同。
// 具有不同种子的两个哈希对象行为不同。
// 在此调用之前添加到 h 的任何字节将被丢弃。
func (h *Hash) SetSeed(seed Seed) {
	h.setSeed(seed)
	h.n = 0
}

// setSeed sets seed without discarding accumulated data.
func (h *Hash) setSeed(seed Seed) {
	if seed.s == 0 {
		panic("maphash: use of uninitialized Seed")
	}
	h.seed = seed
	h.state = seed
}

// Reset 将丢弃添加到 h 的所有字节。
// （种子保持不变）
func (h *Hash) Reset() {
	h.initSeed()
	h.state = h.seed
	h.n = 0
}

// precondition: buffer is full.
func (h *Hash) flush() {
	if h.n != len(h.buf) {
		panic("maphash: flush of partially full buffer")
	}
	h.initSeed()
	h.state.s = rthash(h.buf[:], h.state.s)
	h.n = 0
}

// Sum64 返回 h 的当前 64 位值，该值取决于 h 的种子和上次调用 Reset、SetSeed 以来添加到 h 的字节序列
//
// Sum64 结果的所有位都接近均匀且独立的分布，因此可以通过 bit masking、shifting、modular arithmetic 来安全的减少。
func (h *Hash) Sum64() uint64 {
	h.initSeed()
	return rthash(h.buf[:h.n], h.state.s)
}

// MakeSeed 返回一个新的随机种子。
func MakeSeed() Seed {
	var s1, s2 uint64
	for {
		s1 = uint64(runtime_fastrand())
		s2 = uint64(runtime_fastrand())
		// We use seed 0 to indicate an uninitialized seed/hash,
		// so keep trying until we get a non-zero seed.
		if s1|s2 != 0 {
			break
		}
	}
	return Seed{s: s1<<32 + s2}
}

//go:linkname runtime_fastrand runtime.fastrand
func runtime_fastrand() uint32

func rthash(b []byte, seed uint64) uint64 {
	if len(b) == 0 {
		return seed
	}
	// The runtime hasher only works on uintptr. For 64-bit
	// architectures, we use the hasher directly. Otherwise,
	// we use two parallel hashers on the lower and upper 32 bits.
	if unsafe.Sizeof(uintptr(0)) == 8 {
		return uint64(runtime_memhash(unsafe.Pointer(&b[0]), uintptr(seed), uintptr(len(b))))
	}
	lo := runtime_memhash(unsafe.Pointer(&b[0]), uintptr(seed), uintptr(len(b)))
	hi := runtime_memhash(unsafe.Pointer(&b[0]), uintptr(seed>>32), uintptr(len(b)))
	return uint64(hi)<<32 | uint64(lo)
}

//go:linkname runtime_memhash runtime.memhash
//go:noescape
func runtime_memhash(p unsafe.Pointer, seed, s uintptr) uintptr

// Sum 将哈希的当前 64 位 值附加到 b。
// 它用于实现 hash.Hash。
// 对于直接调用，Sum64 效率更高。
func (h *Hash) Sum(b []byte) []byte {
	x := h.Sum64()
	return append(b,
		byte(x>>0),
		byte(x>>8),
		byte(x>>16),
		byte(x>>24),
		byte(x>>32),
		byte(x>>40),
		byte(x>>48),
		byte(x>>56))
}

// SIze 返回 h 的哈希值大小，即 8 字节。
func (h *Hash) Size() int { return 8 }

// BlockSize 返回 h 的块大小。
func (h *Hash) BlockSize() int { return len(h.buf) }
