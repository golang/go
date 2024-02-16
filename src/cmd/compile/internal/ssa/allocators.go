// Code generated from _gen/allocators.go using 'go generate'; DO NOT EDIT.

package ssa

import (
	"internal/unsafeheader"
	"math/bits"
	"sync"
	"unsafe"
)

var poolFreeValueSlice [27]sync.Pool

func (c *Cache) allocValueSlice(n int) []*Value {
	var s []*Value
	n2 := n
	if n2 < 32 {
		n2 = 32
	}
	b := bits.Len(uint(n2 - 1))
	v := poolFreeValueSlice[b-5].Get()
	if v == nil {
		s = make([]*Value, 1<<b)
	} else {
		sp := v.(*[]*Value)
		s = *sp
		*sp = nil
		c.hdrValueSlice = append(c.hdrValueSlice, sp)
	}
	s = s[:n]
	return s
}
func (c *Cache) freeValueSlice(s []*Value) {
	for i := range s {
		s[i] = nil
	}
	b := bits.Len(uint(cap(s)) - 1)
	var sp *[]*Value
	if len(c.hdrValueSlice) == 0 {
		sp = new([]*Value)
	} else {
		sp = c.hdrValueSlice[len(c.hdrValueSlice)-1]
		c.hdrValueSlice[len(c.hdrValueSlice)-1] = nil
		c.hdrValueSlice = c.hdrValueSlice[:len(c.hdrValueSlice)-1]
	}
	*sp = s
	poolFreeValueSlice[b-5].Put(sp)
}

var poolFreeInt64Slice [27]sync.Pool

func (c *Cache) allocInt64Slice(n int) []int64 {
	var s []int64
	n2 := n
	if n2 < 32 {
		n2 = 32
	}
	b := bits.Len(uint(n2 - 1))
	v := poolFreeInt64Slice[b-5].Get()
	if v == nil {
		s = make([]int64, 1<<b)
	} else {
		sp := v.(*[]int64)
		s = *sp
		*sp = nil
		c.hdrInt64Slice = append(c.hdrInt64Slice, sp)
	}
	s = s[:n]
	return s
}
func (c *Cache) freeInt64Slice(s []int64) {
	for i := range s {
		s[i] = 0
	}
	b := bits.Len(uint(cap(s)) - 1)
	var sp *[]int64
	if len(c.hdrInt64Slice) == 0 {
		sp = new([]int64)
	} else {
		sp = c.hdrInt64Slice[len(c.hdrInt64Slice)-1]
		c.hdrInt64Slice[len(c.hdrInt64Slice)-1] = nil
		c.hdrInt64Slice = c.hdrInt64Slice[:len(c.hdrInt64Slice)-1]
	}
	*sp = s
	poolFreeInt64Slice[b-5].Put(sp)
}

var poolFreeSparseSet [27]sync.Pool

func (c *Cache) allocSparseSet(n int) *sparseSet {
	var s *sparseSet
	n2 := n
	if n2 < 32 {
		n2 = 32
	}
	b := bits.Len(uint(n2 - 1))
	v := poolFreeSparseSet[b-5].Get()
	if v == nil {
		s = newSparseSet(1 << b)
	} else {
		s = v.(*sparseSet)
	}
	return s
}
func (c *Cache) freeSparseSet(s *sparseSet) {
	s.clear()
	b := bits.Len(uint(s.cap()) - 1)
	poolFreeSparseSet[b-5].Put(s)
}

var poolFreeSparseMap [27]sync.Pool

func (c *Cache) allocSparseMap(n int) *sparseMap {
	var s *sparseMap
	n2 := n
	if n2 < 32 {
		n2 = 32
	}
	b := bits.Len(uint(n2 - 1))
	v := poolFreeSparseMap[b-5].Get()
	if v == nil {
		s = newSparseMap(1 << b)
	} else {
		s = v.(*sparseMap)
	}
	return s
}
func (c *Cache) freeSparseMap(s *sparseMap) {
	s.clear()
	b := bits.Len(uint(s.cap()) - 1)
	poolFreeSparseMap[b-5].Put(s)
}

var poolFreeSparseMapPos [27]sync.Pool

func (c *Cache) allocSparseMapPos(n int) *sparseMapPos {
	var s *sparseMapPos
	n2 := n
	if n2 < 32 {
		n2 = 32
	}
	b := bits.Len(uint(n2 - 1))
	v := poolFreeSparseMapPos[b-5].Get()
	if v == nil {
		s = newSparseMapPos(1 << b)
	} else {
		s = v.(*sparseMapPos)
	}
	return s
}
func (c *Cache) freeSparseMapPos(s *sparseMapPos) {
	s.clear()
	b := bits.Len(uint(s.cap()) - 1)
	poolFreeSparseMapPos[b-5].Put(s)
}
func (c *Cache) allocBlockSlice(n int) []*Block {
	var base *Value
	var derived *Block
	if unsafe.Sizeof(base)%unsafe.Sizeof(derived) != 0 {
		panic("bad")
	}
	scale := unsafe.Sizeof(base) / unsafe.Sizeof(derived)
	b := c.allocValueSlice(int((uintptr(n) + scale - 1) / scale))
	s := unsafeheader.Slice{
		Data: unsafe.Pointer(&b[0]),
		Len:  n,
		Cap:  cap(b) * int(scale),
	}
	return *(*[]*Block)(unsafe.Pointer(&s))
}
func (c *Cache) freeBlockSlice(s []*Block) {
	var base *Value
	var derived *Block
	scale := unsafe.Sizeof(base) / unsafe.Sizeof(derived)
	b := unsafeheader.Slice{
		Data: unsafe.Pointer(&s[0]),
		Len:  int((uintptr(len(s)) + scale - 1) / scale),
		Cap:  int((uintptr(cap(s)) + scale - 1) / scale),
	}
	c.freeValueSlice(*(*[]*Value)(unsafe.Pointer(&b)))
}
func (c *Cache) allocIntSlice(n int) []int {
	var base int64
	var derived int
	if unsafe.Sizeof(base)%unsafe.Sizeof(derived) != 0 {
		panic("bad")
	}
	scale := unsafe.Sizeof(base) / unsafe.Sizeof(derived)
	b := c.allocInt64Slice(int((uintptr(n) + scale - 1) / scale))
	s := unsafeheader.Slice{
		Data: unsafe.Pointer(&b[0]),
		Len:  n,
		Cap:  cap(b) * int(scale),
	}
	return *(*[]int)(unsafe.Pointer(&s))
}
func (c *Cache) freeIntSlice(s []int) {
	var base int64
	var derived int
	scale := unsafe.Sizeof(base) / unsafe.Sizeof(derived)
	b := unsafeheader.Slice{
		Data: unsafe.Pointer(&s[0]),
		Len:  int((uintptr(len(s)) + scale - 1) / scale),
		Cap:  int((uintptr(cap(s)) + scale - 1) / scale),
	}
	c.freeInt64Slice(*(*[]int64)(unsafe.Pointer(&b)))
}
func (c *Cache) allocInt32Slice(n int) []int32 {
	var base int64
	var derived int32
	if unsafe.Sizeof(base)%unsafe.Sizeof(derived) != 0 {
		panic("bad")
	}
	scale := unsafe.Sizeof(base) / unsafe.Sizeof(derived)
	b := c.allocInt64Slice(int((uintptr(n) + scale - 1) / scale))
	s := unsafeheader.Slice{
		Data: unsafe.Pointer(&b[0]),
		Len:  n,
		Cap:  cap(b) * int(scale),
	}
	return *(*[]int32)(unsafe.Pointer(&s))
}
func (c *Cache) freeInt32Slice(s []int32) {
	var base int64
	var derived int32
	scale := unsafe.Sizeof(base) / unsafe.Sizeof(derived)
	b := unsafeheader.Slice{
		Data: unsafe.Pointer(&s[0]),
		Len:  int((uintptr(len(s)) + scale - 1) / scale),
		Cap:  int((uintptr(cap(s)) + scale - 1) / scale),
	}
	c.freeInt64Slice(*(*[]int64)(unsafe.Pointer(&b)))
}
func (c *Cache) allocInt8Slice(n int) []int8 {
	var base int64
	var derived int8
	if unsafe.Sizeof(base)%unsafe.Sizeof(derived) != 0 {
		panic("bad")
	}
	scale := unsafe.Sizeof(base) / unsafe.Sizeof(derived)
	b := c.allocInt64Slice(int((uintptr(n) + scale - 1) / scale))
	s := unsafeheader.Slice{
		Data: unsafe.Pointer(&b[0]),
		Len:  n,
		Cap:  cap(b) * int(scale),
	}
	return *(*[]int8)(unsafe.Pointer(&s))
}
func (c *Cache) freeInt8Slice(s []int8) {
	var base int64
	var derived int8
	scale := unsafe.Sizeof(base) / unsafe.Sizeof(derived)
	b := unsafeheader.Slice{
		Data: unsafe.Pointer(&s[0]),
		Len:  int((uintptr(len(s)) + scale - 1) / scale),
		Cap:  int((uintptr(cap(s)) + scale - 1) / scale),
	}
	c.freeInt64Slice(*(*[]int64)(unsafe.Pointer(&b)))
}
func (c *Cache) allocBoolSlice(n int) []bool {
	var base int64
	var derived bool
	if unsafe.Sizeof(base)%unsafe.Sizeof(derived) != 0 {
		panic("bad")
	}
	scale := unsafe.Sizeof(base) / unsafe.Sizeof(derived)
	b := c.allocInt64Slice(int((uintptr(n) + scale - 1) / scale))
	s := unsafeheader.Slice{
		Data: unsafe.Pointer(&b[0]),
		Len:  n,
		Cap:  cap(b) * int(scale),
	}
	return *(*[]bool)(unsafe.Pointer(&s))
}
func (c *Cache) freeBoolSlice(s []bool) {
	var base int64
	var derived bool
	scale := unsafe.Sizeof(base) / unsafe.Sizeof(derived)
	b := unsafeheader.Slice{
		Data: unsafe.Pointer(&s[0]),
		Len:  int((uintptr(len(s)) + scale - 1) / scale),
		Cap:  int((uintptr(cap(s)) + scale - 1) / scale),
	}
	c.freeInt64Slice(*(*[]int64)(unsafe.Pointer(&b)))
}
func (c *Cache) allocIDSlice(n int) []ID {
	var base int64
	var derived ID
	if unsafe.Sizeof(base)%unsafe.Sizeof(derived) != 0 {
		panic("bad")
	}
	scale := unsafe.Sizeof(base) / unsafe.Sizeof(derived)
	b := c.allocInt64Slice(int((uintptr(n) + scale - 1) / scale))
	s := unsafeheader.Slice{
		Data: unsafe.Pointer(&b[0]),
		Len:  n,
		Cap:  cap(b) * int(scale),
	}
	return *(*[]ID)(unsafe.Pointer(&s))
}
func (c *Cache) freeIDSlice(s []ID) {
	var base int64
	var derived ID
	scale := unsafe.Sizeof(base) / unsafe.Sizeof(derived)
	b := unsafeheader.Slice{
		Data: unsafe.Pointer(&s[0]),
		Len:  int((uintptr(len(s)) + scale - 1) / scale),
		Cap:  int((uintptr(cap(s)) + scale - 1) / scale),
	}
	c.freeInt64Slice(*(*[]int64)(unsafe.Pointer(&b)))
}
