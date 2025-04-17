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

var poolFreeLimitSlice [27]sync.Pool

func (c *Cache) allocLimitSlice(n int) []limit {
	var s []limit
	n2 := n
	if n2 < 8 {
		n2 = 8
	}
	b := bits.Len(uint(n2 - 1))
	v := poolFreeLimitSlice[b-3].Get()
	if v == nil {
		s = make([]limit, 1<<b)
	} else {
		sp := v.(*[]limit)
		s = *sp
		*sp = nil
		c.hdrLimitSlice = append(c.hdrLimitSlice, sp)
	}
	s = s[:n]
	return s
}
func (c *Cache) freeLimitSlice(s []limit) {
	for i := range s {
		s[i] = limit{}
	}
	b := bits.Len(uint(cap(s)) - 1)
	var sp *[]limit
	if len(c.hdrLimitSlice) == 0 {
		sp = new([]limit)
	} else {
		sp = c.hdrLimitSlice[len(c.hdrLimitSlice)-1]
		c.hdrLimitSlice[len(c.hdrLimitSlice)-1] = nil
		c.hdrLimitSlice = c.hdrLimitSlice[:len(c.hdrLimitSlice)-1]
	}
	*sp = s
	poolFreeLimitSlice[b-3].Put(sp)
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
func (c *Cache) allocInt64(n int) []int64 {
	var base limit
	var derived int64
	if unsafe.Sizeof(base)%unsafe.Sizeof(derived) != 0 {
		panic("bad")
	}
	scale := unsafe.Sizeof(base) / unsafe.Sizeof(derived)
	b := c.allocLimitSlice(int((uintptr(n) + scale - 1) / scale))
	s := unsafeheader.Slice{
		Data: unsafe.Pointer(&b[0]),
		Len:  n,
		Cap:  cap(b) * int(scale),
	}
	return *(*[]int64)(unsafe.Pointer(&s))
}
func (c *Cache) freeInt64(s []int64) {
	var base limit
	var derived int64
	scale := unsafe.Sizeof(base) / unsafe.Sizeof(derived)
	b := unsafeheader.Slice{
		Data: unsafe.Pointer(&s[0]),
		Len:  int((uintptr(len(s)) + scale - 1) / scale),
		Cap:  int((uintptr(cap(s)) + scale - 1) / scale),
	}
	c.freeLimitSlice(*(*[]limit)(unsafe.Pointer(&b)))
}
func (c *Cache) allocIntSlice(n int) []int {
	var base limit
	var derived int
	if unsafe.Sizeof(base)%unsafe.Sizeof(derived) != 0 {
		panic("bad")
	}
	scale := unsafe.Sizeof(base) / unsafe.Sizeof(derived)
	b := c.allocLimitSlice(int((uintptr(n) + scale - 1) / scale))
	s := unsafeheader.Slice{
		Data: unsafe.Pointer(&b[0]),
		Len:  n,
		Cap:  cap(b) * int(scale),
	}
	return *(*[]int)(unsafe.Pointer(&s))
}
func (c *Cache) freeIntSlice(s []int) {
	var base limit
	var derived int
	scale := unsafe.Sizeof(base) / unsafe.Sizeof(derived)
	b := unsafeheader.Slice{
		Data: unsafe.Pointer(&s[0]),
		Len:  int((uintptr(len(s)) + scale - 1) / scale),
		Cap:  int((uintptr(cap(s)) + scale - 1) / scale),
	}
	c.freeLimitSlice(*(*[]limit)(unsafe.Pointer(&b)))
}
func (c *Cache) allocInt32Slice(n int) []int32 {
	var base limit
	var derived int32
	if unsafe.Sizeof(base)%unsafe.Sizeof(derived) != 0 {
		panic("bad")
	}
	scale := unsafe.Sizeof(base) / unsafe.Sizeof(derived)
	b := c.allocLimitSlice(int((uintptr(n) + scale - 1) / scale))
	s := unsafeheader.Slice{
		Data: unsafe.Pointer(&b[0]),
		Len:  n,
		Cap:  cap(b) * int(scale),
	}
	return *(*[]int32)(unsafe.Pointer(&s))
}
func (c *Cache) freeInt32Slice(s []int32) {
	var base limit
	var derived int32
	scale := unsafe.Sizeof(base) / unsafe.Sizeof(derived)
	b := unsafeheader.Slice{
		Data: unsafe.Pointer(&s[0]),
		Len:  int((uintptr(len(s)) + scale - 1) / scale),
		Cap:  int((uintptr(cap(s)) + scale - 1) / scale),
	}
	c.freeLimitSlice(*(*[]limit)(unsafe.Pointer(&b)))
}
func (c *Cache) allocInt8Slice(n int) []int8 {
	var base limit
	var derived int8
	if unsafe.Sizeof(base)%unsafe.Sizeof(derived) != 0 {
		panic("bad")
	}
	scale := unsafe.Sizeof(base) / unsafe.Sizeof(derived)
	b := c.allocLimitSlice(int((uintptr(n) + scale - 1) / scale))
	s := unsafeheader.Slice{
		Data: unsafe.Pointer(&b[0]),
		Len:  n,
		Cap:  cap(b) * int(scale),
	}
	return *(*[]int8)(unsafe.Pointer(&s))
}
func (c *Cache) freeInt8Slice(s []int8) {
	var base limit
	var derived int8
	scale := unsafe.Sizeof(base) / unsafe.Sizeof(derived)
	b := unsafeheader.Slice{
		Data: unsafe.Pointer(&s[0]),
		Len:  int((uintptr(len(s)) + scale - 1) / scale),
		Cap:  int((uintptr(cap(s)) + scale - 1) / scale),
	}
	c.freeLimitSlice(*(*[]limit)(unsafe.Pointer(&b)))
}
func (c *Cache) allocBoolSlice(n int) []bool {
	var base limit
	var derived bool
	if unsafe.Sizeof(base)%unsafe.Sizeof(derived) != 0 {
		panic("bad")
	}
	scale := unsafe.Sizeof(base) / unsafe.Sizeof(derived)
	b := c.allocLimitSlice(int((uintptr(n) + scale - 1) / scale))
	s := unsafeheader.Slice{
		Data: unsafe.Pointer(&b[0]),
		Len:  n,
		Cap:  cap(b) * int(scale),
	}
	return *(*[]bool)(unsafe.Pointer(&s))
}
func (c *Cache) freeBoolSlice(s []bool) {
	var base limit
	var derived bool
	scale := unsafe.Sizeof(base) / unsafe.Sizeof(derived)
	b := unsafeheader.Slice{
		Data: unsafe.Pointer(&s[0]),
		Len:  int((uintptr(len(s)) + scale - 1) / scale),
		Cap:  int((uintptr(cap(s)) + scale - 1) / scale),
	}
	c.freeLimitSlice(*(*[]limit)(unsafe.Pointer(&b)))
}
func (c *Cache) allocIDSlice(n int) []ID {
	var base limit
	var derived ID
	if unsafe.Sizeof(base)%unsafe.Sizeof(derived) != 0 {
		panic("bad")
	}
	scale := unsafe.Sizeof(base) / unsafe.Sizeof(derived)
	b := c.allocLimitSlice(int((uintptr(n) + scale - 1) / scale))
	s := unsafeheader.Slice{
		Data: unsafe.Pointer(&b[0]),
		Len:  n,
		Cap:  cap(b) * int(scale),
	}
	return *(*[]ID)(unsafe.Pointer(&s))
}
func (c *Cache) freeIDSlice(s []ID) {
	var base limit
	var derived ID
	scale := unsafe.Sizeof(base) / unsafe.Sizeof(derived)
	b := unsafeheader.Slice{
		Data: unsafe.Pointer(&s[0]),
		Len:  int((uintptr(len(s)) + scale - 1) / scale),
		Cap:  int((uintptr(cap(s)) + scale - 1) / scale),
	}
	c.freeLimitSlice(*(*[]limit)(unsafe.Pointer(&b)))
}
