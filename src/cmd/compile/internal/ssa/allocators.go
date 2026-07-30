// Code generated from _gen/allocators.go using 'go generate'; DO NOT EDIT.

package ssa

import (
	"internal/unsafeheader"
	"math/bits"
	"sync"
	"unsafe"
)

var poolFreeValueSlice [27]sync.Pool

func (c *Cache) AllocValueSlice(n int) []*Value {
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
func (c *Cache) FreeValueSlice(s []*Value) {
	clear(s)
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

func (c *Cache) AllocLimitSlice(n int) []Limit {
	var s []Limit
	n2 := n
	if n2 < 8 {
		n2 = 8
	}
	b := bits.Len(uint(n2 - 1))
	v := poolFreeLimitSlice[b-3].Get()
	if v == nil {
		s = make([]Limit, 1<<b)
	} else {
		sp := v.(*[]Limit)
		s = *sp
		*sp = nil
		c.hdrLimitSlice = append(c.hdrLimitSlice, sp)
	}
	s = s[:n]
	return s
}
func (c *Cache) FreeLimitSlice(s []Limit) {
	clear(s)
	b := bits.Len(uint(cap(s)) - 1)
	var sp *[]Limit
	if len(c.hdrLimitSlice) == 0 {
		sp = new([]Limit)
	} else {
		sp = c.hdrLimitSlice[len(c.hdrLimitSlice)-1]
		c.hdrLimitSlice[len(c.hdrLimitSlice)-1] = nil
		c.hdrLimitSlice = c.hdrLimitSlice[:len(c.hdrLimitSlice)-1]
	}
	*sp = s
	poolFreeLimitSlice[b-3].Put(sp)
}

var poolFreeSparseSet [27]sync.Pool

func (c *Cache) AllocSparseSet(n int) *SparseSet {
	var s *SparseSet
	n2 := n
	if n2 < 32 {
		n2 = 32
	}
	b := bits.Len(uint(n2 - 1))
	v := poolFreeSparseSet[b-5].Get()
	if v == nil {
		s = NewSparseSet(1 << b)
	} else {
		s = v.(*SparseSet)
	}
	return s
}
func (c *Cache) FreeSparseSet(s *SparseSet) {
	s.Clear()
	b := bits.Len(uint(s.cap()) - 1)
	poolFreeSparseSet[b-5].Put(s)
}

var poolFreeSparseMap [27]sync.Pool

func (c *Cache) AllocSparseMap(n int) *sparseMap {
	var s *sparseMap
	n2 := n
	if n2 < 32 {
		n2 = 32
	}
	b := bits.Len(uint(n2 - 1))
	v := poolFreeSparseMap[b-5].Get()
	if v == nil {
		s = NewSparseMap(1 << b)
	} else {
		s = v.(*sparseMap)
	}
	return s
}
func (c *Cache) FreeSparseMap(s *sparseMap) {
	s.Clear()
	b := bits.Len(uint(s.cap()) - 1)
	poolFreeSparseMap[b-5].Put(s)
}

var poolFreeSparseMapPos [27]sync.Pool

func (c *Cache) AllocSparseMapPos(n int) *SparseMapPos {
	var s *SparseMapPos
	n2 := n
	if n2 < 32 {
		n2 = 32
	}
	b := bits.Len(uint(n2 - 1))
	v := poolFreeSparseMapPos[b-5].Get()
	if v == nil {
		s = newSparseMapPos(1 << b)
	} else {
		s = v.(*SparseMapPos)
	}
	return s
}
func (c *Cache) FreeSparseMapPos(s *SparseMapPos) {
	s.Clear()
	b := bits.Len(uint(s.cap()) - 1)
	poolFreeSparseMapPos[b-5].Put(s)
}
func (c *Cache) AllocBlockSlice(n int) []*Block {
	var base *Value
	var derived *Block
	if unsafe.Sizeof(base)%unsafe.Sizeof(derived) != 0 {
		panic("bad")
	}
	scale := unsafe.Sizeof(base) / unsafe.Sizeof(derived)
	b := c.AllocValueSlice(int((uintptr(n) + scale - 1) / scale))
	s := unsafeheader.Slice{
		Data: unsafe.Pointer(&b[0]),
		Len:  n,
		Cap:  cap(b) * int(scale),
	}
	return *(*[]*Block)(unsafe.Pointer(&s))
}
func (c *Cache) FreeBlockSlice(s []*Block) {
	var base *Value
	var derived *Block
	scale := unsafe.Sizeof(base) / unsafe.Sizeof(derived)
	b := unsafeheader.Slice{
		Data: unsafe.Pointer(&s[0]),
		Len:  int((uintptr(len(s)) + scale - 1) / scale),
		Cap:  int((uintptr(cap(s)) + scale - 1) / scale),
	}
	c.FreeValueSlice(*(*[]*Value)(unsafe.Pointer(&b)))
}
func (c *Cache) AllocInt64(n int) []int64 {
	var base Limit
	var derived int64
	if unsafe.Sizeof(base)%unsafe.Sizeof(derived) != 0 {
		panic("bad")
	}
	scale := unsafe.Sizeof(base) / unsafe.Sizeof(derived)
	b := c.AllocLimitSlice(int((uintptr(n) + scale - 1) / scale))
	s := unsafeheader.Slice{
		Data: unsafe.Pointer(&b[0]),
		Len:  n,
		Cap:  cap(b) * int(scale),
	}
	return *(*[]int64)(unsafe.Pointer(&s))
}
func (c *Cache) FreeInt64(s []int64) {
	var base Limit
	var derived int64
	scale := unsafe.Sizeof(base) / unsafe.Sizeof(derived)
	b := unsafeheader.Slice{
		Data: unsafe.Pointer(&s[0]),
		Len:  int((uintptr(len(s)) + scale - 1) / scale),
		Cap:  int((uintptr(cap(s)) + scale - 1) / scale),
	}
	c.FreeLimitSlice(*(*[]Limit)(unsafe.Pointer(&b)))
}
func (c *Cache) AllocIntSlice(n int) []int {
	var base Limit
	var derived int
	if unsafe.Sizeof(base)%unsafe.Sizeof(derived) != 0 {
		panic("bad")
	}
	scale := unsafe.Sizeof(base) / unsafe.Sizeof(derived)
	b := c.AllocLimitSlice(int((uintptr(n) + scale - 1) / scale))
	s := unsafeheader.Slice{
		Data: unsafe.Pointer(&b[0]),
		Len:  n,
		Cap:  cap(b) * int(scale),
	}
	return *(*[]int)(unsafe.Pointer(&s))
}
func (c *Cache) FreeIntSlice(s []int) {
	var base Limit
	var derived int
	scale := unsafe.Sizeof(base) / unsafe.Sizeof(derived)
	b := unsafeheader.Slice{
		Data: unsafe.Pointer(&s[0]),
		Len:  int((uintptr(len(s)) + scale - 1) / scale),
		Cap:  int((uintptr(cap(s)) + scale - 1) / scale),
	}
	c.FreeLimitSlice(*(*[]Limit)(unsafe.Pointer(&b)))
}
func (c *Cache) AllocInt32Slice(n int) []int32 {
	var base Limit
	var derived int32
	if unsafe.Sizeof(base)%unsafe.Sizeof(derived) != 0 {
		panic("bad")
	}
	scale := unsafe.Sizeof(base) / unsafe.Sizeof(derived)
	b := c.AllocLimitSlice(int((uintptr(n) + scale - 1) / scale))
	s := unsafeheader.Slice{
		Data: unsafe.Pointer(&b[0]),
		Len:  n,
		Cap:  cap(b) * int(scale),
	}
	return *(*[]int32)(unsafe.Pointer(&s))
}
func (c *Cache) FreeInt32Slice(s []int32) {
	var base Limit
	var derived int32
	scale := unsafe.Sizeof(base) / unsafe.Sizeof(derived)
	b := unsafeheader.Slice{
		Data: unsafe.Pointer(&s[0]),
		Len:  int((uintptr(len(s)) + scale - 1) / scale),
		Cap:  int((uintptr(cap(s)) + scale - 1) / scale),
	}
	c.FreeLimitSlice(*(*[]Limit)(unsafe.Pointer(&b)))
}
func (c *Cache) AllocInt8Slice(n int) []int8 {
	var base Limit
	var derived int8
	if unsafe.Sizeof(base)%unsafe.Sizeof(derived) != 0 {
		panic("bad")
	}
	scale := unsafe.Sizeof(base) / unsafe.Sizeof(derived)
	b := c.AllocLimitSlice(int((uintptr(n) + scale - 1) / scale))
	s := unsafeheader.Slice{
		Data: unsafe.Pointer(&b[0]),
		Len:  n,
		Cap:  cap(b) * int(scale),
	}
	return *(*[]int8)(unsafe.Pointer(&s))
}
func (c *Cache) FreeInt8Slice(s []int8) {
	var base Limit
	var derived int8
	scale := unsafe.Sizeof(base) / unsafe.Sizeof(derived)
	b := unsafeheader.Slice{
		Data: unsafe.Pointer(&s[0]),
		Len:  int((uintptr(len(s)) + scale - 1) / scale),
		Cap:  int((uintptr(cap(s)) + scale - 1) / scale),
	}
	c.FreeLimitSlice(*(*[]Limit)(unsafe.Pointer(&b)))
}
func (c *Cache) AllocBoolSlice(n int) []bool {
	var base Limit
	var derived bool
	if unsafe.Sizeof(base)%unsafe.Sizeof(derived) != 0 {
		panic("bad")
	}
	scale := unsafe.Sizeof(base) / unsafe.Sizeof(derived)
	b := c.AllocLimitSlice(int((uintptr(n) + scale - 1) / scale))
	s := unsafeheader.Slice{
		Data: unsafe.Pointer(&b[0]),
		Len:  n,
		Cap:  cap(b) * int(scale),
	}
	return *(*[]bool)(unsafe.Pointer(&s))
}
func (c *Cache) FreeBoolSlice(s []bool) {
	var base Limit
	var derived bool
	scale := unsafe.Sizeof(base) / unsafe.Sizeof(derived)
	b := unsafeheader.Slice{
		Data: unsafe.Pointer(&s[0]),
		Len:  int((uintptr(len(s)) + scale - 1) / scale),
		Cap:  int((uintptr(cap(s)) + scale - 1) / scale),
	}
	c.FreeLimitSlice(*(*[]Limit)(unsafe.Pointer(&b)))
}
func (c *Cache) AllocIDSlice(n int) []ID {
	var base Limit
	var derived ID
	if unsafe.Sizeof(base)%unsafe.Sizeof(derived) != 0 {
		panic("bad")
	}
	scale := unsafe.Sizeof(base) / unsafe.Sizeof(derived)
	b := c.AllocLimitSlice(int((uintptr(n) + scale - 1) / scale))
	s := unsafeheader.Slice{
		Data: unsafe.Pointer(&b[0]),
		Len:  n,
		Cap:  cap(b) * int(scale),
	}
	return *(*[]ID)(unsafe.Pointer(&s))
}
func (c *Cache) FreeIDSlice(s []ID) {
	var base Limit
	var derived ID
	scale := unsafe.Sizeof(base) / unsafe.Sizeof(derived)
	b := unsafeheader.Slice{
		Data: unsafe.Pointer(&s[0]),
		Len:  int((uintptr(len(s)) + scale - 1) / scale),
		Cap:  int((uintptr(cap(s)) + scale - 1) / scale),
	}
	c.FreeLimitSlice(*(*[]Limit)(unsafe.Pointer(&b)))
}
func (c *Cache) AllocUintSlice(n int) []uint {
	var base Limit
	var derived uint
	if unsafe.Sizeof(base)%unsafe.Sizeof(derived) != 0 {
		panic("bad")
	}
	scale := unsafe.Sizeof(base) / unsafe.Sizeof(derived)
	b := c.AllocLimitSlice(int((uintptr(n) + scale - 1) / scale))
	s := unsafeheader.Slice{
		Data: unsafe.Pointer(&b[0]),
		Len:  n,
		Cap:  cap(b) * int(scale),
	}
	return *(*[]uint)(unsafe.Pointer(&s))
}
func (c *Cache) FreeUintSlice(s []uint) {
	var base Limit
	var derived uint
	scale := unsafe.Sizeof(base) / unsafe.Sizeof(derived)
	b := unsafeheader.Slice{
		Data: unsafe.Pointer(&s[0]),
		Len:  int((uintptr(len(s)) + scale - 1) / scale),
		Cap:  int((uintptr(cap(s)) + scale - 1) / scale),
	}
	c.FreeLimitSlice(*(*[]Limit)(unsafe.Pointer(&b)))
}
func (c *Cache) AllocKnownBitsEntriesSlice(n int) []knownBitsEntry {
	var base Limit
	var derived knownBitsEntry
	if unsafe.Sizeof(base)%unsafe.Sizeof(derived) != 0 {
		panic("bad")
	}
	scale := unsafe.Sizeof(base) / unsafe.Sizeof(derived)
	b := c.AllocLimitSlice(int((uintptr(n) + scale - 1) / scale))
	s := unsafeheader.Slice{
		Data: unsafe.Pointer(&b[0]),
		Len:  n,
		Cap:  cap(b) * int(scale),
	}
	return *(*[]knownBitsEntry)(unsafe.Pointer(&s))
}
func (c *Cache) FreeKnownBitsEntriesSlice(s []knownBitsEntry) {
	var base Limit
	var derived knownBitsEntry
	scale := unsafe.Sizeof(base) / unsafe.Sizeof(derived)
	b := unsafeheader.Slice{
		Data: unsafe.Pointer(&s[0]),
		Len:  int((uintptr(len(s)) + scale - 1) / scale),
		Cap:  int((uintptr(cap(s)) + scale - 1) / scale),
	}
	c.FreeLimitSlice(*(*[]Limit)(unsafe.Pointer(&b)))
}
