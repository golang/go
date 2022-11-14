// Code generated from _gen/allocators.go; DO NOT EDIT.

package ssa

import (
	"math/bits"
	"sync"
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

var poolFreeBlockSlice [27]sync.Pool

func (c *Cache) allocBlockSlice(n int) []*Block {
	var s []*Block
	n2 := n
	if n2 < 32 {
		n2 = 32
	}
	b := bits.Len(uint(n2 - 1))
	v := poolFreeBlockSlice[b-5].Get()
	if v == nil {
		s = make([]*Block, 1<<b)
	} else {
		sp := v.(*[]*Block)
		s = *sp
		*sp = nil
		c.hdrBlockSlice = append(c.hdrBlockSlice, sp)
	}
	s = s[:n]
	return s
}
func (c *Cache) freeBlockSlice(s []*Block) {
	for i := range s {
		s[i] = nil
	}
	b := bits.Len(uint(cap(s)) - 1)
	var sp *[]*Block
	if len(c.hdrBlockSlice) == 0 {
		sp = new([]*Block)
	} else {
		sp = c.hdrBlockSlice[len(c.hdrBlockSlice)-1]
		c.hdrBlockSlice[len(c.hdrBlockSlice)-1] = nil
		c.hdrBlockSlice = c.hdrBlockSlice[:len(c.hdrBlockSlice)-1]
	}
	*sp = s
	poolFreeBlockSlice[b-5].Put(sp)
}

var poolFreeBoolSlice [24]sync.Pool

func (c *Cache) allocBoolSlice(n int) []bool {
	var s []bool
	n2 := n
	if n2 < 256 {
		n2 = 256
	}
	b := bits.Len(uint(n2 - 1))
	v := poolFreeBoolSlice[b-8].Get()
	if v == nil {
		s = make([]bool, 1<<b)
	} else {
		sp := v.(*[]bool)
		s = *sp
		*sp = nil
		c.hdrBoolSlice = append(c.hdrBoolSlice, sp)
	}
	s = s[:n]
	return s
}
func (c *Cache) freeBoolSlice(s []bool) {
	for i := range s {
		s[i] = false
	}
	b := bits.Len(uint(cap(s)) - 1)
	var sp *[]bool
	if len(c.hdrBoolSlice) == 0 {
		sp = new([]bool)
	} else {
		sp = c.hdrBoolSlice[len(c.hdrBoolSlice)-1]
		c.hdrBoolSlice[len(c.hdrBoolSlice)-1] = nil
		c.hdrBoolSlice = c.hdrBoolSlice[:len(c.hdrBoolSlice)-1]
	}
	*sp = s
	poolFreeBoolSlice[b-8].Put(sp)
}

var poolFreeIntSlice [27]sync.Pool

func (c *Cache) allocIntSlice(n int) []int {
	var s []int
	n2 := n
	if n2 < 32 {
		n2 = 32
	}
	b := bits.Len(uint(n2 - 1))
	v := poolFreeIntSlice[b-5].Get()
	if v == nil {
		s = make([]int, 1<<b)
	} else {
		sp := v.(*[]int)
		s = *sp
		*sp = nil
		c.hdrIntSlice = append(c.hdrIntSlice, sp)
	}
	s = s[:n]
	return s
}
func (c *Cache) freeIntSlice(s []int) {
	for i := range s {
		s[i] = 0
	}
	b := bits.Len(uint(cap(s)) - 1)
	var sp *[]int
	if len(c.hdrIntSlice) == 0 {
		sp = new([]int)
	} else {
		sp = c.hdrIntSlice[len(c.hdrIntSlice)-1]
		c.hdrIntSlice[len(c.hdrIntSlice)-1] = nil
		c.hdrIntSlice = c.hdrIntSlice[:len(c.hdrIntSlice)-1]
	}
	*sp = s
	poolFreeIntSlice[b-5].Put(sp)
}

var poolFreeInt32Slice [26]sync.Pool

func (c *Cache) allocInt32Slice(n int) []int32 {
	var s []int32
	n2 := n
	if n2 < 64 {
		n2 = 64
	}
	b := bits.Len(uint(n2 - 1))
	v := poolFreeInt32Slice[b-6].Get()
	if v == nil {
		s = make([]int32, 1<<b)
	} else {
		sp := v.(*[]int32)
		s = *sp
		*sp = nil
		c.hdrInt32Slice = append(c.hdrInt32Slice, sp)
	}
	s = s[:n]
	return s
}
func (c *Cache) freeInt32Slice(s []int32) {
	for i := range s {
		s[i] = 0
	}
	b := bits.Len(uint(cap(s)) - 1)
	var sp *[]int32
	if len(c.hdrInt32Slice) == 0 {
		sp = new([]int32)
	} else {
		sp = c.hdrInt32Slice[len(c.hdrInt32Slice)-1]
		c.hdrInt32Slice[len(c.hdrInt32Slice)-1] = nil
		c.hdrInt32Slice = c.hdrInt32Slice[:len(c.hdrInt32Slice)-1]
	}
	*sp = s
	poolFreeInt32Slice[b-6].Put(sp)
}

var poolFreeInt8Slice [24]sync.Pool

func (c *Cache) allocInt8Slice(n int) []int8 {
	var s []int8
	n2 := n
	if n2 < 256 {
		n2 = 256
	}
	b := bits.Len(uint(n2 - 1))
	v := poolFreeInt8Slice[b-8].Get()
	if v == nil {
		s = make([]int8, 1<<b)
	} else {
		sp := v.(*[]int8)
		s = *sp
		*sp = nil
		c.hdrInt8Slice = append(c.hdrInt8Slice, sp)
	}
	s = s[:n]
	return s
}
func (c *Cache) freeInt8Slice(s []int8) {
	for i := range s {
		s[i] = 0
	}
	b := bits.Len(uint(cap(s)) - 1)
	var sp *[]int8
	if len(c.hdrInt8Slice) == 0 {
		sp = new([]int8)
	} else {
		sp = c.hdrInt8Slice[len(c.hdrInt8Slice)-1]
		c.hdrInt8Slice[len(c.hdrInt8Slice)-1] = nil
		c.hdrInt8Slice = c.hdrInt8Slice[:len(c.hdrInt8Slice)-1]
	}
	*sp = s
	poolFreeInt8Slice[b-8].Put(sp)
}

var poolFreeIDSlice [26]sync.Pool

func (c *Cache) allocIDSlice(n int) []ID {
	var s []ID
	n2 := n
	if n2 < 64 {
		n2 = 64
	}
	b := bits.Len(uint(n2 - 1))
	v := poolFreeIDSlice[b-6].Get()
	if v == nil {
		s = make([]ID, 1<<b)
	} else {
		sp := v.(*[]ID)
		s = *sp
		*sp = nil
		c.hdrIDSlice = append(c.hdrIDSlice, sp)
	}
	s = s[:n]
	return s
}
func (c *Cache) freeIDSlice(s []ID) {
	for i := range s {
		s[i] = 0
	}
	b := bits.Len(uint(cap(s)) - 1)
	var sp *[]ID
	if len(c.hdrIDSlice) == 0 {
		sp = new([]ID)
	} else {
		sp = c.hdrIDSlice[len(c.hdrIDSlice)-1]
		c.hdrIDSlice[len(c.hdrIDSlice)-1] = nil
		c.hdrIDSlice = c.hdrIDSlice[:len(c.hdrIDSlice)-1]
	}
	*sp = s
	poolFreeIDSlice[b-6].Put(sp)
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
