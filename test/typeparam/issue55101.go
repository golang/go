// compile

// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

func F() *Cache[error] { return nil }

type Cache[T any] struct{ l *List[entry[T]] }
type entry[T any] struct{ value T }
type List[T any] struct{ len int }

func (c *Cache[V]) Len() int { return c.l.Len() }
func (l *List[T]) Len() int  { return l.len }
