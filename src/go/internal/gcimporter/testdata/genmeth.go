// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file is used to generate an object file which
// serves as test file for gcimporter_test.go.
//
// Function bodies are never read, so panics are used
// for brevity. Struct contents are also never filled
// in.

package genmeth

// monads
type Either[A, B any] struct {}

func (Either[A, B]) A() A { panic(42) }
func (Either[A, B]) B() B { panic(42) }

type Option[T any] struct {}

func (Option[T]) Get() T { panic(42) }
func (Option[T]) GetOrDefault[U any](u U) Either[T, U] { panic(42) }
func (Option[T]) MapIfPresent[U any](f func(T) U) Option[U] { panic(42) }

// pointer receiver
type Box[P any] struct {}

func (*Box[P]) Get() P { panic(42) }
func (*Box[P]) Set(p P) { panic(42) }

// streaming
type List[E any] []E
func (List[E]) Map[R any](f func(E) R) List[R] { panic(42) }
func (List[E]) FlatMap[R any](f func(E) List[R]) List[R] { panic(42) }
func (List[E1]) Zip[E2, R any](l List[E2], f func(E1, E2) R) List[R] { panic(42) }

type Pair[A, B any] struct {}
type BiList[A, B any] List[Pair[A, B]]

func (BiList[A, B]) Flip() BiList[B, A] { panic(42) }
func (BiList[A, B]) MapKeys[R any](f func(A) R) BiList[R, B] { panic(42) }
func (BiList[A, B]) MapValues[R any](f func(B) R) BiList[A, R] { panic(42) }
func (BiList[A, B]) FlatMapValues[R any](f func(B) List[R]) BiList[A, R] { panic(42) }
func (BiList[A, B]) MapEntries[K, V any](f func(Pair[A, B]) Pair[K, V]) BiList[K, V] { panic(42) }

type OrderedList[E comparable] []E
func (OrderedList[E]) Min() Option[E] { panic(42) }
func (OrderedList[E]) Max() Option[E] { panic(42) }
