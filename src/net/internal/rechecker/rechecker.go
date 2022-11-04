// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rechecker

import (
	"io"
	"os"
	"sync/atomic"
	"time"

	"sync"
)

type value[T any] struct {
	v   *T
	err error
}

type Rechecker[T any] struct {
	File     string
	Duration time.Duration
	Parse    func(content []byte) (*T, error)

	val         atomic.Pointer[value[T]]
	once        sync.Once
	recheckSema atomic.Bool

	lastCheched time.Time
	modTime     time.Time
	size        int64
}

func (r *Rechecker[T]) Get() (v *T, err error) {
	var val *value[T]

	r.once.Do(func() {
		val = &value[T]{}
		val.v, r.modTime, r.size, val.err = r.initialFileParse()
		r.lastCheched = time.Now()
		r.val.Store(val)
	})

	if val != nil {
		return val.v, val.err
	}

	val = r.val.Load()

	// one goroutine at a time
	if r.recheckSema.CompareAndSwap(false, true) {
		defer r.recheckSema.Store(false)

		now := time.Now()
		if now.After(r.lastCheched.Add(r.Duration)) {
			r.lastCheched = now

			stat, err := os.Stat(r.File)
			if err != nil {
				val = &value[T]{err: err}
				r.val.Store(val)
				return nil, err
			}

			if stat.Size() != r.size || !stat.ModTime().Equal(r.modTime) {
				val = &value[T]{}
				val.v, val.err = r.recheckParse()
				r.modTime = stat.ModTime()
				r.size = stat.Size()
				r.val.Store(val)
				return val.v, val.err
			}
		}
	}

	return val.v, val.err
}

func (r *Rechecker[T]) recheckParse() (val *T, err error) {
	f, err := os.OpenFile(r.File, os.O_RDONLY, 0)
	if err != nil {
		return nil, err
	}

	data, err := io.ReadAll(f)
	if err != nil {
		return nil, err
	}

	return r.Parse(data)
}

func (r *Rechecker[T]) initialFileParse() (val *T, modTime time.Time, size int64, err error) {
	f, err := os.OpenFile(r.File, os.O_RDONLY, 0)
	if err != nil {
		return nil, time.Time{}, 0, err
	}

	stat, err := f.Stat()
	if err != nil {
		return nil, time.Time{}, 0, err
	}

	data, err := io.ReadAll(f)
	if err != nil {
		return nil, time.Time{}, 0, err
	}

	val, err = r.Parse(data)
	return val, stat.ModTime(), stat.Size(), err
}
