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

// Get on the initial call reads r.File and calls r.Parse with the contents of that file.
// On next calls when more than r.Duration has passed, Get stats the r.File to detect
// changes to it, when it is modified it calls r.Parse again. Get is safe to call concurrently.
func (r *Rechecker[T]) Get() (v *T, err error) {
	var initVal *value[T]

	r.once.Do(func() {
		initVal = &value[T]{}
		initVal.v, r.modTime, r.size, initVal.err = r.initialFileParse()
		r.lastCheched = time.Now()
		r.val.Store(initVal)
	})

	if initVal != nil {
		return initVal.v, initVal.err
	}

	// one goroutine at a time
	if r.recheckSema.CompareAndSwap(false, true) {
		defer r.recheckSema.Store(false)
		val := r.val.Load()

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

		return val.v, val.err
	}

	val := r.val.Load()
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

// ForceNextValues changes the values returned in the next Get() calls for at least addDur.
// It should be used only inside tests.
func (r *Rechecker[T]) ForceNextValues(v *T, err error, addDur time.Duration) bool {
	if r.recheckSema.CompareAndSwap(false, true) {
		defer r.recheckSema.Store(false)
		r.lastCheched = time.Now().Add(addDur)
		r.val.Store(&value[T]{v: v, err: err})
		return true
	}
	return false
}
