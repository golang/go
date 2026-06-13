// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package http2

import (
	"sync/atomic"
	"time"
)

type dateCache struct {
	sec int64
	str string
}

var globalDateCache atomic.Value // of dateCache

// cachedDate returns the current time in [TimeFormat], updating at most once
// per UTC second. The returned string is shared and must not be mutated.
func cachedDate() string {
	now := time.Now()
	sec := now.Unix()
	if c, ok := globalDateCache.Load().(dateCache); ok && c.sec == sec {
		return c.str
	}
	s := now.UTC().Format(TimeFormat)
	globalDateCache.Store(dateCache{sec: sec, str: s})
	return s
}
