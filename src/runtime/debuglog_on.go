// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build debuglog

package runtime

const dlogEnabled = true

// dlogPerM is the per-M debug log data. This is embedded in the m
// struct.
type dlogPerM struct {
	dlogCache *dloggerImpl
}

// getCachedDlogger returns a cached dlogger if it can do so
// efficiently, or nil otherwise. The returned dlogger will be owned.
func getCachedDlogger() *dloggerImpl {
	mp := acquirem()
	// We don't return a cached dlogger if we're running on the
	// signal stack in case the signal arrived while in
	// get/putCachedDlogger. (Too bad we don't have non-atomic
	// exchange!)
	var l *dloggerImpl
	if getg() != mp.gsignal {
		l = mp.dlogCache
		mp.dlogCache = nil
	}
	releasem(mp)
	return l
}

// putCachedDlogger attempts to return l to the local cache. It
// returns false if this fails.
func putCachedDlogger(l *dloggerImpl) bool {
	mp := acquirem()
	if getg() != mp.gsignal && mp.dlogCache == nil {
		mp.dlogCache = l
		releasem(mp)
		return true
	}
	releasem(mp)
	return false
}
