// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cache

import (
	"fmt"
	"net/http"
	"time"

	"appengine"
	"appengine/memcache"
)

const (
	nocache = "nocache"
	timeKey = "cachetime"
	expiry  = 600 // 10 minutes
)

func newTime() uint64 { return uint64(time.Now().Unix()) << 32 }

// Now returns the current logical datastore time to use for cache lookups.
func Now(c appengine.Context) uint64 {
	t, err := memcache.Increment(c, timeKey, 0, newTime())
	if err != nil {
		c.Errorf("cache.Now: %v", err)
		return 0
	}
	return t
}

// Tick sets the current logical datastore time to a never-before-used time
// and returns that time. It should be called to invalidate the cache.
func Tick(c appengine.Context) uint64 {
	t, err := memcache.Increment(c, timeKey, 1, newTime())
	if err != nil {
		c.Errorf("cache.Tick: %v", err)
		return 0
	}
	return t
}

// Get fetches data for name at time now from memcache and unmarshals it into
// value. It reports whether it found the cache record and logs any errors to
// the admin console.
func Get(r *http.Request, now uint64, name string, value interface{}) bool {
	if now == 0 || r.FormValue(nocache) != "" {
		return false
	}
	c := appengine.NewContext(r)
	key := fmt.Sprintf("%s.%d", name, now)
	_, err := memcache.JSON.Get(c, key, value)
	if err == nil {
		c.Debugf("cache hit %q", key)
		return true
	}
	c.Debugf("cache miss %q", key)
	if err != memcache.ErrCacheMiss {
		c.Errorf("get cache %q: %v", key, err)
	}
	return false
}

// Set puts value into memcache under name at time now.
// It logs any errors to the admin console.
func Set(r *http.Request, now uint64, name string, value interface{}) {
	if now == 0 || r.FormValue(nocache) != "" {
		return
	}
	c := appengine.NewContext(r)
	key := fmt.Sprintf("%s.%d", name, now)
	err := memcache.JSON.Set(c, &memcache.Item{
		Key:        key,
		Object:     value,
		Expiration: expiry,
	})
	if err != nil {
		c.Errorf("set cache %q: %v", key, err)
	}
}
