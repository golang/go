// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package build

import (
	"appengine"
	"appengine/memcache"
	"json"
	"os"
)

const (
	todoCacheKey    = "build-todo"
	todoCacheExpiry = 3600 // 1 hour in seconds
	uiCacheKey      = "build-ui"
	uiCacheExpiry   = 10 * 60 // 10 minutes in seconds
)

// invalidateCache deletes the build cache records from memcache.
// This function should be called whenever the datastore changes.
func invalidateCache(c appengine.Context) {
	keys := []string{uiCacheKey, todoCacheKey}
	errs := memcache.DeleteMulti(c, keys)
	for i, err := range errs {
		if err != nil && err != memcache.ErrCacheMiss {
			c.Errorf("memcache.Delete(%q): %v", keys[i], err)
		}
	}
}

// cachedTodo gets the specified todo cache entry (if it exists) from the
// shared todo cache.
func cachedTodo(c appengine.Context, todoKey string) (todo *Todo, ok bool) {
	t := todoCache(c)
	if t == nil {
		return nil, false
	}
	todos := unmarshalTodo(c, t)
	if todos == nil {
		return nil, false
	}
	todo, ok = todos[todoKey]
	return
}

// cacheTodo puts the provided todo cache entry into the shared todo cache.
// The todo cache is a JSON-encoded map[string]*Todo, where the key is todoKey.
func cacheTodo(c appengine.Context, todoKey string, todo *Todo) {
	// Get the todo cache record (or create a new one).
	newItem := false
	t := todoCache(c)
	if t == nil {
		newItem = true
		t = &memcache.Item{
			Key:   todoCacheKey,
			Value: []byte("{}"), // default is an empty JSON object
		}
	}

	// Unmarshal the JSON value.
	todos := unmarshalTodo(c, t)
	if todos == nil {
		return
	}

	// Update the map.
	todos[todoKey] = todo

	// Marshal the updated JSON value.
	var err os.Error
	t.Value, err = json.Marshal(todos)
	if err != nil {
		// This shouldn't happen.
		c.Criticalf("marshal todo cache: %v", err)
		return
	}

	// Set a new expiry.
	t.Expiration = todoCacheExpiry

	// Update the cache record (or Set it, if new).
	if newItem {
		err = memcache.Set(c, t)
	} else {
		err = memcache.CompareAndSwap(c, t)
	}
	if err == memcache.ErrCASConflict || err == memcache.ErrNotStored {
		// No big deal if it didn't work; it should next time.
		c.Warningf("didn't update todo cache: %v", err)
	} else if err != nil {
		c.Errorf("update todo cache: %v", err)
	}
}

// todoCache gets the todo cache record from memcache (if it exists).
func todoCache(c appengine.Context) *memcache.Item {
	t, err := memcache.Get(c, todoCacheKey)
	if err != nil {
		if err != memcache.ErrCacheMiss {
			c.Errorf("get todo cache: %v", err)
		}
		return nil
	}
	return t
}

// unmarshalTodo decodes the given item's memcache value into a map.
func unmarshalTodo(c appengine.Context, t *memcache.Item) map[string]*Todo {
	todos := make(map[string]*Todo)
	if err := json.Unmarshal(t.Value, &todos); err != nil {
		// This shouldn't happen.
		c.Criticalf("unmarshal todo cache: %v", err)
		// Kill the bad record.
		if err := memcache.Delete(c, todoCacheKey); err != nil {
			c.Errorf("delete todo cache: %v", err)
		}
		return nil
	}
	return todos
}
