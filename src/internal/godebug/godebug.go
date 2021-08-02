// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package godebug parses the GODEBUG environment variable.
package godebug

import "os"

// Get returns the value for the provided GODEBUG key.
func Get(key string) string {
	return get(os.Getenv("GODEBUG"), key)
}

// get returns the value part of key=value in s (a GODEBUG value).
func get(s, key string) string {
	for i := 0; i < len(s)-len(key)-1; i++ {
		if i > 0 && s[i-1] != ',' {
			continue
		}
		afterKey := s[i+len(key):]
		if afterKey[0] != '=' || s[i:i+len(key)] != key {
			continue
		}
		val := afterKey[1:]
		for i, b := range val {
			if b == ',' {
				return val[:i]
			}
		}
		return val
	}
	return ""
}
