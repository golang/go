// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package golangorgenv provides environment information for programs running at
// golang.org and its subdomains.
package golangorgenv

import (
	"log"
	"os"
	"strconv"
)

var (
	checkCountry = boolEnv("GOLANGORG_CHECK_COUNTRY")
	enforceHosts = boolEnv("GOLANGORG_ENFORCE_HOSTS")
)

// CheckCountry reports whether country restrictions should be enforced.
func CheckCountry() bool {
	return checkCountry
}

// EnforceHosts reports whether host filtering should be enforced.
func EnforceHosts() bool {
	return enforceHosts
}

func boolEnv(key string) bool {
	v := os.Getenv(key)
	if v == "" {
		// TODO(dmitshur): In the future, consider detecting if running in App Engine,
		// and if so, making the environment variables mandatory rather than optional.
		return false
	}
	b, err := strconv.ParseBool(v)
	if err != nil {
		log.Fatalf("environment variable %s (%q) must be a boolean", key, v)
	}
	return b
}
