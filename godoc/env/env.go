// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package env provides environment information for the godoc server running on
// golang.org.
package env

import (
	"log"
	"os"
	"strconv"
)

var (
	isProd       = boolEnv("GODOC_PROD")
	enforceHosts = boolEnv("GODOC_ENFORCE_HOSTS")
)

// IsProd reports whether the server is running in its production configuration
// on golang.org.
func IsProd() bool {
	return isProd
}

// EnforceHosts reports whether host filtering should be enforced.
func EnforceHosts() bool {
	return enforceHosts
}

func boolEnv(key string) bool {
	v := os.Getenv(key)
	if v == "" {
		return false
	}
	b, err := strconv.ParseBool(v)
	if err != nil {
		log.Fatalf("environment variable %s (%q) must be a boolean", key, v)
	}
	return b
}
