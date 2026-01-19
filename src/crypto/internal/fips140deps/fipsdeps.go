// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package fipsdeps contains wrapper packages for internal APIs that are exposed
// to the FIPS module. Since modules are frozen upon validation and supported
// for a number of future versions, APIs exposed by crypto/internal/fips140deps/...
// must not be changed until the modules that use them are no longer supported.
package fipsdeps
