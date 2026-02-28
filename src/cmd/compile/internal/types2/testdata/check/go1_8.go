// -lang=go1.8

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Check Go language version-specific errors.

package p

// type alias declarations
type any /* ERROR type aliases requires go1.9 or later */ = interface{}
