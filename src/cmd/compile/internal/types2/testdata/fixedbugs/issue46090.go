// -lang=go1.17

// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// The predeclared type comparable is not visible before Go 1.18.

package p

type _ comparable // ERROR undeclared
