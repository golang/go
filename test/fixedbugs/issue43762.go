// errorcheck

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

var _ = true == '\\' // ERROR "invalid operation: true == '\\\\'|cannot convert true"
var _ = true == '\'' // ERROR "invalid operation: true == '\\''|cannot convert true"
var _ = true == '\n' // ERROR "invalid operation: true == '\\n'|cannot convert true"
