// errorcheck

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

var _ = true == '\\' // ERROR "invalid operation: (cannot compare true)|(true) == '\\\\' \(mismatched types untyped bool and untyped rune\)"
var _ = true == '\'' // ERROR "invalid operation: (cannot compare true)|(true) == '\\'' \(mismatched types untyped bool and untyped rune\)"
var _ = true == '\n' // ERROR "invalid operation: (cannot compare true)|(true) == '\\n' \(mismatched types untyped bool and untyped rune\)"
