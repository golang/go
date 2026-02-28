// -lang=go1.8

// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package aliasTypes

type _ = /* ERROR "type alias requires go1.9 or later" */ int
type _[P /* ERROR "generic type alias requires go1.23 or later" */ interface{}] = int
