// asmcheck

// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package codegen

import "reflect"

func f() reflect.Type {
	// amd64:`LEAQ\stype:\*int\(SB\)`
	// arm64:`MOVD\s\$type:\*int\(SB\)`
	return reflect.TypeFor[*int]()
}

func g() reflect.Type {
        // amd64:`LEAQ\stype:int\(SB\)`
        // arm64:`MOVD\s\$type:int\(SB\)`
        return reflect.TypeFor[int]()
}

