// asmcheck

// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package codegen

import "reflect"

func intPtrTypeSize() uintptr {
	// amd64:"MOVL\t[$]8,",-"CALL"
	// arm64:"MOVD\t[$]8,",-"CALL"
	return reflect.TypeFor[*int]().Size()
}

func intPtrTypeKind() reflect.Kind {
	// amd64:"MOVL\t[$]22,",-"CALL"
	// arm64:"MOVD\t[$]22,",-"CALL"
	return reflect.TypeFor[*int]().Kind()
}
