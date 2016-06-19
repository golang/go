// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sys

type ArchFamilyType int

const (
	AMD64 ArchFamilyType = iota
	ARM
	ARM64
	I386
	MIPS64
	PPC64
	S390X
)
