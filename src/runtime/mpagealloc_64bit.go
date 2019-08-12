// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build amd64 !darwin,arm64 mips64 mips64le ppc64 ppc64le s390x

// See mpagealloc_32bit.go for why darwin/arm64 is excluded here.

package runtime

const (
	// The number of levels in the radix tree.
	summaryLevels = 5
)
