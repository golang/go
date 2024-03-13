// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build cgo

package cgotest

// void issue46893(void);
import "C"
import (
	"testing"
)

type s46893 int

//export goPanicIssue
func (p *s46893) goPanicIssue(_ string) ([]byte, error) {
	return []byte{0}, nil
}

func test46893(t *testing.T) {
	C.issue46893()
}
