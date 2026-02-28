// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file contains support functions for exportdata.

package exportdata

import (
	"bufio"
	"io"
	"strconv"
	"strings"
)

// Copy of cmd/internal/archive.ReadHeader.
func readArchiveHeader(b *bufio.Reader, name string) int {
	// architecture-independent object file output
	const HeaderSize = 60

	var buf [HeaderSize]byte
	if _, err := io.ReadFull(b, buf[:]); err != nil {
		return -1
	}
	aname := strings.Trim(string(buf[0:16]), " ")
	if !strings.HasPrefix(aname, name) {
		return -1
	}
	asize := strings.Trim(string(buf[48:58]), " ")
	i, _ := strconv.Atoi(asize)
	return i
}
