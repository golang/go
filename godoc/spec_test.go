// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package godoc

import (
	"bytes"
	"strings"
	"testing"
)

func TestParseEBNFString(t *testing.T) {
	var p ebnfParser
	var buf bytes.Buffer
	src := []byte("octal_byte_value = `\\` octal_digit octal_digit octal_digit .")
	p.parse(&buf, src)

	if strings.Contains(buf.String(), "error") {
		t.Error(buf.String())
	}
}
