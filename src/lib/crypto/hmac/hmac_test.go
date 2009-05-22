// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package hmac

// TODO(rsc): better test

import (
	"crypto/hmac";
	"crypto/md5";
	"io";
	"fmt";
	"testing";
)

func TestHMAC_MD5(t *testing.T) {
	// presotto's test
	inner := md5.New();
	h := HMAC(inner, io.StringBytes("Jefe"));
	io.WriteString(h, "what do ya want for nothing?");
	s := fmt.Sprintf("%x", h.Sum());
	answer := "750c783e6ab0b503eaa86e310a5db738";
	if s != answer {
		t.Error("have", s, "\nwant", answer);
	}
}
