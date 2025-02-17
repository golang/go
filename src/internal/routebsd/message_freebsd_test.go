// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package routebsd

import (
	"syscall"
	"testing"
)

func TestFetchRIBMessagesOnFreeBSD(t *testing.T) {
	for _, typ := range []int{syscall.NET_RT_IFMALIST} {
		ms, err := FetchRIBMessages(typ, 0)
		if err != nil {
			t.Error(typ, err)
			continue
		}
		ss, err := msgs(ms).validate()
		if err != nil {
			t.Error(typ, err)
			continue
		}
		for _, s := range ss {
			t.Log(s)
		}
	}
}
