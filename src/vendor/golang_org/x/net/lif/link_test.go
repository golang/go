// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build solaris

package lif

import (
	"fmt"
	"testing"
)

func (ll *Link) String() string {
	return fmt.Sprintf("name=%s index=%d type=%d flags=%#x mtu=%d addr=%v", ll.Name, ll.Index, ll.Type, ll.Flags, ll.MTU, llAddr(ll.Addr))
}

type linkPack struct {
	af  int
	lls []Link
}

func linkPacks() ([]linkPack, error) {
	var lps []linkPack
	for _, af := range [...]int{sysAF_UNSPEC, sysAF_INET, sysAF_INET6} {
		lls, err := Links(af, "")
		if err != nil {
			return nil, err
		}
		lps = append(lps, linkPack{af: af, lls: lls})
	}
	return lps, nil
}

func TestLinks(t *testing.T) {
	lps, err := linkPacks()
	if len(lps) == 0 && err != nil {
		t.Fatal(err)
	}
	for _, lp := range lps {
		n := 0
		for _, sll := range lp.lls {
			lls, err := Links(lp.af, sll.Name)
			if err != nil {
				t.Fatal(lp.af, sll.Name, err)
			}
			for _, ll := range lls {
				if ll.Name != sll.Name || ll.Index != sll.Index {
					t.Errorf("af=%s got %v; want %v", addrFamily(lp.af), &ll, &sll)
					continue
				}
				t.Logf("af=%s name=%s %v", addrFamily(lp.af), sll.Name, &ll)
				n++
			}
		}
		if n != len(lp.lls) {
			t.Errorf("af=%s got %d; want %d", addrFamily(lp.af), n, len(lp.lls))
			continue
		}
	}
}
