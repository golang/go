// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package x509

import "testing"

func TestCertPoolCertificates(t *testing.T) {
	pool, err := SystemCertPool()
	if err != nil {
		t.Fatal(err)
	}

	if certs := pool.Certificates(); len(certs) > 0 {
		if certs[0] = nil; pool.certs[0] == nil {
			t.Error("returned slice shouldn't share storage with pool")
		}
	}

}
