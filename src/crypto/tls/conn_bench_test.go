// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tls

import (
	"net"
	"testing"
)

func BenchmarkConnOutbufWithPool(b *testing.B) {
	data := make([]byte, 10240)
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		c := &Conn{config: testConfig.Clone(), conn: testConn{}}
		c.writeRecord(recordTypeApplicationData, data)
	}
}

type testConn struct{
	net.Conn
}

// mock this Write for outBuf pooling test
func (m testConn) Write(b []byte) (n int, err error) {
	return len(b), nil
}

