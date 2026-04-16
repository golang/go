// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package quic

import "time"

func (c *Conn) ping(space numberSpace) {
	c.sendMsg(func(now time.Time, c *Conn) {
		c.testSendPing.setUnsent()
		c.testSendPingSpace = space
	})
}
