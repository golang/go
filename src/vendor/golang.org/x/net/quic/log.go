// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package quic

import (
	"fmt"
	"os"
	"strings"
)

var logPackets bool

// Parse GODEBUG settings.
//
// GODEBUG=quiclogpackets=1 -- log every packet sent and received.
func init() {
	s := os.Getenv("GODEBUG")
	for len(s) > 0 {
		var opt string
		opt, s, _ = strings.Cut(s, ",")
		switch opt {
		case "quiclogpackets=1":
			logPackets = true
		}
	}
}

func logInboundLongPacket(c *Conn, p longPacket) {
	if !logPackets {
		return
	}
	prefix := c.String()
	fmt.Printf("%v recv %v %v\n", prefix, p.ptype, p.num)
	logFrames(prefix+"   <- ", p.payload)
}

func logInboundShortPacket(c *Conn, p shortPacket) {
	if !logPackets {
		return
	}
	prefix := c.String()
	fmt.Printf("%v recv 1-RTT %v\n", prefix, p.num)
	logFrames(prefix+"   <- ", p.payload)
}

func logSentPacket(c *Conn, ptype packetType, pnum packetNumber, src, dst, payload []byte) {
	if !logPackets || len(payload) == 0 {
		return
	}
	prefix := c.String()
	fmt.Printf("%v send %v %v\n", prefix, ptype, pnum)
	logFrames(prefix+"   -> ", payload)
}

func logFrames(prefix string, payload []byte) {
	for len(payload) > 0 {
		f, n := parseDebugFrame(payload)
		if n < 0 {
			fmt.Printf("%vBAD DATA\n", prefix)
			break
		}
		payload = payload[n:]
		fmt.Printf("%v%v\n", prefix, f)
	}
}
