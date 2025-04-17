// compile

// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

type NodeLink struct{}

// A role our end of NodeLink is intended to play
type LinkRole int64

const (
	LinkServer LinkRole = iota // link created as server
	LinkClient                 // link created as client

	// for testing:
	linkNoRecvSend LinkRole = 1 << 16 // do not spawn serveRecv & serveSend
	linkFlagsMask  LinkRole = (1<<32 - 1) << 16
)

func NewNodeLink(role LinkRole) *NodeLink {
	var nextConnId uint32
	switch role &^ linkFlagsMask {
	case LinkServer:
		nextConnId = 0 // all initiated by us connId will be even
	case LinkClient:
		nextConnId = 1 // ----//---- odd
	default:
		panic("invalid conn role")
	}

	_ = nextConnId
	return nil
}
