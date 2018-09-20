// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Bluetooth sockets and messages

package unix

// Bluetooth Protocols
const (
	BTPROTO_L2CAP  = 0
	BTPROTO_HCI    = 1
	BTPROTO_SCO    = 2
	BTPROTO_RFCOMM = 3
	BTPROTO_BNEP   = 4
	BTPROTO_CMTP   = 5
	BTPROTO_HIDP   = 6
	BTPROTO_AVDTP  = 7
)

const (
	HCI_CHANNEL_RAW     = 0
	HCI_CHANNEL_USER    = 1
	HCI_CHANNEL_MONITOR = 2
	HCI_CHANNEL_CONTROL = 3
)

// Socketoption Level
const (
	SOL_BLUETOOTH = 0x112
	SOL_HCI       = 0x0
	SOL_L2CAP     = 0x6
	SOL_RFCOMM    = 0x12
	SOL_SCO       = 0x11
)
