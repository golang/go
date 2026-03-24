// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Flow control

package http2

// inflowMinRefresh is the minimum number of bytes we'll send for a
// flow control window update.
const inflowMinRefresh = 4 << 10

// inflow accounts for an inbound flow control window.
// It tracks both the latest window sent to the peer (used for enforcement)
// and the accumulated unsent window.
type inflow struct {
	avail  int32
	unsent int32
}

// init sets the initial window.
func (f *inflow) init(n int32) {
	f.avail = n
}

// add adds n bytes to the window, with a maximum window size of max,
// indicating that the peer can now send us more data.
// For example, the user read from a {Request,Response} body and consumed
// some of the buffered data, so the peer can now send more.
// It returns the number of bytes to send in a WINDOW_UPDATE frame to the peer.
// Window updates are accumulated and sent when the unsent capacity
// is at least inflowMinRefresh or will at least double the peer's available window.
func (f *inflow) add(n int) (connAdd int32) {
	if n < 0 {
		panic("negative update")
	}
	unsent := int64(f.unsent) + int64(n)
	// "A sender MUST NOT allow a flow-control window to exceed 2^31-1 octets."
	// RFC 7540 Section 6.9.1.
	const maxWindow = 1<<31 - 1
	if unsent+int64(f.avail) > maxWindow {
		panic("flow control update exceeds maximum window size")
	}
	f.unsent = int32(unsent)
	if f.unsent < inflowMinRefresh && f.unsent < f.avail {
		// If there aren't at least inflowMinRefresh bytes of window to send,
		// and this update won't at least double the window, buffer the update for later.
		return 0
	}
	f.avail += f.unsent
	f.unsent = 0
	return int32(unsent)
}

// take attempts to take n bytes from the peer's flow control window.
// It reports whether the window has available capacity.
func (f *inflow) take(n uint32) bool {
	if n > uint32(f.avail) {
		return false
	}
	f.avail -= int32(n)
	return true
}

// takeInflows attempts to take n bytes from two inflows,
// typically connection-level and stream-level flows.
// It reports whether both windows have available capacity.
func takeInflows(f1, f2 *inflow, n uint32) bool {
	if n > uint32(f1.avail) || n > uint32(f2.avail) {
		return false
	}
	f1.avail -= int32(n)
	f2.avail -= int32(n)
	return true
}

// outflow is the outbound flow control window's size.
type outflow struct {
	_ incomparable

	// n is the number of DATA bytes we're allowed to send.
	// An outflow is kept both on a conn and a per-stream.
	n int32

	// conn points to the shared connection-level outflow that is
	// shared by all streams on that conn. It is nil for the outflow
	// that's on the conn directly.
	conn *outflow
}

func (f *outflow) setConnFlow(cf *outflow) { f.conn = cf }

func (f *outflow) available() int32 {
	n := f.n
	if f.conn != nil && f.conn.n < n {
		n = f.conn.n
	}
	return n
}

func (f *outflow) take(n int32) {
	if n > f.available() {
		panic("internal error: took too much")
	}
	f.n -= n
	if f.conn != nil {
		f.conn.n -= n
	}
}

// add adds n bytes (positive or negative) to the flow control window.
// It returns false if the sum would exceed 2^31-1.
func (f *outflow) add(n int32) bool {
	sum := f.n + n
	if (sum > n) == (f.n > 0) {
		f.n = sum
		return true
	}
	return false
}
