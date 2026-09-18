// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package http3

import (
	"errors"
	"testing"
	"testing/synctest"

	"golang.org/x/net/quic"
)

// Tests which apply to both client and server connections.

func TestConnCreatesControlStream(t *testing.T) {
	runConnTest(t, func(t testing.TB, tc *testQUICConn) {
		controlStream := tc.wantStream(streamTypeControl)
		controlStream.wantFrameHeader(
			"server sends SETTINGS frame on control stream",
			frameTypeSettings)
		controlStream.discardFrame()
	})
}

func TestConnUnknownUnidirectionalStream(t *testing.T) {
	// "Recipients of unknown stream types MUST either abort reading of the stream
	// or discard incoming data without further processing."
	// https://www.rfc-editor.org/rfc/rfc9114.html#section-6.2-7
	runConnTest(t, func(t testing.TB, tc *testQUICConn) {
		st := tc.newStream(0x21) // reserved stream type

		// The endpoint should send a STOP_SENDING for this stream,
		// but it should not close the connection.
		synctest.Wait()
		wantErr := quic.StreamError(errH3StreamCreationError)
		if _, err := st.Write([]byte("hello")); !errors.Is(err, wantErr) {
			t.Fatalf("write to send-only stream with an unknown type returned error %v; want %v", err, wantErr)
		}
		tc.wantNotClosed("after receiving unknown unidirectional stream type")
	})
}

func TestConnUnknownSettings(t *testing.T) {
	// "An implementation MUST ignore any [settings] parameter with
	// an identifier it does not understand."
	// https://www.rfc-editor.org/rfc/rfc9114.html#section-7.2.4-9
	runConnTest(t, func(t testing.TB, tc *testQUICConn) {
		controlStream := tc.newStream(streamTypeControl)
		controlStream.writeSettings(0x1f+0x21, 0) // reserved settings type
		controlStream.Flush()
		tc.wantNotClosed("after receiving unknown settings")
	})
}

func TestConnInvalidSettings(t *testing.T) {
	// "These reserved settings MUST NOT be sent, and their receipt MUST
	// be treated as a connection error of type H3_SETTINGS_ERROR."
	// https://www.rfc-editor.org/rfc/rfc9114.html#section-7.2.4.1-5
	runConnTest(t, func(t testing.TB, tc *testQUICConn) {
		controlStream := tc.newStream(streamTypeControl)
		controlStream.writeSettings(0x02, 0) // HTTP/2 SETTINGS_ENABLE_PUSH
		controlStream.Flush()
		tc.wantClosed("invalid setting", errH3SettingsError)
	})
}

func TestConnDuplicateStream(t *testing.T) {
	for _, stype := range []streamType{
		streamTypeControl,
		streamTypeEncoder,
		streamTypeDecoder,
	} {
		t.Run(stype.String(), func(t *testing.T) {
			runConnTest(t, func(t testing.TB, tc *testQUICConn) {
				_ = tc.newStream(stype)
				tc.wantNotClosed("after creating one " + stype.String() + " stream")

				// Opening a second control, encoder, or decoder stream
				// is a protocol violation.
				_ = tc.newStream(stype)
				tc.wantClosed("duplicate stream", errH3StreamCreationError)
			})
		})
	}
}

func TestConnUnknownFrames(t *testing.T) {
	for _, stype := range []streamType{
		streamTypeControl,
	} {
		t.Run(stype.String(), func(t *testing.T) {
			runConnTest(t, func(t testing.TB, tc *testQUICConn) {
				st := tc.newStream(stype)

				if stype == streamTypeControl {
					// First frame on the control stream must be settings.
					st.writeVarint(int64(frameTypeSettings))
					st.writeVarint(0) // size
				}

				data := "frame content"
				st.writeVarint(0x1f + 0x21)      // reserved frame type
				st.writeVarint(int64(len(data))) // size
				st.Write([]byte(data))
				st.Flush()

				tc.wantNotClosed("after writing unknown frame")
			})
		})
	}
}

func TestConnInvalidFrames(t *testing.T) {
	runConnTest(t, func(t testing.TB, tc *testQUICConn) {
		control := tc.newStream(streamTypeControl)

		// SETTINGS frame.
		control.writeVarint(int64(frameTypeSettings))
		control.writeVarint(0) // size

		// DATA frame (invalid on the control stream).
		control.writeVarint(int64(frameTypeData))
		control.writeVarint(0) // size
		control.Flush()
		tc.wantClosed("after writing DATA frame to control stream", errH3FrameUnexpected)
	})
}

func TestConnPeerCreatesBadUnidirectionalStream(t *testing.T) {
	runConnTest(t, func(t testing.TB, tc *testQUICConn) {
		// Create and close a stream without sending the unidirectional stream header.
		qs, err := tc.qconn.NewSendOnlyStream(canceledCtx)
		if err != nil {
			t.Fatal(err)
		}
		st := newTestQUICStream(tc.t, newStream(qs))
		st.Close()

		tc.wantClosed("after peer creates and closes uni stream", errH3StreamCreationError)
	})
}

func runConnTest(t *testing.T, f func(testing.TB, *testQUICConn)) {
	t.Helper()
	synctestSubtest(t, "client", func(t *testing.T) {
		tc := newTestClientConn(t)
		f(t, tc.testQUICConn)
	})
	synctestSubtest(t, "server", func(t *testing.T) {
		ts := newTestServer(t, nil)
		tc := ts.connect()
		f(t, tc.testQUICConn)
	})
}
