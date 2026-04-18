// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package nettest_test

import (
	"bytes"
	"errors"
	"internal/nettest"
	"io"
	"net"
	"os"
	"testing"
	"testing/synctest"
	"time"
)

func TestConnReadWrite(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		cliConn, srvConn := nettest.NewConnPair()

		cliData := []byte("hello")
		srvData := []byte("HELLO")
		if n, err := cliConn.Write(cliData); n != len(cliData) || err != nil {
			t.Fatalf("cliConn.Write(%q) = %v, %v; want %v, nil", cliData, n, err, len(cliData))
		}
		if err := cliConn.CloseWrite(); err != nil {
			t.Fatalf("cliConn.CloseWrite() = %v, want nil", err)
		}
		if n, err := srvConn.Write(srvData); n != len(srvData) || err != nil {
			t.Fatalf("srvConn.Write(%q) = %v, %v; want %v, nil", srvData, n, err, len(srvData))
		}
		if err := srvConn.CloseWrite(); err != nil {
			t.Fatalf("cliConn.CloseWrite() = %v, want nil", err)
		}
		gotCli, err := io.ReadAll(cliConn)
		if !bytes.Equal(gotCli, srvData) || err != nil {
			t.Fatalf("io.ReadAll(cliConn) = %q, %v; want %v, nil", gotCli, err, srvData)
		}
		gotSrv, err := io.ReadAll(srvConn)
		if !bytes.Equal(gotSrv, cliData) || err != nil {
			t.Fatalf("io.ReadAll(srvConn) = %q, %v; want %v, nil", gotSrv, err, cliData)
		}
	})
}

func TestConnZeroBuffer(t *testing.T) {
	// Exercise the case where one side of the conn is blocked writing and the
	// other side is blocked reading.
	// This can only happen when the read buffer has been set to 0, blocking all writes.
	synctest.Test(t, func(t *testing.T) {
		rconn, wconn := nettest.NewConnPair()
		rconn.SetReadBufferSize(0)
		var readDone, writeDone bool
		go func() {
			rconn.Read(make([]byte, 100))
			readDone = true
		}()
		go func() {
			wconn.Write([]byte("a"))
			writeDone = true
		}()
		synctest.Wait()
		if readDone || writeDone {
			t.Errorf("before unblocking: readDone=%v, writeDone=%v; want false", readDone, writeDone)
		}
		wconn.Close()
		synctest.Wait()
		if !readDone || !writeDone {
			t.Errorf("after unblocking: readDone=%v, writeDone=%v; want true", readDone, writeDone)
		}
	})
}

func TestConnPartialWrite(t *testing.T) {
	// A blocking write to a conn successfully writes some, but not all data.
	synctest.Test(t, func(t *testing.T) {
		const readSize = 5
		data := []byte("0123456789")
		rconn, wconn := nettest.NewConnPair()
		rconn.SetReadBufferSize(1)
		go func() {
			got := make([]byte, readSize)
			if n, err := io.ReadFull(rconn, got); n != readSize || err != nil {
				t.Errorf("io.ReadFull() = %v, %v; want %v, nil", n, err, readSize)
			}
			if want := data[:readSize]; !bytes.Equal(got, want) {
				t.Errorf("read %q, want %q", got, want)
			}
			rconn.Close()
		}()
		n, err := wconn.Write(data)
		if n != readSize+1 || err == nil {
			t.Errorf("Write() = %v, %v; want %v, error", n, err, readSize+1)
		}
	})
}

func TestConnReadDeadline(t *testing.T) {
	for _, unblock := range []struct {
		name string
		f    func(*nettest.Conn)
	}{{
		name: "Write",
		f: func(c *nettest.Conn) {
			c.Write([]byte("x"))
		},
	}, {
		name: "Close",
		f: func(c *nettest.Conn) {
			c.Close()
		},
	}, {
		name: "CloseWrite",
		f: func(c *nettest.Conn) {
			c.CloseWrite()
		},
	}} {
		for _, setDeadline := range []struct {
			name string
			f    func(*nettest.Conn, time.Time) error
		}{{
			name: "SetDeadline",
			f:    (*nettest.Conn).SetDeadline,
		}, {
			name: "SetReadDeadline",
			f:    (*nettest.Conn).SetReadDeadline,
		}} {
			t.Run(unblock.name+"/"+setDeadline.name, func(t *testing.T) {
				testDeadline(t, func() deadlineTest {
					rconn, wconn := nettest.NewConnPair()
					return deadlineTest{
						what: "Read()",
						block: func() error {
							_, err := rconn.Read(make([]byte, 1))
							return err
						},
						unblock: func() {
							unblock.f(wconn)
						},
						setDeadline: func(d time.Duration) {
							setDeadline.f(rconn, time.Now().Add(d))
						},
					}
				})
			})
		}
	}
}

func TestConnWriteDeadline(t *testing.T) {
	for _, unblock := range []struct {
		name string
		f    func(*nettest.Conn)
	}{{
		name: "Read",
		f: func(c *nettest.Conn) {
			io.Copy(io.Discard, c)
		},
	}, {
		name: "Close",
		f: func(c *nettest.Conn) {
			c.Close()
		},
	}, {
		name: "CloseRead",
		f: func(c *nettest.Conn) {
			c.CloseRead()
		},
	}} {
		for _, setDeadline := range []struct {
			name string
			f    func(*nettest.Conn, time.Time) error
		}{{
			name: "SetDeadline",
			f:    (*nettest.Conn).SetDeadline,
		}, {
			name: "SetWriteDeadline",
			f:    (*nettest.Conn).SetWriteDeadline,
		}} {
			t.Run(unblock.name+"/"+setDeadline.name, func(t *testing.T) {
				testDeadline(t, func() deadlineTest {
					rconn, wconn := nettest.NewConnPair()
					rconn.SetReadBufferSize(1)
					return deadlineTest{
						what: "Write()",
						block: func() error {
							_, err := wconn.Write([]byte("1234"))
							wconn.Close()
							return err
						},
						unblock: func() {
							go unblock.f(rconn)
						},
						setDeadline: func(d time.Duration) {
							setDeadline.f(wconn, time.Now().Add(d))
						},
					}
				})
			})
		}
	}
}

func TestConnCanRead(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		rconn, wconn := nettest.NewConnPair()
		if got, want := rconn.CanRead(), false; got != want {
			t.Fatalf("before writing data: rconn.CanRead() = %v, want %v", got, want)
		}
		wconn.Write([]byte("a"))
		if got, want := rconn.CanRead(), true; got != want {
			t.Fatalf("after writing data: rconn.CanRead() = %v, want %v", got, want)
		}
		rconn.Read(make([]byte, 1))
		if got, want := rconn.CanRead(), false; got != want {
			t.Fatalf("after reading data: rconn.CanRead() = %v, want %v", got, want)
		}
		wconn.Close()
		if got, want := rconn.CanRead(), true; got != want {
			t.Fatalf("after closing: rconn.CanRead() = %v, want %v", got, want)
		}
	})
}

func TestConnIsClosed(t *testing.T) {
	for _, test := range []struct {
		name string
		f    func() *nettest.Conn
		want bool
	}{{
		name: "unclosed",
		f: func() *nettest.Conn {
			conn, _ := nettest.NewConnPair()
			return conn
		},
		want: false,
	}, {
		name: "closed",
		f: func() *nettest.Conn {
			conn, _ := nettest.NewConnPair()
			conn.Close()
			return conn
		},
		want: true,
	}, {
		name: "read-closed",
		f: func() *nettest.Conn {
			conn, _ := nettest.NewConnPair()
			conn.CloseRead()
			return conn
		},
		want: false,
	}, {
		name: "write-closed",
		f: func() *nettest.Conn {
			conn, _ := nettest.NewConnPair()
			conn.CloseWrite()
			return conn
		},
		want: false,
	}, {
		name: "read-write-closed",
		f: func() *nettest.Conn {
			conn, _ := nettest.NewConnPair()
			conn.CloseRead()
			conn.CloseWrite()
			return conn
		},
		want: true,
	}} {
		synctestSubtest(t, test.name, func(t *testing.T) {
			conn := test.f()
			if got, want := conn.IsClosed(), test.want; got != want {
				t.Fatalf("conn.IsClosed() = %v, want %v", got, want)
			}
			if got, want := conn.Peer().IsClosed(), false; got != want {
				t.Fatalf("conn.Peer().IsClosed() = %v, want %v", got, want)
			}
		})
	}
}

var anyError = errors.New("any") // anyError is passed to isOpError to match any error

func isOpError(err, want error) bool {
	oe, ok := err.(*net.OpError)
	return ok && (oe.Err == want || want == anyError)
}

func wantConnReadBytes(t *testing.T, c *nettest.Conn, want []byte) {
	t.Helper()
	got := make([]byte, len(want))
	n, err := io.ReadFull(c, got)
	if n < len(want) || err != nil {
		t.Fatalf("io.ReadFull = %v, %v; want %v, nil", n, err, len(want))
	}

	if !bytes.Equal(got, want) {
		t.Fatalf("io.ReadFull read %q, want %q", got, want)
	}
}

func wantConnReadErr(t *testing.T, c *nettest.Conn, want error) {
	t.Helper()
	n, err := c.Read(make([]byte, 1))
	if want == io.EOF {
		if n != 0 || err != io.EOF {
			t.Fatalf("c.Read() = %v, %v; want 0, io.EOF", n, err)
		}
	} else {
		if n != 0 || !isOpError(err, want) {
			t.Fatalf("c.Read() = %v, %v; want 0, OpError{Err: %q}", n, err, want)
		}
	}
}

func wantConnReadBlocked(t *testing.T, c *nettest.Conn) {
	done := false
	go func() {
		n, err := c.Read(make([]byte, 1))
		if n != 0 || !errors.Is(err, os.ErrDeadlineExceeded) {
			t.Errorf("c.Read() = %v, %v; want 0, ErrDeadlineExceeded", n, err)
		}
		done = true
	}()
	synctest.Wait()
	if done {
		t.Fatalf("Read unexpectedly returned before setting deadline")
	}
	c.SetReadDeadline(time.Now().Add(-1 * time.Second))
	synctest.Wait()
	c.SetReadDeadline(time.Time{})
	if !done {
		t.Fatalf("Read unexpectedly did not return after setting deadline")
	}
}

func TestConnSetReadError(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		wantErr := errors.New("error")
		rconn, wconn := nettest.NewConnPair()
		rconn.SetReadError(wantErr)

		// Consume buffer before returning error.
		wconn.Write([]byte("one"))
		wantConnReadBytes(t, rconn, []byte("one"))
		wantConnReadErr(t, rconn, wantErr)

		// Write more to the buffer, suppressing error until buffer drains again.
		wconn.Write([]byte("two"))
		wantConnReadBytes(t, rconn, []byte("two"))
		wantConnReadErr(t, rconn, wantErr)

		// Error may be cleared.
		rconn.SetReadError(nil)
		wantConnReadBlocked(t, rconn)

		// Close overrides read error.
		rconn.SetReadError(wantErr)
		wconn.Write([]byte("three"))
		wconn.Close()
		wantConnReadBytes(t, rconn, []byte("three"))
		wantConnReadErr(t, rconn, io.EOF)

		// Setting another read error does not override Close.
		rconn.SetReadError(nil)
		wantConnReadErr(t, rconn, io.EOF)
		rconn.SetReadError(wantErr)
		wantConnReadErr(t, rconn, io.EOF)

		// ErrClosed takes precedence over read error.
		rconn.Close()
		wantConnReadErr(t, rconn, net.ErrClosed)
	})
}

func wantConnWriteBytes(t *testing.T, c *nettest.Conn, b []byte) {
	t.Helper()
	if n, err := c.Write(b); n != len(b) || err != nil {
		t.Fatalf("c.Write() = %v, %v; want %v, nil", n, err, len(b))
	}
}

func wantConnWriteErr(t *testing.T, c *nettest.Conn, want error) {
	t.Helper()
	n, err := c.Write(make([]byte, 1))
	if n != 0 || !isOpError(err, want) {
		t.Fatalf("c.Write() = %v, %v; want 0, OpError{Err: %q}", n, err, want)
	}
}

func TestConnSetWriteError(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		wantErr := errors.New("error")
		rconn, wconn := nettest.NewConnPair()
		wconn.SetWriteError(wantErr)

		// Error blocks writes.
		wantConnWriteErr(t, wconn, wantErr)
		wantConnReadBlocked(t, rconn)

		// Error may be cleared.
		wconn.SetWriteError(nil)
		wantConnWriteBytes(t, wconn, []byte("one"))

		// Restoring error does not prevent reading buffered data.
		wconn.SetWriteError(wantErr)
		wantConnWriteErr(t, wconn, wantErr)
		wantConnReadBytes(t, rconn, []byte("one"))

		// Error does not interfere with closing the conn.
		wconn.Close()
		wantConnReadErr(t, rconn, io.EOF)
	})
}

func TestConnSetCloseError(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		wantErr := errors.New("error")
		rconn, wconn := nettest.NewConnPair()

		wconn.SetCloseError(wantErr)
		if _, err := wconn.Write([]byte("one")); err != nil {
			t.Fatalf("wconn.Write = %v, want success", err)
		}
		if err := wconn.Close(); !isOpError(err, wantErr) {
			t.Fatalf("wconn.Close = %v, want OpError{Err: %v}", err, wantErr)
		}
		if err := wconn.Close(); !isOpError(err, net.ErrClosed) {
			t.Fatalf("wconn.Close = %v, want OpError{Err: net.ErrClosed}", err)
		}
		wantConnReadBytes(t, rconn, []byte("one"))
		wantConnReadErr(t, rconn, io.EOF)
	})
}

func TestConnCloseReadWriteError(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		conn, _ := nettest.NewConnPair()
		conn.SetCloseError(errors.New("error"))
		if err := conn.CloseRead(); err != nil {
			t.Fatalf("conn.CloseRead = %v, want nil", err)
		}
		if err := conn.CloseWrite(); err != nil {
			t.Fatalf("conn.CloseRead = %v, want nil", err)
		}
	})
}
