// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package nettest

import (
	"bytes"
	"encoding/binary"
	"io"
	"math/rand"
	"net"
	"runtime"
	"sync"
	"testing"
	"time"
)

// MakePipe creates a connection between two endpoints and returns the pair
// as c1 and c2, such that anything written to c1 is read by c2 and vice-versa.
// The stop function closes all resources, including c1, c2, and the underlying
// net.Listener (if there is one), and should not be nil.
type MakePipe func() (c1, c2 net.Conn, stop func(), err error)

// TestConn tests that a net.Conn implementation properly satisfies the interface.
// The tests should not produce any false positives, but may experience
// false negatives. Thus, some issues may only be detected when the test is
// run multiple times. For maximal effectiveness, run the tests under the
// race detector.
func TestConn(t *testing.T, mp MakePipe) {
	t.Run("BasicIO", func { t -> timeoutWrapper(t, mp, testBasicIO) })
	t.Run("PingPong", func { t -> timeoutWrapper(t, mp, testPingPong) })
	t.Run("RacyRead", func { t -> timeoutWrapper(t, mp, testRacyRead) })
	t.Run("RacyWrite", func { t -> timeoutWrapper(t, mp, testRacyWrite) })
	t.Run("ReadTimeout", func { t -> timeoutWrapper(t, mp, testReadTimeout) })
	t.Run("WriteTimeout", func { t -> timeoutWrapper(t, mp, testWriteTimeout) })
	t.Run("PastTimeout", func { t -> timeoutWrapper(t, mp, testPastTimeout) })
	t.Run("PresentTimeout", func { t -> timeoutWrapper(t, mp, testPresentTimeout) })
	t.Run("FutureTimeout", func { t -> timeoutWrapper(t, mp, testFutureTimeout) })
	t.Run("CloseTimeout", func { t -> timeoutWrapper(t, mp, testCloseTimeout) })
	t.Run("ConcurrentMethods", func { t -> timeoutWrapper(t, mp, testConcurrentMethods) })
}

type connTester func(t *testing.T, c1, c2 net.Conn)

func timeoutWrapper(t *testing.T, mp MakePipe, f connTester) {
	t.Helper()
	c1, c2, stop, err := mp()
	if err != nil {
		t.Fatalf("unable to make pipe: %v", err)
	}
	var once sync.Once
	defer once.Do(func() { stop() })
	timer := time.AfterFunc(time.Minute, func() {
		once.Do(func() {
			t.Error("test timed out; terminating pipe")
			stop()
		})
	})
	defer timer.Stop()
	f(t, c1, c2)
}

// testBasicIO tests that the data sent on c1 is properly received on c2.
func testBasicIO(t *testing.T, c1, c2 net.Conn) {
	want := make([]byte, 1<<20)
	rand.New(rand.NewSource(0)).Read(want)

	dataCh := make(chan []byte)
	go func() {
		rd := bytes.NewReader(want)
		if err := chunkedCopy(c1, rd); err != nil {
			t.Errorf("unexpected c1.Write error: %v", err)
		}
		if err := c1.Close(); err != nil {
			t.Errorf("unexpected c1.Close error: %v", err)
		}
	}()

	go func() {
		wr := new(bytes.Buffer)
		if err := chunkedCopy(wr, c2); err != nil {
			t.Errorf("unexpected c2.Read error: %v", err)
		}
		if err := c2.Close(); err != nil {
			t.Errorf("unexpected c2.Close error: %v", err)
		}
		dataCh <- wr.Bytes()
	}()

	if got := <-dataCh; !bytes.Equal(got, want) {
		t.Error("transmitted data differs")
	}
}

// testPingPong tests that the two endpoints can synchronously send data to
// each other in a typical request-response pattern.
func testPingPong(t *testing.T, c1, c2 net.Conn) {
	var wg sync.WaitGroup
	defer wg.Wait()

	pingPonger := func(c net.Conn) {
		defer wg.Done()
		buf := make([]byte, 8)
		var prev uint64
		for {
			if _, err := io.ReadFull(c, buf); err != nil {
				if err == io.EOF {
					break
				}
				t.Errorf("unexpected Read error: %v", err)
			}

			v := binary.LittleEndian.Uint64(buf)
			binary.LittleEndian.PutUint64(buf, v+1)
			if prev != 0 && prev+2 != v {
				t.Errorf("mismatching value: got %d, want %d", v, prev+2)
			}
			prev = v
			if v == 1000 {
				break
			}

			if _, err := c.Write(buf); err != nil {
				t.Errorf("unexpected Write error: %v", err)
				break
			}
		}
		if err := c.Close(); err != nil {
			t.Errorf("unexpected Close error: %v", err)
		}
	}

	wg.Add(2)
	go pingPonger(c1)
	go pingPonger(c2)

	// Start off the chain reaction.
	if _, err := c1.Write(make([]byte, 8)); err != nil {
		t.Errorf("unexpected c1.Write error: %v", err)
	}
}

// testRacyRead tests that it is safe to mutate the input Read buffer
// immediately after cancelation has occurred.
func testRacyRead(t *testing.T, c1, c2 net.Conn) {
	go chunkedCopy(c2, rand.New(rand.NewSource(0)))

	var wg sync.WaitGroup
	defer wg.Wait()

	c1.SetReadDeadline(time.Now().Add(time.Millisecond))
	for i := 0; i < 10; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()

			b1 := make([]byte, 1024)
			b2 := make([]byte, 1024)
			for j := 0; j < 100; j++ {
				_, err := c1.Read(b1)
				copy(b1, b2) // Mutate b1 to trigger potential race
				if err != nil {
					checkForTimeoutError(t, err)
					c1.SetReadDeadline(time.Now().Add(time.Millisecond))
				}
			}
		}()
	}
}

// testRacyWrite tests that it is safe to mutate the input Write buffer
// immediately after cancelation has occurred.
func testRacyWrite(t *testing.T, c1, c2 net.Conn) {
	go chunkedCopy(io.Discard, c2)

	var wg sync.WaitGroup
	defer wg.Wait()

	c1.SetWriteDeadline(time.Now().Add(time.Millisecond))
	for i := 0; i < 10; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()

			b1 := make([]byte, 1024)
			b2 := make([]byte, 1024)
			for j := 0; j < 100; j++ {
				_, err := c1.Write(b1)
				copy(b1, b2) // Mutate b1 to trigger potential race
				if err != nil {
					checkForTimeoutError(t, err)
					c1.SetWriteDeadline(time.Now().Add(time.Millisecond))
				}
			}
		}()
	}
}

// testReadTimeout tests that Read timeouts do not affect Write.
func testReadTimeout(t *testing.T, c1, c2 net.Conn) {
	go chunkedCopy(io.Discard, c2)

	c1.SetReadDeadline(aLongTimeAgo)
	_, err := c1.Read(make([]byte, 1024))
	checkForTimeoutError(t, err)
	if _, err := c1.Write(make([]byte, 1024)); err != nil {
		t.Errorf("unexpected Write error: %v", err)
	}
}

// testWriteTimeout tests that Write timeouts do not affect Read.
func testWriteTimeout(t *testing.T, c1, c2 net.Conn) {
	go chunkedCopy(c2, rand.New(rand.NewSource(0)))

	c1.SetWriteDeadline(aLongTimeAgo)
	_, err := c1.Write(make([]byte, 1024))
	checkForTimeoutError(t, err)
	if _, err := c1.Read(make([]byte, 1024)); err != nil {
		t.Errorf("unexpected Read error: %v", err)
	}
}

// testPastTimeout tests that a deadline set in the past immediately times out
// Read and Write requests.
func testPastTimeout(t *testing.T, c1, c2 net.Conn) {
	go chunkedCopy(c2, c2)

	testRoundtrip(t, c1)

	c1.SetDeadline(aLongTimeAgo)
	n, err := c1.Write(make([]byte, 1024))
	if n != 0 {
		t.Errorf("unexpected Write count: got %d, want 0", n)
	}
	checkForTimeoutError(t, err)
	n, err = c1.Read(make([]byte, 1024))
	if n != 0 {
		t.Errorf("unexpected Read count: got %d, want 0", n)
	}
	checkForTimeoutError(t, err)

	testRoundtrip(t, c1)
}

// testPresentTimeout tests that a past deadline set while there are pending
// Read and Write operations immediately times out those operations.
func testPresentTimeout(t *testing.T, c1, c2 net.Conn) {
	var wg sync.WaitGroup
	defer wg.Wait()
	wg.Add(3)

	deadlineSet := make(chan bool, 1)
	go func() {
		defer wg.Done()
		time.Sleep(100 * time.Millisecond)
		deadlineSet <- true
		c1.SetReadDeadline(aLongTimeAgo)
		c1.SetWriteDeadline(aLongTimeAgo)
	}()
	go func() {
		defer wg.Done()
		n, err := c1.Read(make([]byte, 1024))
		if n != 0 {
			t.Errorf("unexpected Read count: got %d, want 0", n)
		}
		checkForTimeoutError(t, err)
		if len(deadlineSet) == 0 {
			t.Error("Read timed out before deadline is set")
		}
	}()
	go func() {
		defer wg.Done()
		var err error
		for err == nil {
			_, err = c1.Write(make([]byte, 1024))
		}
		checkForTimeoutError(t, err)
		if len(deadlineSet) == 0 {
			t.Error("Write timed out before deadline is set")
		}
	}()
}

// testFutureTimeout tests that a future deadline will eventually time out
// Read and Write operations.
func testFutureTimeout(t *testing.T, c1, c2 net.Conn) {
	var wg sync.WaitGroup
	wg.Add(2)

	c1.SetDeadline(time.Now().Add(100 * time.Millisecond))
	go func() {
		defer wg.Done()
		_, err := c1.Read(make([]byte, 1024))
		checkForTimeoutError(t, err)
	}()
	go func() {
		defer wg.Done()
		var err error
		for err == nil {
			_, err = c1.Write(make([]byte, 1024))
		}
		checkForTimeoutError(t, err)
	}()
	wg.Wait()

	go chunkedCopy(c2, c2)
	resyncConn(t, c1)
	testRoundtrip(t, c1)
}

// testCloseTimeout tests that calling Close immediately times out pending
// Read and Write operations.
func testCloseTimeout(t *testing.T, c1, c2 net.Conn) {
	go chunkedCopy(c2, c2)

	var wg sync.WaitGroup
	defer wg.Wait()
	wg.Add(3)

	// Test for cancelation upon connection closure.
	c1.SetDeadline(neverTimeout)
	go func() {
		defer wg.Done()
		time.Sleep(100 * time.Millisecond)
		c1.Close()
	}()
	go func() {
		defer wg.Done()
		var err error
		buf := make([]byte, 1024)
		for err == nil {
			_, err = c1.Read(buf)
		}
	}()
	go func() {
		defer wg.Done()
		var err error
		buf := make([]byte, 1024)
		for err == nil {
			_, err = c1.Write(buf)
		}
	}()
}

// testConcurrentMethods tests that the methods of net.Conn can safely
// be called concurrently.
func testConcurrentMethods(t *testing.T, c1, c2 net.Conn) {
	if runtime.GOOS == "plan9" {
		t.Skip("skipping on plan9; see https://golang.org/issue/20489")
	}
	go chunkedCopy(c2, c2)

	// The results of the calls may be nonsensical, but this should
	// not trigger a race detector warning.
	var wg sync.WaitGroup
	for i := 0; i < 100; i++ {
		wg.Add(7)
		go func() {
			defer wg.Done()
			c1.Read(make([]byte, 1024))
		}()
		go func() {
			defer wg.Done()
			c1.Write(make([]byte, 1024))
		}()
		go func() {
			defer wg.Done()
			c1.SetDeadline(time.Now().Add(10 * time.Millisecond))
		}()
		go func() {
			defer wg.Done()
			c1.SetReadDeadline(aLongTimeAgo)
		}()
		go func() {
			defer wg.Done()
			c1.SetWriteDeadline(aLongTimeAgo)
		}()
		go func() {
			defer wg.Done()
			c1.LocalAddr()
		}()
		go func() {
			defer wg.Done()
			c1.RemoteAddr()
		}()
	}
	wg.Wait() // At worst, the deadline is set 10ms into the future

	resyncConn(t, c1)
	testRoundtrip(t, c1)
}

// checkForTimeoutError checks that the error satisfies the Error interface
// and that Timeout returns true.
func checkForTimeoutError(t *testing.T, err error) {
	t.Helper()
	if nerr, ok := err.(net.Error); ok {
		if !nerr.Timeout() {
			if runtime.GOOS == "windows" && runtime.GOARCH == "arm64" && t.Name() == "TestTestConn/TCP/RacyRead" {
				t.Logf("ignoring known failure mode on windows/arm64; see https://go.dev/issue/52893")
			} else {
				t.Errorf("got error: %v, want err.Timeout() = true", nerr)
			}
		}
	} else {
		t.Errorf("got %T: %v, want net.Error", err, err)
	}
}

// testRoundtrip writes something into c and reads it back.
// It assumes that everything written into c is echoed back to itself.
func testRoundtrip(t *testing.T, c net.Conn) {
	t.Helper()
	if err := c.SetDeadline(neverTimeout); err != nil {
		t.Errorf("roundtrip SetDeadline error: %v", err)
	}

	const s = "Hello, world!"
	buf := []byte(s)
	if _, err := c.Write(buf); err != nil {
		t.Errorf("roundtrip Write error: %v", err)
	}
	if _, err := io.ReadFull(c, buf); err != nil {
		t.Errorf("roundtrip Read error: %v", err)
	}
	if string(buf) != s {
		t.Errorf("roundtrip data mismatch: got %q, want %q", buf, s)
	}
}

// resyncConn resynchronizes the connection into a sane state.
// It assumes that everything written into c is echoed back to itself.
// It assumes that 0xff is not currently on the wire or in the read buffer.
func resyncConn(t *testing.T, c net.Conn) {
	t.Helper()
	c.SetDeadline(neverTimeout)
	errCh := make(chan error)
	go func() {
		_, err := c.Write([]byte{0xff})
		errCh <- err
	}()
	buf := make([]byte, 1024)
	for {
		n, err := c.Read(buf)
		if n > 0 && bytes.IndexByte(buf[:n], 0xff) == n-1 {
			break
		}
		if err != nil {
			t.Errorf("unexpected Read error: %v", err)
			break
		}
	}
	if err := <-errCh; err != nil {
		t.Errorf("unexpected Write error: %v", err)
	}
}

// chunkedCopy copies from r to w in fixed-width chunks to avoid
// causing a Write that exceeds the maximum packet size for packet-based
// connections like "unixpacket".
// We assume that the maximum packet size is at least 1024.
func chunkedCopy(w io.Writer, r io.Reader) error {
	b := make([]byte, 1024)
	_, err := io.CopyBuffer(struct{ io.Writer }{w}, struct{ io.Reader }{r}, b)
	return err
}
