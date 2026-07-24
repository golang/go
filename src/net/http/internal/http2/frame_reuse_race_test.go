// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package http2

import (
	"bytes"
	"io"
	"os"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"golang.org/x/net/http2/hpack"
)

// The tests in this file exercise the Framer.SetReuseFrames contract
// from a concurrency angle. The reuse contract says the *Frame returned
// by ReadFrame, any slice aliasing the Framer's read buffer reachable
// from that frame, and the MetaHeadersFrame Fields slice are only valid
// until the next ReadFrame call. A retention bug under reuse manifests
// as one goroutine reading a retained slice while another (the reader)
// writes into the same backing memory on the next ReadFrame.
//
// These tests model the server's readFrames goroutine pattern (one
// reader goroutine, one consumer goroutine separated by a gate
// channel) and aim to be effective under "go test -race". Each runs
// twice: a "meta" variant with ReadMetaHeaders set, where HEADERS
// surface as *MetaHeadersFrame, and a "raw" variant without it, where
// HEADERS and CONTINUATION surface individually and their
// HeaderBlockFragment aliases the read buffer.
//
//   - TestFrameReuseRaceCorrect: faithful gated handoff. Must NOT race
//     under -race, regardless of how many frames are read. This is a
//     positive control that asserts the gated pattern is safe with
//     reuse on.
//
//   - TestFrameReuseRaceAdversarial: deliberately leaks a slice past
//     the gate. Must race under -race. Guarded behind the
//     H2_REUSE_RACE_NEGATIVE=1 env var so normal `go test` and CI
//     stay green; the assertion of the negative case is the race
//     detector itself.
//
// A companion end-to-end stress test exists in
// frame_reuse_e2e_test.go (package http2_test) which drives the real
// net/http HTTP/2 server and Transport under -race. The synthetic
// tests in this file cannot reach the production process* code paths;
// the e2e test does.

// preEncodeReuseRaceStream encodes a long sequence of frames
// exercising every frame type extended by SetReuseFrames: DATA,
// WINDOW_UPDATE, HEADERS, and HEADERS+CONTINUATION (so the meta
// header path runs). All HEADERS/DATA payloads are the same size so
// the framer's readBuf does not reallocate between frames; the
// retained slice in the adversarial subtest therefore continues to
// alias the same backing array that the next ReadFrame writes into.
// The header blocks stay at four fields, well under
// maxRetainedMetaFields, so the meta path keeps reusing one Fields
// backing array for the same reason.
func preEncodeReuseRaceStream(tb testing.TB, iters int) []byte {
	tb.Helper()
	var buf bytes.Buffer
	fr := NewFramer(&buf, nil)
	// Encode HPACK fields once; reuse to keep payload sizes stable.
	hdrBlock := encodeHeaderRaw(tb,
		":method", "GET",
		":path", "/",
		":scheme", "http",
		":authority", "example.com",
	)
	// Split hdrBlock so we can emit HEADERS+CONTINUATION for the
	// meta path.
	half := len(hdrBlock) / 2
	if half == 0 {
		half = 1
	}
	// Use a fixed-length data payload so the readBuf size stays
	// constant once it grows on the first read.
	dataPayload := bytes.Repeat([]byte{0xab}, 64)

	for i := 0; i < iters; i++ {
		streamID := uint32(2*i + 1)
		// DATA
		if err := fr.WriteData(streamID, false, dataPayload); err != nil {
			tb.Fatal(err)
		}
		// WINDOW_UPDATE
		if err := fr.WriteWindowUpdate(streamID, uint32(1+i%4096)); err != nil {
			tb.Fatal(err)
		}
		// HEADERS (single-frame, EndHeaders=true)
		if err := fr.WriteHeaders(HeadersFrameParam{
			StreamID:      streamID,
			BlockFragment: hdrBlock,
			EndHeaders:    true,
		}); err != nil {
			tb.Fatal(err)
		}
		// HEADERS + CONTINUATION (drives the meta-headers path when
		// ReadMetaHeaders is set, and the CONTINUATION path when not)
		if err := fr.WriteHeaders(HeadersFrameParam{
			StreamID:      streamID,
			BlockFragment: hdrBlock[:half],
			EndHeaders:    false,
		}); err != nil {
			tb.Fatal(err)
		}
		if err := fr.WriteContinuation(streamID, true, hdrBlock[half:]); err != nil {
			tb.Fatal(err)
		}
	}
	return buf.Bytes()
}

// newReuseRaceFramer builds a SetReuseFrames Framer over the
// pre-encoded stream, with the meta-headers decoder installed when
// meta is set.
func newReuseRaceFramer(encoded []byte, meta bool) *Framer {
	fr := NewFramer(io.Discard, bytes.NewReader(encoded))
	fr.SetReuseFrames()
	if meta {
		fr.ReadMetaHeaders = hpack.NewDecoder(initialHeaderTableSize, nil)
	}
	return fr
}

// runReadFramesGoroutine launches a goroutine that mimics
// serverConn.readFrames: reads a frame, sends it to ch, waits on the
// returned gate before reading the next. Returns the gate channel and
// a done channel that closes once the reader goroutine exits.
type framedRead struct {
	frame Frame
	err   error
	// done must be called once the consumer no longer retains frame
	// or any slice aliasing the framer's read buffer.
	done chan<- struct{}
}

func runReadFramesGoroutine(tb testing.TB, fr *Framer, ch chan<- framedRead) (exited <-chan struct{}) {
	tb.Helper()
	exitc := make(chan struct{})
	go func() {
		defer close(exitc)
		for {
			gate := make(chan struct{})
			f, err := fr.ReadFrame()
			ch <- framedRead{frame: f, err: err, done: gate}
			// Wait for the consumer to signal done before reading
			// again. This re-creates the readFrames gate.
			<-gate
			if err != nil {
				return
			}
		}
	}()
	return exitc
}

// consumeSliceWork is a noinline helper that reads every byte of b
// once. Its only purpose is to ensure the race detector observes a
// read of the memory backing b. If a concurrent goroutine writes the
// same memory without a happens-before edge, -race will fire here.
//
//go:noinline
func consumeSliceWork(b []byte) byte {
	var x byte
	for _, v := range b {
		x ^= v
	}
	return x
}

// consumeHeaderFieldWork reads every HeaderField element of fields,
// plus the name+value string bytes. Ranging over a retained Fields
// slice reads the HeaderField structs out of the backing array the
// Framer owns; with SetReuseFrames, the next meta parse clears and
// re-appends into that same array, so the race detector observes the
// conflict on the struct memory. The string bytes themselves are
// independent allocations (hpack hands out fresh strings per decode),
// so they never alias the framer's read buffer.
//
//go:noinline
func consumeHeaderFieldWork(fields []hpack.HeaderField) byte {
	var x byte
	for _, hf := range fields {
		for i := 0; i < len(hf.Name); i++ {
			x ^= hf.Name[i]
		}
		for i := 0; i < len(hf.Value); i++ {
			x ^= hf.Value[i]
		}
	}
	return x
}

// TestFrameReuseRaceCorrect runs the full reader-consumer dance with
// SetReuseFrames enabled, consuming every frame's payload before
// signaling the gate. Under -race, this MUST NOT fire.
//
// What this catches: a reuse implementation that mutates the cached
// frame or its backing buffer before the consumer signals done (for
// example, by parsing the next frame eagerly on a background
// goroutine or by reusing the buffer for some other purpose).
func TestFrameReuseRaceCorrect(t *testing.T) {
	t.Run("meta", func(t *testing.T) { testFrameReuseRaceCorrect(t, true) })
	t.Run("raw", func(t *testing.T) { testFrameReuseRaceCorrect(t, false) })
}

func testFrameReuseRaceCorrect(t *testing.T, meta bool) {
	const iters = 200

	encoded := preEncodeReuseRaceStream(t, iters)
	fr := newReuseRaceFramer(encoded, meta)

	ch := make(chan framedRead)
	exitc := runReadFramesGoroutine(t, fr, ch)

	// sink lets the compiler keep the work observable.
	var sink atomic.Uint64
	frames := 0

	for {
		select {
		case res := <-ch:
			if res.err == io.EOF {
				close(res.done)
				goto drain
			}
			if res.err != nil {
				t.Fatalf("unexpected ReadFrame error: %v", res.err)
			}
			frames++
			// Consume the slice/fields synchronously. Once we
			// signal done, we forget the frame.
			switch f := res.frame.(type) {
			case *DataFrame:
				sink.Add(uint64(consumeSliceWork(f.Data())))
			case *HeadersFrame:
				sink.Add(uint64(consumeSliceWork(f.HeaderBlockFragment())))
			case *ContinuationFrame:
				sink.Add(uint64(consumeSliceWork(f.HeaderBlockFragment())))
			case *MetaHeadersFrame:
				sink.Add(uint64(consumeHeaderFieldWork(f.Fields)))
			case *WindowUpdateFrame:
				sink.Add(uint64(f.Increment))
			default:
				t.Fatalf("unexpected frame type %T", res.frame)
			}
			close(res.done) // gate: reader may proceed.
		case <-time.After(30 * time.Second):
			t.Fatal("timed out waiting for next frame")
		}
	}
drain:
	<-exitc
	// meta: DATA, WINDOW_UPDATE, and two MetaHeadersFrames per iter.
	// raw: the second HEADERS surfaces as HEADERS+CONTINUATION.
	wantFrames := iters * 4
	if !meta {
		wantFrames = iters * 5
	}
	if frames != wantFrames {
		t.Errorf("consumed %d frames, want %d", frames, wantFrames)
	}
	t.Logf("consumed %d frames, sink=%d", frames, sink.Load())
}

// TestFrameReuseRaceAdversarial deliberately violates the reuse
// contract: the consumer hands a retained slice to a sidecar
// goroutine that keeps reading it indefinitely, while the gate is
// signaled immediately so the reader proceeds to overwrite the same
// memory. Under -race, this MUST fire.
//
// It is guarded behind H2_REUSE_RACE_NEGATIVE=1 because it is a
// negative control: failure (i.e., no race detected) is what we want
// to be loud about during development, but a passing test on stock
// CI is uninteresting. Run it as:
//
//	H2_REUSE_RACE_NEGATIVE=1 go test -race -run TestFrameReuseRaceAdversarial
//
// What it would catch in production code: any handler/code path that
// holds onto Data, HeaderBlockFragment, or the MetaHeadersFrame
// Fields slice past the readMore call.
func TestFrameReuseRaceAdversarial(t *testing.T) {
	if os.Getenv("H2_REUSE_RACE_NEGATIVE") != "1" {
		t.Skip("skipping adversarial negative-control test; " +
			"set H2_REUSE_RACE_NEGATIVE=1 to run under -race")
	}
	t.Run("meta", func(t *testing.T) { testFrameReuseRaceAdversarial(t, true) })
	t.Run("raw", func(t *testing.T) { testFrameReuseRaceAdversarial(t, false) })
}

func testFrameReuseRaceAdversarial(t *testing.T, meta bool) {
	// 50 outer iterations -> 200+ frames. That is far more than
	// enough scheduler interleavings for the race detector to
	// observe at least one concurrent read/write conflict.
	const iters = 50
	const maxLiveAttackers = 8 // cap concurrent attacker goroutines

	encoded := preEncodeReuseRaceStream(t, iters)
	fr := newReuseRaceFramer(encoded, meta)

	ch := make(chan framedRead)
	exitc := runReadFramesGoroutine(t, fr, ch)

	// stopAttackers is closed when the test is winding down to let
	// any retained-slice readers exit.
	stopAttackers := make(chan struct{})
	var attackers sync.WaitGroup
	// Token bucket bounds the number of attacker goroutines alive
	// at any time; without it the race detector slows to a crawl as
	// many hundreds of goroutines all hammer the read buffer.
	tokens := make(chan struct{}, maxLiveAttackers)

	// retainedRefs accumulates references we deliberately leak past
	// the gate. Holding them in the test goroutine keeps them
	// reachable for the GC's view, but the race detector still
	// catches read/write conflicts on the underlying memory.
	type retainedSlice struct {
		b []byte
	}
	type retainedFields struct {
		hf []hpack.HeaderField
	}
	var refs []any

	startAttacker := func(read func()) {
		// Block briefly if too many attackers are already running.
		// This will bound the runtime overhead without weakening the
		// race detector signal (each retained reference still gets
		// a goroutine that overlaps the next ReadFrame).
		select {
		case tokens <- struct{}{}:
		case <-stopAttackers:
			return
		}
		attackers.Add(1)
		go func() {
			defer attackers.Done()
			defer func() { <-tokens }()
			for i := 0; i < 200; i++ {
				select {
				case <-stopAttackers:
					return
				default:
				}
				read()
			}
		}()
	}

	var sink atomic.Uint64

	for {
		select {
		case res := <-ch:
			if res.err == io.EOF {
				close(res.done)
				goto drain
			}
			if res.err != nil {
				t.Fatalf("unexpected ReadFrame error: %v", res.err)
			}
			switch f := res.frame.(type) {
			case *DataFrame:
				retained := retainedSlice{b: f.Data()}
				refs = append(refs, retained)
				startAttacker(func() {
					sink.Add(uint64(consumeSliceWork(retained.b)))
				})
			case *HeadersFrame:
				// The fragment aliases the framer's read buffer,
				// which the next ReadFrame overwrites.
				retained := retainedSlice{b: f.HeaderBlockFragment()}
				refs = append(refs, retained)
				startAttacker(func() {
					sink.Add(uint64(consumeSliceWork(retained.b)))
				})
			case *ContinuationFrame:
				retained := retainedSlice{b: f.HeaderBlockFragment()}
				refs = append(refs, retained)
				startAttacker(func() {
					sink.Add(uint64(consumeSliceWork(retained.b)))
				})
			case *MetaHeadersFrame:
				// Retaining the Fields slice past the gate violates
				// the reuse contract: the next meta parse clears and
				// re-appends into the same backing array, so the
				// attacker's reads of the HeaderField structs race
				// with the Framer's writes.
				retained := retainedFields{hf: f.Fields}
				refs = append(refs, retained)
				startAttacker(func() {
					sink.Add(uint64(consumeHeaderFieldWork(retained.hf)))
				})
			case *WindowUpdateFrame:
				// No slice to retain on WindowUpdateFrame, just
				// signal done.
			default:
				t.Fatalf("unexpected frame type %T", res.frame)
			}
			// Adversarial: signal done immediately, even though
			// we still have outstanding readers of the frame's
			// memory.
			close(res.done)
		case <-time.After(30 * time.Second):
			t.Fatal("timed out waiting for next frame")
		}
	}
drain:
	close(stopAttackers)
	attackers.Wait()
	<-exitc
	// Force refs to outlive the loop above so the compiler does
	// not eliminate the retentions.
	if len(refs) == 0 {
		t.Fatalf("expected retained references, got 0")
	}
	t.Logf("retained %d references, sink=%d", len(refs), sink.Load())
}
