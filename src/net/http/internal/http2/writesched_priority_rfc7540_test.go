// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package http2

import (
	"bytes"
	"fmt"
	"sort"
	"testing"
)

func defaultPriorityWriteScheduler() *priorityWriteSchedulerRFC7540 {
	return NewPriorityWriteScheduler(nil).(*priorityWriteSchedulerRFC7540)
}

func checkPriorityWellFormed(ws *priorityWriteSchedulerRFC7540) error {
	for id, n := range ws.nodes {
		if id != n.id {
			return fmt.Errorf("bad ws.nodes: ws.nodes[%d] = %d", id, n.id)
		}
		if n.parent == nil {
			if n.next != nil || n.prev != nil {
				return fmt.Errorf("bad node %d: nil parent but prev/next not nil", id)
			}
			continue
		}
		found := false
		for k := n.parent.kids; k != nil; k = k.next {
			if k.id == id {
				found = true
				break
			}
		}
		if !found {
			return fmt.Errorf("bad node %d: not found in parent %d kids list", id, n.parent.id)
		}
	}
	return nil
}

func fmtTree(ws *priorityWriteSchedulerRFC7540, fmtNode func(*priorityNodeRFC7540) string) string {
	var ids []int
	for _, n := range ws.nodes {
		ids = append(ids, int(n.id))
	}
	sort.Ints(ids)

	var buf bytes.Buffer
	for _, id := range ids {
		if buf.Len() != 0 {
			buf.WriteString(" ")
		}
		if id == 0 {
			buf.WriteString(fmtNode(&ws.root))
		} else {
			buf.WriteString(fmtNode(ws.nodes[uint32(id)]))
		}
	}
	return buf.String()
}

func fmtNodeParentSkipRoot(n *priorityNodeRFC7540) string {
	switch {
	case n.id == 0:
		return ""
	case n.parent == nil:
		return fmt.Sprintf("%d{parent:nil}", n.id)
	default:
		return fmt.Sprintf("%d{parent:%d}", n.id, n.parent.id)
	}
}

func fmtNodeWeightParentSkipRoot(n *priorityNodeRFC7540) string {
	switch {
	case n.id == 0:
		return ""
	case n.parent == nil:
		return fmt.Sprintf("%d{weight:%d,parent:nil}", n.id, n.weight)
	default:
		return fmt.Sprintf("%d{weight:%d,parent:%d}", n.id, n.weight, n.parent.id)
	}
}

func TestPriorityTwoStreams(t *testing.T) {
	ws := defaultPriorityWriteScheduler()
	ws.OpenStream(1, OpenStreamOptions{})
	ws.OpenStream(2, OpenStreamOptions{})

	want := "1{weight:15,parent:0} 2{weight:15,parent:0}"
	if got := fmtTree(ws, fmtNodeWeightParentSkipRoot); got != want {
		t.Errorf("After open\ngot  %q\nwant %q", got, want)
	}

	// Move 1's parent to 2.
	ws.AdjustStream(1, PriorityParam{
		StreamDep: 2,
		Weight:    32,
		Exclusive: false,
	})
	want = "1{weight:32,parent:2} 2{weight:15,parent:0}"
	if got := fmtTree(ws, fmtNodeWeightParentSkipRoot); got != want {
		t.Errorf("After adjust\ngot  %q\nwant %q", got, want)
	}

	if err := checkPriorityWellFormed(ws); err != nil {
		t.Error(err)
	}
}

func TestPriorityAdjustExclusiveZero(t *testing.T) {
	// 1, 2, and 3 are all children of the 0 stream.
	// Exclusive reprioritization to any of the streams should bring
	// the rest of the streams under the reprioritized stream.
	ws := defaultPriorityWriteScheduler()
	ws.OpenStream(1, OpenStreamOptions{})
	ws.OpenStream(2, OpenStreamOptions{})
	ws.OpenStream(3, OpenStreamOptions{})

	want := "1{weight:15,parent:0} 2{weight:15,parent:0} 3{weight:15,parent:0}"
	if got := fmtTree(ws, fmtNodeWeightParentSkipRoot); got != want {
		t.Errorf("After open\ngot  %q\nwant %q", got, want)
	}

	ws.AdjustStream(2, PriorityParam{
		StreamDep: 0,
		Weight:    20,
		Exclusive: true,
	})
	want = "1{weight:15,parent:2} 2{weight:20,parent:0} 3{weight:15,parent:2}"
	if got := fmtTree(ws, fmtNodeWeightParentSkipRoot); got != want {
		t.Errorf("After adjust\ngot  %q\nwant %q", got, want)
	}

	if err := checkPriorityWellFormed(ws); err != nil {
		t.Error(err)
	}
}

func TestPriorityAdjustOwnParent(t *testing.T) {
	// Assigning a node as its own parent should have no effect.
	ws := defaultPriorityWriteScheduler()
	ws.OpenStream(1, OpenStreamOptions{})
	ws.OpenStream(2, OpenStreamOptions{})
	ws.AdjustStream(2, PriorityParam{
		StreamDep: 2,
		Weight:    20,
		Exclusive: true,
	})
	want := "1{weight:15,parent:0} 2{weight:15,parent:0}"
	if got := fmtTree(ws, fmtNodeWeightParentSkipRoot); got != want {
		t.Errorf("After adjust\ngot  %q\nwant %q", got, want)
	}
	if err := checkPriorityWellFormed(ws); err != nil {
		t.Error(err)
	}
}

func TestPriorityClosedStreams(t *testing.T) {
	ws := NewPriorityWriteScheduler(&PriorityWriteSchedulerConfig{MaxClosedNodesInTree: 2}).(*priorityWriteSchedulerRFC7540)
	ws.OpenStream(1, OpenStreamOptions{})
	ws.OpenStream(2, OpenStreamOptions{PusherID: 1})
	ws.OpenStream(3, OpenStreamOptions{PusherID: 2})
	ws.OpenStream(4, OpenStreamOptions{PusherID: 3})

	// Close the first three streams. We lose 1, but keep 2 and 3.
	ws.CloseStream(1)
	ws.CloseStream(2)
	ws.CloseStream(3)

	want := "2{weight:15,parent:0} 3{weight:15,parent:2} 4{weight:15,parent:3}"
	if got := fmtTree(ws, fmtNodeWeightParentSkipRoot); got != want {
		t.Errorf("After close\ngot  %q\nwant %q", got, want)
	}
	if err := checkPriorityWellFormed(ws); err != nil {
		t.Error(err)
	}

	// Adding a stream as an exclusive child of 1 gives it default
	// priorities, since 1 is gone.
	ws.OpenStream(5, OpenStreamOptions{})
	ws.AdjustStream(5, PriorityParam{StreamDep: 1, Weight: 15, Exclusive: true})

	// Adding a stream as an exclusive child of 2 should work, since 2 is not gone.
	ws.OpenStream(6, OpenStreamOptions{})
	ws.AdjustStream(6, PriorityParam{StreamDep: 2, Weight: 15, Exclusive: true})

	want = "2{weight:15,parent:0} 3{weight:15,parent:6} 4{weight:15,parent:3} 5{weight:15,parent:0} 6{weight:15,parent:2}"
	if got := fmtTree(ws, fmtNodeWeightParentSkipRoot); got != want {
		t.Errorf("After add streams\ngot  %q\nwant %q", got, want)
	}
	if err := checkPriorityWellFormed(ws); err != nil {
		t.Error(err)
	}
}

func TestPriorityClosedStreamsDisabled(t *testing.T) {
	ws := NewPriorityWriteScheduler(&PriorityWriteSchedulerConfig{}).(*priorityWriteSchedulerRFC7540)
	ws.OpenStream(1, OpenStreamOptions{})
	ws.OpenStream(2, OpenStreamOptions{PusherID: 1})
	ws.OpenStream(3, OpenStreamOptions{PusherID: 2})

	// Close the first two streams. We keep only 3.
	ws.CloseStream(1)
	ws.CloseStream(2)

	want := "3{weight:15,parent:0}"
	if got := fmtTree(ws, fmtNodeWeightParentSkipRoot); got != want {
		t.Errorf("After close\ngot  %q\nwant %q", got, want)
	}
	if err := checkPriorityWellFormed(ws); err != nil {
		t.Error(err)
	}
}

func TestPriorityIdleStreams(t *testing.T) {
	ws := NewPriorityWriteScheduler(&PriorityWriteSchedulerConfig{MaxIdleNodesInTree: 2}).(*priorityWriteSchedulerRFC7540)
	ws.AdjustStream(1, PriorityParam{StreamDep: 0, Weight: 15}) // idle
	ws.AdjustStream(2, PriorityParam{StreamDep: 0, Weight: 15}) // idle
	ws.AdjustStream(3, PriorityParam{StreamDep: 2, Weight: 20}) // idle
	ws.OpenStream(4, OpenStreamOptions{})
	ws.OpenStream(5, OpenStreamOptions{})
	ws.OpenStream(6, OpenStreamOptions{})
	ws.AdjustStream(4, PriorityParam{StreamDep: 1, Weight: 15})
	ws.AdjustStream(5, PriorityParam{StreamDep: 2, Weight: 15})
	ws.AdjustStream(6, PriorityParam{StreamDep: 3, Weight: 15})

	want := "2{weight:15,parent:0} 3{weight:20,parent:2} 4{weight:15,parent:0} 5{weight:15,parent:2} 6{weight:15,parent:3}"
	if got := fmtTree(ws, fmtNodeWeightParentSkipRoot); got != want {
		t.Errorf("After open\ngot  %q\nwant %q", got, want)
	}
	if err := checkPriorityWellFormed(ws); err != nil {
		t.Error(err)
	}
}

func TestPriorityIdleStreamsDisabled(t *testing.T) {
	ws := NewPriorityWriteScheduler(&PriorityWriteSchedulerConfig{}).(*priorityWriteSchedulerRFC7540)
	ws.AdjustStream(1, PriorityParam{StreamDep: 0, Weight: 15}) // idle
	ws.AdjustStream(2, PriorityParam{StreamDep: 0, Weight: 15}) // idle
	ws.AdjustStream(3, PriorityParam{StreamDep: 2, Weight: 20}) // idle
	ws.OpenStream(4, OpenStreamOptions{})

	want := "4{weight:15,parent:0}"
	if got := fmtTree(ws, fmtNodeWeightParentSkipRoot); got != want {
		t.Errorf("After open\ngot  %q\nwant %q", got, want)
	}
	if err := checkPriorityWellFormed(ws); err != nil {
		t.Error(err)
	}
}

func TestPrioritySection531NonExclusive(t *testing.T) {
	// Example from RFC 7540 Section 5.3.1.
	// A,B,C,D = 1,2,3,4
	ws := defaultPriorityWriteScheduler()
	ws.OpenStream(1, OpenStreamOptions{})
	ws.OpenStream(2, OpenStreamOptions{PusherID: 1})
	ws.OpenStream(3, OpenStreamOptions{PusherID: 1})
	ws.OpenStream(4, OpenStreamOptions{})
	ws.AdjustStream(4, PriorityParam{
		StreamDep: 1,
		Weight:    15,
		Exclusive: false,
	})
	want := "1{parent:0} 2{parent:1} 3{parent:1} 4{parent:1}"
	if got := fmtTree(ws, fmtNodeParentSkipRoot); got != want {
		t.Errorf("After adjust\ngot  %q\nwant %q", got, want)
	}
	if err := checkPriorityWellFormed(ws); err != nil {
		t.Error(err)
	}
}

func TestPrioritySection531Exclusive(t *testing.T) {
	// Example from RFC 7540 Section 5.3.1.
	// A,B,C,D = 1,2,3,4
	ws := defaultPriorityWriteScheduler()
	ws.OpenStream(1, OpenStreamOptions{})
	ws.OpenStream(2, OpenStreamOptions{PusherID: 1})
	ws.OpenStream(3, OpenStreamOptions{PusherID: 1})
	ws.OpenStream(4, OpenStreamOptions{})
	ws.AdjustStream(4, PriorityParam{
		StreamDep: 1,
		Weight:    15,
		Exclusive: true,
	})
	want := "1{parent:0} 2{parent:4} 3{parent:4} 4{parent:1}"
	if got := fmtTree(ws, fmtNodeParentSkipRoot); got != want {
		t.Errorf("After adjust\ngot  %q\nwant %q", got, want)
	}
	if err := checkPriorityWellFormed(ws); err != nil {
		t.Error(err)
	}
}

func makeSection533Tree() *priorityWriteSchedulerRFC7540 {
	// Initial tree from RFC 7540 Section 5.3.3.
	// A,B,C,D,E,F = 1,2,3,4,5,6
	ws := defaultPriorityWriteScheduler()
	ws.OpenStream(1, OpenStreamOptions{})
	ws.OpenStream(2, OpenStreamOptions{PusherID: 1})
	ws.OpenStream(3, OpenStreamOptions{PusherID: 1})
	ws.OpenStream(4, OpenStreamOptions{PusherID: 3})
	ws.OpenStream(5, OpenStreamOptions{PusherID: 3})
	ws.OpenStream(6, OpenStreamOptions{PusherID: 4})
	return ws
}

func TestPrioritySection533NonExclusive(t *testing.T) {
	// Example from RFC 7540 Section 5.3.3.
	// A,B,C,D,E,F = 1,2,3,4,5,6
	ws := defaultPriorityWriteScheduler()
	ws.OpenStream(1, OpenStreamOptions{})
	ws.OpenStream(2, OpenStreamOptions{PusherID: 1})
	ws.OpenStream(3, OpenStreamOptions{PusherID: 1})
	ws.OpenStream(4, OpenStreamOptions{PusherID: 3})
	ws.OpenStream(5, OpenStreamOptions{PusherID: 3})
	ws.OpenStream(6, OpenStreamOptions{PusherID: 4})
	ws.AdjustStream(1, PriorityParam{
		StreamDep: 4,
		Weight:    15,
		Exclusive: false,
	})
	want := "1{parent:4} 2{parent:1} 3{parent:1} 4{parent:0} 5{parent:3} 6{parent:4}"
	if got := fmtTree(ws, fmtNodeParentSkipRoot); got != want {
		t.Errorf("After adjust\ngot  %q\nwant %q", got, want)
	}
	if err := checkPriorityWellFormed(ws); err != nil {
		t.Error(err)
	}
}

func TestPrioritySection533Exclusive(t *testing.T) {
	// Example from RFC 7540 Section 5.3.3.
	// A,B,C,D,E,F = 1,2,3,4,5,6
	ws := defaultPriorityWriteScheduler()
	ws.OpenStream(1, OpenStreamOptions{})
	ws.OpenStream(2, OpenStreamOptions{PusherID: 1})
	ws.OpenStream(3, OpenStreamOptions{PusherID: 1})
	ws.OpenStream(4, OpenStreamOptions{PusherID: 3})
	ws.OpenStream(5, OpenStreamOptions{PusherID: 3})
	ws.OpenStream(6, OpenStreamOptions{PusherID: 4})
	ws.AdjustStream(1, PriorityParam{
		StreamDep: 4,
		Weight:    15,
		Exclusive: true,
	})
	want := "1{parent:4} 2{parent:1} 3{parent:1} 4{parent:0} 5{parent:3} 6{parent:1}"
	if got := fmtTree(ws, fmtNodeParentSkipRoot); got != want {
		t.Errorf("After adjust\ngot  %q\nwant %q", got, want)
	}
	if err := checkPriorityWellFormed(ws); err != nil {
		t.Error(err)
	}
}

func checkPopAll(ws WriteScheduler, order []uint32) error {
	for k, id := range order {
		wr, ok := ws.Pop()
		if !ok {
			return fmt.Errorf("Pop[%d]: got ok=false, want %d (order=%v)", k, id, order)
		}
		if got := wr.StreamID(); got != id {
			return fmt.Errorf("Pop[%d]: got %v, want %d (order=%v)", k, got, id, order)
		}
	}
	wr, ok := ws.Pop()
	if ok {
		return fmt.Errorf("Pop[%d]: got %v, want ok=false (order=%v)", len(order), wr.StreamID(), order)
	}
	return nil
}

func TestPriorityPopFrom533Tree(t *testing.T) {
	ws := makeSection533Tree()

	ws.Push(makeWriteHeadersRequest(3 /*C*/))
	ws.Push(makeWriteNonStreamRequest())
	ws.Push(makeWriteHeadersRequest(5 /*E*/))
	ws.Push(makeWriteHeadersRequest(1 /*A*/))
	t.Log("tree:", fmtTree(ws, fmtNodeParentSkipRoot))

	if err := checkPopAll(ws, []uint32{0 /*NonStream*/, 1, 3, 5}); err != nil {
		t.Error(err)
	}
}

// #49741 RST_STREAM and Control frames should have more priority than data
// frames to avoid blocking streams caused by clients not able to drain the
// queue.
func TestPriorityRSTFrames(t *testing.T) {
	ws := defaultPriorityWriteScheduler()
	ws.OpenStream(1, OpenStreamOptions{})

	sc := &serverConn{maxFrameSize: 16}
	st1 := &stream{id: 1, sc: sc}

	ws.Push(FrameWriteRequest{&writeData{1, make([]byte, 16), false}, st1, nil})
	ws.Push(FrameWriteRequest{&writeData{1, make([]byte, 16), false}, st1, nil})
	ws.Push(makeWriteRSTStream(1))
	// No flow-control bytes available.
	wr, ok := ws.Pop()
	if !ok {
		t.Fatalf("Pop should work for control frames and not be limited by flow control")
	}
	if _, ok := wr.write.(StreamError); !ok {
		t.Fatal("expected RST stream frames first", wr)
	}
}

func TestPriorityPopFromLinearTree(t *testing.T) {
	ws := defaultPriorityWriteScheduler()
	ws.OpenStream(1, OpenStreamOptions{})
	ws.OpenStream(2, OpenStreamOptions{PusherID: 1})
	ws.OpenStream(3, OpenStreamOptions{PusherID: 2})
	ws.OpenStream(4, OpenStreamOptions{PusherID: 3})

	ws.Push(makeWriteHeadersRequest(3))
	ws.Push(makeWriteHeadersRequest(4))
	ws.Push(makeWriteHeadersRequest(1))
	ws.Push(makeWriteHeadersRequest(2))
	ws.Push(makeWriteNonStreamRequest())
	ws.Push(makeWriteNonStreamRequest())
	t.Log("tree:", fmtTree(ws, fmtNodeParentSkipRoot))

	if err := checkPopAll(ws, []uint32{0, 0 /*NonStreams*/, 1, 2, 3, 4}); err != nil {
		t.Error(err)
	}
}

func TestPriorityFlowControl(t *testing.T) {
	ws := NewPriorityWriteScheduler(&PriorityWriteSchedulerConfig{ThrottleOutOfOrderWrites: false})
	ws.OpenStream(1, OpenStreamOptions{})
	ws.OpenStream(2, OpenStreamOptions{PusherID: 1})

	sc := &serverConn{maxFrameSize: 16}
	st1 := &stream{id: 1, sc: sc}
	st2 := &stream{id: 2, sc: sc}

	ws.Push(FrameWriteRequest{&writeData{1, make([]byte, 16), false}, st1, nil})
	ws.Push(FrameWriteRequest{&writeData{2, make([]byte, 16), false}, st2, nil})
	ws.AdjustStream(2, PriorityParam{StreamDep: 1})

	// No flow-control bytes available.
	if wr, ok := ws.Pop(); ok {
		t.Fatalf("Pop(limited by flow control)=%v,true, want false", wr)
	}

	// Add enough flow-control bytes to write st2 in two Pop calls.
	// Should write data from st2 even though it's lower priority than st1.
	for i := 1; i <= 2; i++ {
		st2.flow.add(8)
		wr, ok := ws.Pop()
		if !ok {
			t.Fatalf("Pop(%d)=false, want true", i)
		}
		if got, want := wr.DataSize(), 8; got != want {
			t.Fatalf("Pop(%d)=%d bytes, want %d bytes", i, got, want)
		}
	}
}

func TestPriorityThrottleOutOfOrderWrites(t *testing.T) {
	ws := NewPriorityWriteScheduler(&PriorityWriteSchedulerConfig{ThrottleOutOfOrderWrites: true})
	ws.OpenStream(1, OpenStreamOptions{})
	ws.OpenStream(2, OpenStreamOptions{PusherID: 1})

	sc := &serverConn{maxFrameSize: 4096}
	st1 := &stream{id: 1, sc: sc}
	st2 := &stream{id: 2, sc: sc}
	st1.flow.add(4096)
	st2.flow.add(4096)
	ws.Push(FrameWriteRequest{&writeData{2, make([]byte, 4096), false}, st2, nil})
	ws.AdjustStream(2, PriorityParam{StreamDep: 1})

	// We have enough flow-control bytes to write st2 in a single Pop call.
	// However, due to out-of-order write throttling, the first call should
	// only write 1KB.
	wr, ok := ws.Pop()
	if !ok {
		t.Fatalf("Pop(st2.first)=false, want true")
	}
	if got, want := wr.StreamID(), uint32(2); got != want {
		t.Fatalf("Pop(st2.first)=stream %d, want stream %d", got, want)
	}
	if got, want := wr.DataSize(), 1024; got != want {
		t.Fatalf("Pop(st2.first)=%d bytes, want %d bytes", got, want)
	}

	// Now add data on st1. This should take precedence.
	ws.Push(FrameWriteRequest{&writeData{1, make([]byte, 4096), false}, st1, nil})
	wr, ok = ws.Pop()
	if !ok {
		t.Fatalf("Pop(st1)=false, want true")
	}
	if got, want := wr.StreamID(), uint32(1); got != want {
		t.Fatalf("Pop(st1)=stream %d, want stream %d", got, want)
	}
	if got, want := wr.DataSize(), 4096; got != want {
		t.Fatalf("Pop(st1)=%d bytes, want %d bytes", got, want)
	}

	// Should go back to writing 1KB from st2.
	wr, ok = ws.Pop()
	if !ok {
		t.Fatalf("Pop(st2.last)=false, want true")
	}
	if got, want := wr.StreamID(), uint32(2); got != want {
		t.Fatalf("Pop(st2.last)=stream %d, want stream %d", got, want)
	}
	if got, want := wr.DataSize(), 1024; got != want {
		t.Fatalf("Pop(st2.last)=%d bytes, want %d bytes", got, want)
	}
}

func TestPriorityWeights(t *testing.T) {
	ws := defaultPriorityWriteScheduler()
	ws.OpenStream(1, OpenStreamOptions{})
	ws.OpenStream(2, OpenStreamOptions{})

	sc := &serverConn{maxFrameSize: 8}
	st1 := &stream{id: 1, sc: sc}
	st2 := &stream{id: 2, sc: sc}
	st1.flow.add(40)
	st2.flow.add(40)

	ws.Push(FrameWriteRequest{&writeData{1, make([]byte, 40), false}, st1, nil})
	ws.Push(FrameWriteRequest{&writeData{2, make([]byte, 40), false}, st2, nil})
	ws.AdjustStream(1, PriorityParam{StreamDep: 0, Weight: 34})
	ws.AdjustStream(2, PriorityParam{StreamDep: 0, Weight: 9})

	// st1 gets 3.5x the bandwidth of st2 (3.5 = (34+1)/(9+1)).
	// The maximum frame size is 8 bytes. The write sequence should be:
	//   st1, total bytes so far is (st1=8,  st=0)
	//   st2, total bytes so far is (st1=8,  st=8)
	//   st1, total bytes so far is (st1=16, st=8)
	//   st1, total bytes so far is (st1=24, st=8)   // 3x bandwidth
	//   st1, total bytes so far is (st1=32, st=8)   // 4x bandwidth
	//   st2, total bytes so far is (st1=32, st=16)  // 2x bandwidth
	//   st1, total bytes so far is (st1=40, st=16)
	//   st2, total bytes so far is (st1=40, st=24)
	//   st2, total bytes so far is (st1=40, st=32)
	//   st2, total bytes so far is (st1=40, st=40)
	if err := checkPopAll(ws, []uint32{1, 2, 1, 1, 1, 2, 1, 2, 2, 2}); err != nil {
		t.Error(err)
	}
}

func TestPriorityWeightsMinMax(t *testing.T) {
	ws := defaultPriorityWriteScheduler()
	ws.OpenStream(1, OpenStreamOptions{})
	ws.OpenStream(2, OpenStreamOptions{})

	sc := &serverConn{maxFrameSize: 8}
	st1 := &stream{id: 1, sc: sc}
	st2 := &stream{id: 2, sc: sc}
	st1.flow.add(40)
	st2.flow.add(40)

	// st2 gets 256x the bandwidth of st1 (256 = (255+1)/(0+1)).
	// The maximum frame size is 8 bytes. The write sequence should be:
	//   st2, total bytes so far is (st1=0,  st=8)
	//   st1, total bytes so far is (st1=8,  st=8)
	//   st2, total bytes so far is (st1=8,  st=16)
	//   st2, total bytes so far is (st1=8,  st=24)
	//   st2, total bytes so far is (st1=8,  st=32)
	//   st2, total bytes so far is (st1=8,  st=40)  // 5x bandwidth
	//   st1, total bytes so far is (st1=16, st=40)
	//   st1, total bytes so far is (st1=24, st=40)
	//   st1, total bytes so far is (st1=32, st=40)
	//   st1, total bytes so far is (st1=40, st=40)
	ws.Push(FrameWriteRequest{&writeData{1, make([]byte, 40), false}, st1, nil})
	ws.Push(FrameWriteRequest{&writeData{2, make([]byte, 40), false}, st2, nil})
	ws.AdjustStream(1, PriorityParam{StreamDep: 0, Weight: 0})
	ws.AdjustStream(2, PriorityParam{StreamDep: 0, Weight: 255})

	if err := checkPopAll(ws, []uint32{2, 1, 2, 2, 2, 2, 1, 1, 1, 1}); err != nil {
		t.Error(err)
	}
}

func TestPriorityRstStreamOnNonOpenStreams(t *testing.T) {
	ws := NewPriorityWriteScheduler(&PriorityWriteSchedulerConfig{
		MaxClosedNodesInTree: 0,
		MaxIdleNodesInTree:   0,
	})
	ws.OpenStream(1, OpenStreamOptions{})
	ws.CloseStream(1)
	ws.Push(FrameWriteRequest{write: streamError(1, ErrCodeProtocol)})
	ws.Push(FrameWriteRequest{write: streamError(2, ErrCodeProtocol)})

	if err := checkPopAll(ws, []uint32{1, 2}); err != nil {
		t.Error(err)
	}
}

// https://go.dev/issue/66514
func TestPriorityIssue66514(t *testing.T) {
	addDep := func(ws *priorityWriteSchedulerRFC7540, child uint32, parent uint32) {
		ws.AdjustStream(child, PriorityParam{
			StreamDep: parent,
			Exclusive: false,
			Weight:    16,
		})
	}

	validateDepTree := func(ws *priorityWriteSchedulerRFC7540, id uint32, t *testing.T) {
		for n := ws.nodes[id]; n != nil; n = n.parent {
			if n.parent == nil {
				if n.id != uint32(0) {
					t.Errorf("detected nodes not parented to 0")
				}
			}
		}
	}

	ws := NewPriorityWriteScheduler(nil).(*priorityWriteSchedulerRFC7540)

	// Root entry
	addDep(ws, uint32(1), uint32(0))
	addDep(ws, uint32(3), uint32(1))
	addDep(ws, uint32(5), uint32(1))

	for id := uint32(7); id < uint32(100); id += uint32(4) {
		addDep(ws, id, id-uint32(4))
		addDep(ws, id+uint32(2), id-uint32(4))
		validateDepTree(ws, id, t)
	}
}
