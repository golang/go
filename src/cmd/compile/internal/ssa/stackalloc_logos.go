// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import (
	"fmt"
	"math/bits"
)

type stackLive struct {
	ids   []ID
	words [][]stackLiveWord
}

type stackLiveWord struct {
	index int
	bits  uint64
}

func (l stackLive) blockWords(bid ID) []stackLiveWord {
	row := l.words[bid]
	return row[:len(row):len(row)]
}

func (s *stackAllocState) clearLiveLogos() {
	s.live = stackLive{}
}

func (s *stackAllocState) addLiveLogos(live *sparseSet, bid ID) {
	for _, chunk := range s.live.blockWords(bid) {
		for word := chunk.bits; word != 0; {
			live.add(s.live.ids[chunk.index*64+bits.TrailingZeros64(word)])
			word &= word - 1
		}
	}
}

func (s *stackAllocState) computeLiveLogos(spillLive [][]ID) {
	s.collectLiveUses(spillLive)
	s.initializeLiveSlots()
	s.computeLiveWords()

	if s.f.pass.debug > stackDebug {
		for _, b := range s.f.Blocks {
			var ids []ID
			for _, chunk := range s.live.blockWords(b.ID) {
				for word := chunk.bits; word != 0; {
					ids = append(ids, s.live.ids[chunk.index*64+bits.TrailingZeros64(word)])
					word &= word - 1
				}
			}
			fmt.Printf("stacklive %s %v\n", b, ids)
		}
	}
}

func (s *stackAllocState) collectLiveUses(spillLive [][]ID) {
	f := s.f
	for _, b := range f.Blocks {
		for _, spillvid := range spillLive[b.ID] {
			val := &s.values[spillvid]
			val.addUseBlock(b, true)
		}
		for _, v := range b.Values {
			for i, a := range v.Args {
				val := &s.values[a.ID]
				useBlock := b
				forceLiveout := false
				if v.Op == OpPhi {
					useBlock = b.Preds[i].b
					forceLiveout = true
					if spill := val.spill; spill != nil {
						s.values[spill.ID].addUseBlock(useBlock, true)
					}
				}
				if !val.needSlot {
					continue
				}
				val.addUseBlock(useBlock, forceLiveout)
			}
		}
	}
}

func (s *stackAllocState) initializeLiveSlots() {
	f := s.f
	nslots := 0
	for _, v := range s.values {
		if v.needSlot {
			nslots++
		}
	}
	s.live = stackLive{
		ids:   make([]ID, nslots),
		words: make([][]stackLiveWord, f.NumBlocks()),
	}
	ordinal := 0
	for vid, v := range s.values {
		if v.needSlot {
			s.live.ids[ordinal] = ID(vid)
			ordinal++
		}
	}
}

type stackLiveBlock struct {
	defs    uint64
	in      uint64
	out     uint64
	pending uint64
}

type stackLiveFlow struct {
	blocks  []stackLiveBlock
	touched []ID
	queue   []ID
	offsets []int
	preds   []ID
}

func newStackLiveFlow(f *Func) stackLiveFlow {
	nblocks := f.NumBlocks()
	offsets := make([]int, nblocks+1)
	for _, b := range f.Blocks {
		offsets[b.ID+1] = len(b.Preds)
	}
	for i := 1; i < len(offsets); i++ {
		offsets[i] += offsets[i-1]
	}
	ids := make([]ID, 2*nblocks+offsets[nblocks])
	flow := stackLiveFlow{
		blocks:  make([]stackLiveBlock, nblocks),
		touched: ids[:0:nblocks],
		queue:   ids[nblocks : nblocks : 2*nblocks],
		offsets: offsets,
		preds:   ids[2*nblocks:],
	}
	for _, b := range f.Blocks {
		for i, edge := range b.Preds {
			flow.preds[offsets[b.ID]+i] = edge.b.ID
		}
	}
	return flow
}

func (flow *stackLiveFlow) touch(bid ID) {
	block := &flow.blocks[bid]
	if block.defs|block.in|block.out == 0 {
		flow.touched = append(flow.touched, bid)
	}
}

func (flow *stackLiveFlow) addInput(bid ID, mask uint64) {
	block := &flow.blocks[bid]
	fresh := mask &^ (block.defs | block.in)
	if fresh == 0 {
		return
	}
	flow.touch(bid)
	block.in |= fresh
	if block.pending == 0 {
		flow.queue = append(flow.queue, bid)
	}
	block.pending |= fresh
}

func (s *stackAllocState) seedLiveWord(flow *stackLiveFlow, ids []ID) {
	for bit, vid := range ids {
		bid := s.values[vid].defBlock
		flow.touch(bid)
		flow.blocks[bid].defs |= uint64(1) << uint(bit)
	}
	for bit, vid := range ids {
		mask := uint64(1) << uint(bit)
		for _, use := range s.values[vid].useBlocks {
			if use.liveout {
				flow.touch(use.b.ID)
				flow.blocks[use.b.ID].out |= mask
			}
			flow.addInput(use.b.ID, mask)
		}
	}
}

func (flow *stackLiveFlow) propagate() {
	for len(flow.queue) > 0 {
		work := flow.queue[len(flow.queue)-1]
		flow.queue = flow.queue[:len(flow.queue)-1]
		delta := flow.blocks[work].pending
		flow.blocks[work].pending = 0
		for _, bid := range flow.preds[flow.offsets[work]:flow.offsets[work+1]] {
			block := &flow.blocks[bid]
			fresh := delta &^ block.out
			if fresh == 0 {
				continue
			}
			flow.touch(bid)
			block.out |= fresh
			flow.addInput(bid, fresh)
		}
	}
}

func (s *stackAllocState) computeLiveWords() {
	if len(s.live.ids) == 0 {
		return
	}
	flow := newStackLiveFlow(s.f)
	for start := 0; start < len(s.live.ids); start += 64 {
		end := start + 64
		if end > len(s.live.ids) {
			end = len(s.live.ids)
		}
		s.seedLiveWord(&flow, s.live.ids[start:end])
		flow.propagate()
		for _, bid := range flow.touched {
			if out := flow.blocks[bid].out; out != 0 {
				s.live.words[bid] = append(s.live.words[bid], stackLiveWord{index: start / 64, bits: out})
			}
			flow.blocks[bid] = stackLiveBlock{}
		}
		flow.touched = flow.touched[:0]
	}
}
