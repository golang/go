// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package traceparser

import (
	"bytes"
	"fmt"
	"io"
)

// scan the trace file finding the header, starts of batches, and the trailer.
// the trailer contains strings, stacks, and the clock frequency

// There are two ways of thinking about the raw trace file. It starts with a 16
// byte header "go 1.11 trace\0\0\0"
// From the point of
// view of the runtime, there is a collection of initializations for each goroutine.
// These consist of an EvGoCreate, possibly followed by one of EvGoWaiting or
// EvGoInSyscall if the go routine is waiting or in a syscall.
// Then there is an EvProcStart for the first running goroutine, so there's a running P,
// and then an EvGoStart for the first running goroutine. Then as the program runs, the
// runtime emits trace events. Finally when the tracing stops, the runtime emits a footer
// consisting of an EvFrequency (to convert ticks to nanoseconds) and some EvTimerGoroutines,
// followed by EvStacks for all the stack frames.
//
// In the file, the header and footer are as described, but all the events in between come
// in batches headed by EvBatch with the same P, and have to be rearranged into timestamp order.

// New() scans once through the file to find the beginnings of all the batches (EvBatch) and
// processes the footer extracting the strings and stacks.
// Parse() finds the batches that overlap the desired time interval, and processes them into
// events, dropping those outside the desired time interval. But it has to derive the missing
// initializations from the events it sees, as it has no other access to the state of the runtime.
// This is done in robust.go.

// In more detail, scanFile() is called by commonInit() which is called by either New() or ParseBuffer().
// It extracts the strings, the stacks, and remembers the locations of the Batches (all saved in *Parsed).

// Parse first computes the rawEvents for the batches that overlap the requested interval.
// It then calls createEvents() (events.go) which produces Events.

func (p *Parsed) parseHeader() error {
	p.r.Seek(0, 0)
	var buf [16]byte
	n, err := p.r.Read(buf[:])
	if n != 16 || err != nil {
		return fmt.Errorf("failed to red header: read %d bytes, not 16 %v", n, err)
	}
	// by hand. there are only 6 or so legitimate values; we could search for a match
	if buf[0] != 'g' || buf[1] != 'o' || buf[2] != ' ' ||
		buf[3] < '1' || buf[3] > '9' ||
		buf[4] != '.' ||
		buf[5] < '1' || buf[5] > '9' {
		return fmt.Errorf("not a trace file")
	}
	ver := int(buf[5] - '0')
	i := 0
	for ; buf[6+i] >= '0' && buf[6+i] <= '9' && i < 2; i++ {
		ver = ver*10 + int(buf[6+i]-'0')
	}
	ver += int(buf[3]-'0') * 1000
	if !bytes.Equal(buf[6+i:], []byte(" trace\x00\x00\x00\x00")[:10-i]) {
		return fmt.Errorf("not a trace file")
	}
	p.Version = ver
	// PJW: reject 1005 and 1007? They need symbolization, which we don't do.
	// Further, doing these would require 1.7 or earlier binaries.
	switch ver {
	case 1005, 1007:
		break // no longer supported
	case 1008, 1009:
		return nil
	case 1010, 1011:
		return nil
	}
	return fmt.Errorf("%d unsupported version", ver)
}

func (p *Parsed) scanFile() error {
	r := p.r
	// fill in the following values for sure
	strings := make(map[uint64]string)
	p.Strings = strings // ok to save maps immediately
	timerGoIDs := make(map[uint64]bool)
	p.timerGoids = timerGoIDs
	stacks := make(map[uint32][]*Frame)
	framer := make(map[Frame]*Frame) // uniqify *Frame
	p.Stacks = stacks
	footerLoc := 0

	var buf [1]byte
	off := 16 // skip the header
	n, err := r.Seek(int64(off), 0)
	if err != nil || n != int64(off) {
		return fmt.Errorf("Seek to %d got %d, err=%v", off, n, err)
	}
	var batchts int64 // from preceding batch
	var lastEv byte
	for {
		off0 := off
		n, err := r.Read(buf[:1])
		if err == io.EOF {
			break
		} else if err != nil || n != 1 {
			return fmt.Errorf("read failed at 0x%x, n=%d, %v",
				off, n, err)
		}
		off += n
		typ := buf[0] << 2 >> 2
		if typ == EvNone || typ >= EvCount ||
			EventDescriptions[typ].MinVersion > p.Version {
			err = fmt.Errorf("unknown event type %v at offset 0x%x, pass 1", typ, off0)
			return err
		}
		// extract and save the strings
		if typ == EvString {
			// String dictionary entry [ID, length, string].
			var id uint64
			id, off, err = readVal(r, off)
			if err != nil {
				return err
			}
			if id == 0 {
				err = fmt.Errorf("string at offset %d has invalid id 0", off)
				return err
			}
			if strings[id] != "" {
				err = fmt.Errorf("string at offset %d has duplicate id %v", off, id)
				return err
			}
			var ln uint64
			ln, off, err = readVal(r, off)
			if err != nil {
				return err
			}
			if ln == 0 {
				err = fmt.Errorf("string at offset %d has invalid length 0", off)
				return err
			}
			if ln > 1e6 {
				err = fmt.Errorf("string at offset %d has too large length %v", off, ln)
				return err
			}
			buf := make([]byte, ln)
			var n int
			n, err = io.ReadFull(r, buf)
			if err != nil {
				err = fmt.Errorf("failed to read trace at offset %d: read %v, want %v, error %v", off, n, ln, err)
				return err
			}
			off += n
			strings[id] = string(buf)
			lastEv = EvString
			continue
		}
		p.Count++
		if typ == EvFrequency {
			// found footer, remember location, save value
			footerLoc = off0
		}
		var args []uint64
		off, args, err = p.argsAt(off0, typ)
		if err != nil {
			err = fmt.Errorf("argsAt error %v; off=%d off0=%d %s",
				err, off, off0, evname(typ))
			return err
		}
		r.Seek(int64(off), 0)
		if typ == EvUserLog {
			_, off, err = readStr(r, off)
			if err != nil {
				return err
			}
		}
		if len(args) == 0 { // can't happen in well-formed trace file
			return fmt.Errorf("len(args)==0 off=0x%x typ=%s", off, evname(typ))
		}
		switch typ {
		case EvBatch:
			if footerLoc == 0 {
				// EvBatch in footer is just to have a header for stacks
				locp := int64(args[0])
				p.batches = append(p.batches,
					batch{Off: off0, P: locp, Cycles: int64(args[1])})
				// at this point we know when the previous batch ended!!
				batchts = int64(args[1])
				if batchts > p.maxticks {
					p.maxticks = batchts
				}
			}
		case EvFrequency:
			p.TicksPerSec = int64(args[0])
		case EvTimerGoroutine:
			timerGoIDs[args[0]] = true
		case EvStack:
			if len(args) < 2 {
				return fmt.Errorf("EvStack has too few args %d at 0x%x",
					len(args), off0)
			}
			size := args[1]
			if size > 1000 {
				return fmt.Errorf("EvStack has %d frames at 0x%x",
					size, off0)
			}
			want := 2 + 4*size
			if uint64(len(args)) != want {
				return fmt.Errorf("EvStack wants %d args, got %d, at 0x%x",
					len(args), want, off0)
			}
			id := args[0]
			if id != 0 && size > 0 {
				stk := make([]*Frame, size)
				for i := 0; i < int(size); i++ {
					pc := args[2+i*4+0]
					fn := args[2+i*4+1]
					file := args[2+i*4+2]
					line := args[2+i*4+3]
					stk[i] = &Frame{PC: pc, Fn: strings[fn], File: strings[file], Line: int(line)}
					if _, ok := framer[*stk[i]]; !ok {
						framer[*stk[i]] = stk[i]
					}
					stk[i] = framer[*stk[i]]
				}
				stacks[uint32(id)] = stk
			}
		default:
			if lastEv == EvBatch {
				// p.MinTsVal is set by the first real event, not the first EvBatch
				x := batchts + int64(args[0])
				if x < p.minticks {
					p.minticks = x
				}
			}
			batchts += int64(args[0])
			if batchts > p.maxticks {
				p.maxticks = batchts
			}
		}
		lastEv = typ
	}
	if footerLoc <= 0 {
		return fmt.Errorf("malformed trace file, no EvFrequency")
	}
	return nil
}
