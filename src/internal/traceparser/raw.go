// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package traceparser

import (
	"encoding/binary"
	"fmt"
	"hash/fnv"
	"io"
	"log"
)

// convert batches into their raw events. For small intervals (1 or 10 seconds)
// this takes about 40% of the total Parse time.

func (p *Parsed) batchify(b *batch) error {
	evs := make([]rawEvent, 0)
	p.seenArgs = make(map[uint64]*[]uint64)
	hasher := fnv.New64()
	r := p.r
	r.Seek(int64(b.Off), 0)
	var buf [1]byte
	seenBatch := false // to terminate the loop on the second EvBatch

	for off := b.Off; ; {
		off0 := off // remember the beginning of the event
		n, err := r.Read(buf[:])
		if err != nil {
			return err
		}
		off += n
		typ := buf[0] << 2 >> 2 // event type is bottom 6 bits
		if typ == EvFrequency || (typ == EvBatch && seenBatch) {
			break // found trailer, or next batch
		}
		if typ == EvBatch {
			seenBatch = true
		}
		if typ == EvString {
			// skip over it. error checking was done in file.go
			_, off, _ = readVal(r, off)
			var ln uint64
			ln, off, _ = readVal(r, off)
			// PJW: why not just seek ahead ln bytes?
			if false {
				buf := make([]byte, ln)
				var n int
				n, _ = io.ReadFull(r, buf)
				off += n
			} else {
				n, _ := r.Seek(int64(ln), 1)
				off = int(n)
			}
			continue
		}
		// build the raw event and collect its arguments
		ev := rawEvent{typ: typ, off: uint32(off0 - b.Off)}
		var args []uint64
		off, args, err = p.argsAt(off0, typ)
		if err != nil {
			// PJW: make sure this is useful
			return fmt.Errorf("parsing %s failed at P=%d off=%d %v", evname(typ),
				b.P, off0, err)
		}

		// have we seen the args before?
		if len(args) > 0 {
			ev.arg0 = args[0]
			if len(args) > 1 {
				hasher.Reset()
				for i := 1; i < len(args); i++ {
					var x [8]byte
					binary.LittleEndian.PutUint64(x[:], args[i])
					_, err := hasher.Write(x[:])
					if err != nil {
						log.Fatal(err)
					}
				}
				hc := hasher.Sum64()
				old, ok := p.seenArgs[hc]
				if !ok {
					final := make([]uint64, len(args)-1)
					copy(final, args[1:])
					p.seenArgs[hc] = &final
				} else {
					// is this a collision? PJW: make this precisely right
					if len(*old) != len(args[1:]) {
						log.Fatalf("COLLISION old:%v this:%v", *old, args[1:])
					}
				}
				ev.args = p.seenArgs[hc]
			}
		}
		if typ == EvUserLog {
			// argsAt didn't read the string argument
			var s string
			s, off, err = readStr(r, off)
			ev.sarg = s
		}
		evs = append(evs, ev)
	}
	b.raws = evs
	return nil
}
