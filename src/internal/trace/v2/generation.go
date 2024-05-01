// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package trace

import (
	"bufio"
	"bytes"
	"cmp"
	"encoding/binary"
	"fmt"
	"io"
	"slices"
	"strings"

	"internal/trace/v2/event"
	"internal/trace/v2/event/go122"
)

// generation contains all the trace data for a single
// trace generation. It is purely data: it does not
// track any parse state nor does it contain a cursor
// into the generation.
type generation struct {
	gen        uint64
	batches    map[ThreadID][]batch
	cpuSamples []cpuSample
	*evTable
}

// spilledBatch represents a batch that was read out for the next generation,
// while reading the previous one. It's passed on when parsing the next
// generation.
type spilledBatch struct {
	gen uint64
	*batch
}

// readGeneration buffers and decodes the structural elements of a trace generation
// out of r. spill is the first batch of the new generation (already buffered and
// parsed from reading the last generation). Returns the generation and the first
// batch read of the next generation, if any.
func readGeneration(r *bufio.Reader, spill *spilledBatch) (*generation, *spilledBatch, error) {
	g := &generation{
		evTable: &evTable{
			pcs: make(map[uint64]frame),
		},
		batches: make(map[ThreadID][]batch),
	}
	// Process the spilled batch.
	if spill != nil {
		g.gen = spill.gen
		if err := processBatch(g, *spill.batch); err != nil {
			return nil, nil, err
		}
		spill = nil
	}
	// Read batches one at a time until we either hit EOF or
	// the next generation.
	for {
		b, gen, err := readBatch(r)
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, nil, err
		}
		if gen == 0 {
			// 0 is a sentinel used by the runtime, so we'll never see it.
			return nil, nil, fmt.Errorf("invalid generation number %d", gen)
		}
		if g.gen == 0 {
			// Initialize gen.
			g.gen = gen
		}
		if gen == g.gen+1 { // TODO: advance this the same way the runtime does.
			spill = &spilledBatch{gen: gen, batch: &b}
			break
		}
		if gen != g.gen {
			// N.B. Fail as fast as possible if we see this. At first it
			// may seem prudent to be fault-tolerant and assume we have a
			// complete generation, parsing and returning that first. However,
			// if the batches are mixed across generations then it's likely
			// we won't be able to parse this generation correctly at all.
			// Rather than return a cryptic error in that case, indicate the
			// problem as soon as we see it.
			return nil, nil, fmt.Errorf("generations out of order")
		}
		if err := processBatch(g, b); err != nil {
			return nil, nil, err
		}
	}

	// Check some invariants.
	if g.freq == 0 {
		return nil, nil, fmt.Errorf("no frequency event found")
	}
	// N.B. Trust that the batch order is correct. We can't validate the batch order
	// by timestamp because the timestamps could just be plain wrong. The source of
	// truth is the order things appear in the trace and the partial order sequence
	// numbers on certain events. If it turns out the batch order is actually incorrect
	// we'll very likely fail to advance a partial order from the frontier.

	// Compactify stacks and strings for better lookup performance later.
	g.stacks.compactify()
	g.strings.compactify()

	// Validate stacks.
	if err := validateStackStrings(&g.stacks, &g.strings, g.pcs); err != nil {
		return nil, nil, err
	}

	// Fix up the CPU sample timestamps, now that we have freq.
	for i := range g.cpuSamples {
		s := &g.cpuSamples[i]
		s.time = g.freq.mul(timestamp(s.time))
	}
	// Sort the CPU samples.
	slices.SortFunc(g.cpuSamples, func(a, b cpuSample) int {
		return cmp.Compare(a.time, b.time)
	})
	return g, spill, nil
}

// processBatch adds the batch to the generation.
func processBatch(g *generation, b batch) error {
	switch {
	case b.isStringsBatch():
		if err := addStrings(&g.strings, b); err != nil {
			return err
		}
	case b.isStacksBatch():
		if err := addStacks(&g.stacks, g.pcs, b); err != nil {
			return err
		}
	case b.isCPUSamplesBatch():
		samples, err := addCPUSamples(g.cpuSamples, b)
		if err != nil {
			return err
		}
		g.cpuSamples = samples
	case b.isFreqBatch():
		freq, err := parseFreq(b)
		if err != nil {
			return err
		}
		if g.freq != 0 {
			return fmt.Errorf("found multiple frequency events")
		}
		g.freq = freq
	default:
		g.batches[b.m] = append(g.batches[b.m], b)
	}
	return nil
}

// validateStackStrings makes sure all the string references in
// the stack table are present in the string table.
func validateStackStrings(
	stacks *dataTable[stackID, stack],
	strings *dataTable[stringID, string],
	frames map[uint64]frame,
) error {
	var err error
	stacks.forEach(func(id stackID, stk stack) bool {
		for _, pc := range stk.pcs {
			frame, ok := frames[pc]
			if !ok {
				err = fmt.Errorf("found unknown pc %x for stack %d", pc, id)
				return false
			}
			_, ok = strings.get(frame.funcID)
			if !ok {
				err = fmt.Errorf("found invalid func string ID %d for stack %d", frame.funcID, id)
				return false
			}
			_, ok = strings.get(frame.fileID)
			if !ok {
				err = fmt.Errorf("found invalid file string ID %d for stack %d", frame.fileID, id)
				return false
			}
		}
		return true
	})
	return err
}

// addStrings takes a batch whose first byte is an EvStrings event
// (indicating that the batch contains only strings) and adds each
// string contained therein to the provided strings map.
func addStrings(stringTable *dataTable[stringID, string], b batch) error {
	if !b.isStringsBatch() {
		return fmt.Errorf("internal error: addStrings called on non-string batch")
	}
	r := bytes.NewReader(b.data)
	hdr, err := r.ReadByte() // Consume the EvStrings byte.
	if err != nil || event.Type(hdr) != go122.EvStrings {
		return fmt.Errorf("missing strings batch header")
	}

	var sb strings.Builder
	for r.Len() != 0 {
		// Read the header.
		ev, err := r.ReadByte()
		if err != nil {
			return err
		}
		if event.Type(ev) != go122.EvString {
			return fmt.Errorf("expected string event, got %d", ev)
		}

		// Read the string's ID.
		id, err := binary.ReadUvarint(r)
		if err != nil {
			return err
		}

		// Read the string's length.
		len, err := binary.ReadUvarint(r)
		if err != nil {
			return err
		}
		if len > go122.MaxStringSize {
			return fmt.Errorf("invalid string size %d, maximum is %d", len, go122.MaxStringSize)
		}

		// Copy out the string.
		n, err := io.CopyN(&sb, r, int64(len))
		if n != int64(len) {
			return fmt.Errorf("failed to read full string: read %d but wanted %d", n, len)
		}
		if err != nil {
			return fmt.Errorf("copying string data: %w", err)
		}

		// Add the string to the map.
		s := sb.String()
		sb.Reset()
		if err := stringTable.insert(stringID(id), s); err != nil {
			return err
		}
	}
	return nil
}

// addStacks takes a batch whose first byte is an EvStacks event
// (indicating that the batch contains only stacks) and adds each
// string contained therein to the provided stacks map.
func addStacks(stackTable *dataTable[stackID, stack], pcs map[uint64]frame, b batch) error {
	if !b.isStacksBatch() {
		return fmt.Errorf("internal error: addStacks called on non-stacks batch")
	}
	r := bytes.NewReader(b.data)
	hdr, err := r.ReadByte() // Consume the EvStacks byte.
	if err != nil || event.Type(hdr) != go122.EvStacks {
		return fmt.Errorf("missing stacks batch header")
	}

	for r.Len() != 0 {
		// Read the header.
		ev, err := r.ReadByte()
		if err != nil {
			return err
		}
		if event.Type(ev) != go122.EvStack {
			return fmt.Errorf("expected stack event, got %d", ev)
		}

		// Read the stack's ID.
		id, err := binary.ReadUvarint(r)
		if err != nil {
			return err
		}

		// Read how many frames are in each stack.
		nFrames, err := binary.ReadUvarint(r)
		if err != nil {
			return err
		}
		if nFrames > go122.MaxFramesPerStack {
			return fmt.Errorf("invalid stack size %d, maximum is %d", nFrames, go122.MaxFramesPerStack)
		}

		// Each frame consists of 4 fields: pc, funcID (string), fileID (string), line.
		frames := make([]uint64, 0, nFrames)
		for i := uint64(0); i < nFrames; i++ {
			// Read the frame data.
			pc, err := binary.ReadUvarint(r)
			if err != nil {
				return fmt.Errorf("reading frame %d's PC for stack %d: %w", i+1, id, err)
			}
			funcID, err := binary.ReadUvarint(r)
			if err != nil {
				return fmt.Errorf("reading frame %d's funcID for stack %d: %w", i+1, id, err)
			}
			fileID, err := binary.ReadUvarint(r)
			if err != nil {
				return fmt.Errorf("reading frame %d's fileID for stack %d: %w", i+1, id, err)
			}
			line, err := binary.ReadUvarint(r)
			if err != nil {
				return fmt.Errorf("reading frame %d's line for stack %d: %w", i+1, id, err)
			}
			frames = append(frames, pc)

			if _, ok := pcs[pc]; !ok {
				pcs[pc] = frame{
					pc:     pc,
					funcID: stringID(funcID),
					fileID: stringID(fileID),
					line:   line,
				}
			}
		}

		// Add the stack to the map.
		if err := stackTable.insert(stackID(id), stack{pcs: frames}); err != nil {
			return err
		}
	}
	return nil
}

// addCPUSamples takes a batch whose first byte is an EvCPUSamples event
// (indicating that the batch contains only CPU samples) and adds each
// sample contained therein to the provided samples list.
func addCPUSamples(samples []cpuSample, b batch) ([]cpuSample, error) {
	if !b.isCPUSamplesBatch() {
		return nil, fmt.Errorf("internal error: addCPUSamples called on non-CPU-sample batch")
	}
	r := bytes.NewReader(b.data)
	hdr, err := r.ReadByte() // Consume the EvCPUSamples byte.
	if err != nil || event.Type(hdr) != go122.EvCPUSamples {
		return nil, fmt.Errorf("missing CPU samples batch header")
	}

	for r.Len() != 0 {
		// Read the header.
		ev, err := r.ReadByte()
		if err != nil {
			return nil, err
		}
		if event.Type(ev) != go122.EvCPUSample {
			return nil, fmt.Errorf("expected CPU sample event, got %d", ev)
		}

		// Read the sample's timestamp.
		ts, err := binary.ReadUvarint(r)
		if err != nil {
			return nil, err
		}

		// Read the sample's M.
		m, err := binary.ReadUvarint(r)
		if err != nil {
			return nil, err
		}
		mid := ThreadID(m)

		// Read the sample's P.
		p, err := binary.ReadUvarint(r)
		if err != nil {
			return nil, err
		}
		pid := ProcID(p)

		// Read the sample's G.
		g, err := binary.ReadUvarint(r)
		if err != nil {
			return nil, err
		}
		goid := GoID(g)
		if g == 0 {
			goid = NoGoroutine
		}

		// Read the sample's stack.
		s, err := binary.ReadUvarint(r)
		if err != nil {
			return nil, err
		}

		// Add the sample to the slice.
		samples = append(samples, cpuSample{
			schedCtx: schedCtx{
				M: mid,
				P: pid,
				G: goid,
			},
			time:  Time(ts), // N.B. this is really a "timestamp," not a Time.
			stack: stackID(s),
		})
	}
	return samples, nil
}

// parseFreq parses out a lone EvFrequency from a batch.
func parseFreq(b batch) (frequency, error) {
	if !b.isFreqBatch() {
		return 0, fmt.Errorf("internal error: parseFreq called on non-frequency batch")
	}
	r := bytes.NewReader(b.data)
	r.ReadByte() // Consume the EvFrequency byte.

	// Read the frequency. It'll come out as timestamp units per second.
	f, err := binary.ReadUvarint(r)
	if err != nil {
		return 0, err
	}
	// Convert to nanoseconds per timestamp unit.
	return frequency(1.0 / (float64(f) / 1e9)), nil
}
