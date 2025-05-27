// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package trace

import (
	"fmt"
	"internal/trace/tracev2"
)

// timestamp is an unprocessed timestamp.
type timestamp uint64

// batch represents a batch of trace events.
// It is unparsed except for its header.
type batch struct {
	m    threadID
	time timestamp
	data []byte
}

// threadID is the runtime-internal M structure's ID. This is unique
// for each OS thread.
type threadID int64

// readBatch copies b and parses the trace batch header inside.
// Returns the batch, the generation, bytes read, and an error.
func readBatch(b []byte) (batch, uint64, uint64, error) {
	if len(b) == 0 {
		return batch{}, 0, 0, fmt.Errorf("batch is empty")
	}
	data := make([]byte, len(b))
	if nw := copy(data, b); nw != len(b) {
		return batch{}, 0, 0, fmt.Errorf("unexpected error copying batch")
	}
	// Read batch header byte.
	if typ := tracev2.EventType(b[0]); typ != tracev2.EvEventBatch && typ != tracev2.EvExperimentalBatch {
		return batch{}, 0, 1, fmt.Errorf("expected batch event, got event %d", typ)
	}

	// Read the batch header: gen (generation), thread (M) ID, base timestamp
	// for the batch.
	total := 1
	b = b[1:]
	gen, n, err := readUvarint(b)
	if err != nil {
		return batch{}, gen, uint64(total + n), fmt.Errorf("error reading batch gen: %w", err)
	}
	total += n
	b = b[n:]
	m, n, err := readUvarint(b)
	if err != nil {
		return batch{}, gen, uint64(total + n), fmt.Errorf("error reading batch M ID: %w", err)
	}
	total += n
	b = b[n:]
	ts, n, err := readUvarint(b)
	if err != nil {
		return batch{}, gen, uint64(total + n), fmt.Errorf("error reading batch timestamp: %w", err)
	}
	total += n
	b = b[n:]

	// Read in the size of the batch to follow.
	size, n, err := readUvarint(b)
	if err != nil {
		return batch{}, gen, uint64(total + n), fmt.Errorf("error reading batch size: %w", err)
	}
	if size > tracev2.MaxBatchSize {
		return batch{}, gen, uint64(total + n), fmt.Errorf("invalid batch size %d, maximum is %d", size, tracev2.MaxBatchSize)
	}
	total += n
	total += int(size)
	data = data[:total]

	// Return the batch.
	return batch{
		m:    threadID(m),
		time: timestamp(ts),
		data: data,
	}, gen, uint64(total), nil
}
