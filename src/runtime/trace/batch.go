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

type batch struct {
	time timestamp
	gen  uint64
	data []byte
}

// readBatch copies b and parses the trace batch header inside.
// Returns the batch, bytes read, and an error.
func readBatch(b []byte) (batch, uint64, error) {
	if len(b) == 0 {
		return batch{}, 0, fmt.Errorf("batch is empty")
	}
	data := make([]byte, len(b))
	copy(data, b)

	// Read batch header byte.
	if typ := tracev2.EventType(b[0]); typ == tracev2.EvEndOfGeneration {
		if len(b) != 1 {
			return batch{}, 1, fmt.Errorf("unexpected end of generation in batch of size >1")
		}
		return batch{data: data}, 1, nil
	}
	if typ := tracev2.EventType(b[0]); typ != tracev2.EvEventBatch && typ != tracev2.EvExperimentalBatch {
		return batch{}, 1, fmt.Errorf("expected batch event, got event %d", typ)
	}
	total := 1
	b = b[1:]

	// Read the generation
	gen, n, err := readUvarint(b)
	if err != nil {
		return batch{}, uint64(total + n), fmt.Errorf("error reading batch gen: %w", err)
	}
	total += n
	b = b[n:]

	// Read the M (discard it).
	_, n, err = readUvarint(b)
	if err != nil {
		return batch{}, uint64(total + n), fmt.Errorf("error reading batch M ID: %w", err)
	}
	total += n
	b = b[n:]

	// Read the timestamp.
	ts, n, err := readUvarint(b)
	if err != nil {
		return batch{}, uint64(total + n), fmt.Errorf("error reading batch timestamp: %w", err)
	}
	total += n
	b = b[n:]

	// Read the size of the batch to follow.
	size, n, err := readUvarint(b)
	if err != nil {
		return batch{}, uint64(total + n), fmt.Errorf("error reading batch size: %w", err)
	}
	if size > tracev2.MaxBatchSize {
		return batch{}, uint64(total + n), fmt.Errorf("invalid batch size %d, maximum is %d", size, tracev2.MaxBatchSize)
	}
	total += n
	total += int(size)
	if total != len(data) {
		return batch{}, uint64(total), fmt.Errorf("expected complete batch")
	}
	data = data[:total]

	// Return the batch.
	return batch{
		gen:  gen,
		time: timestamp(ts),
		data: data,
	}, uint64(total), nil
}
