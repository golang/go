// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rt

import (
	"context"
	"errors"
	"iter"
	"sync"
	"sync/atomic"
	"testing"
)

func benchConcMapErrShape(input []int, limit int, f func(int) (int, error)) ([]int, error) {
	results := make([]int, len(input))
	var next atomic.Int64
	var wg sync.WaitGroup
	var mu sync.Mutex
	var errs []error
	for range min(limit, len(input)) {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for {
				i := int(next.Add(1) - 1)
				if i >= len(input) {
					return
				}
				value, err := f(input[i])
				results[i] = value
				if err != nil {
					mu.Lock()
					errs = append(errs, err)
					mu.Unlock()
				}
			}
		}()
	}
	wg.Wait()
	return results, errors.Join(errs...)
}

func BenchmarkConcMapErrShape(b *testing.B) {
	input := make([]int, benchElements)
	for i := range input {
		input[i] = i
	}
	benchStage(b, func(b *testing.B, limit int, f func(context.Context, int) (int, bool, error)) iter.Seq2[int, error] {
		return func(yield func(int, error) bool) {
			results, err := benchConcMapErrShape(input, limit, func(i int) (int, error) {
				value, _, err := f(context.Background(), i)
				return value, err
			})
			if err != nil {
				yield(0, err)
				return
			}
			for _, value := range results {
				if !yield(value, nil) {
					return
				}
			}
		}
	})
}
