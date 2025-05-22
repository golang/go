// compile

// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

type Backend interface {
	Hash(ignores func(bucketName, keyName []byte) bool) (uint32, error)
}

type backend struct {
}

func first() (key []byte, value []byte) {
	return
}

func (b *backend) View(fn func() error) error {
	return nil
}

func (b *backend) Hash(ignores func(bucketName, keyName []byte) bool) (uint32, error) {
	err := b.View(func() error {
		for next, _ := first(); next != nil; next, _ = first() {
			_ = next
		}
		return nil
	})
	return 0, err
}

func defragdb() error {
	for next, _ := first(); next != nil; next, _ = first() {
		_ = f(next)
		ForEach(func(k, v []byte) error {
			_ = next
			return nil
		})
	}

	return nil
}

func ForEach(fn func(k, v []byte) error) error {
	for k, v := first(); k != nil; k, v = first() {
		if err := fn(k, v); err != nil {
			return err
		}
	}
	return nil
}

//go:noinline
func f(any) string {
	return ""
}
