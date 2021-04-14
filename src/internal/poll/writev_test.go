// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package poll_test

import (
	"internal/poll"
	"reflect"
	"testing"
)

func TestConsume(t *testing.T) {
	tests := []struct {
		in      [][]byte
		consume int64
		want    [][]byte
	}{
		{
			in:      [][]byte{[]byte("foo"), []byte("bar")},
			consume: 0,
			want:    [][]byte{[]byte("foo"), []byte("bar")},
		},
		{
			in:      [][]byte{[]byte("foo"), []byte("bar")},
			consume: 2,
			want:    [][]byte{[]byte("o"), []byte("bar")},
		},
		{
			in:      [][]byte{[]byte("foo"), []byte("bar")},
			consume: 3,
			want:    [][]byte{[]byte("bar")},
		},
		{
			in:      [][]byte{[]byte("foo"), []byte("bar")},
			consume: 4,
			want:    [][]byte{[]byte("ar")},
		},
		{
			in:      [][]byte{nil, nil, nil, []byte("bar")},
			consume: 1,
			want:    [][]byte{[]byte("ar")},
		},
		{
			in:      [][]byte{nil, nil, nil, []byte("foo")},
			consume: 0,
			want:    [][]byte{[]byte("foo")},
		},
		{
			in:      [][]byte{nil, nil, nil},
			consume: 0,
			want:    [][]byte{},
		},
	}
	for i, tt := range tests {
		in := tt.in
		poll.Consume(&in, tt.consume)
		if !reflect.DeepEqual(in, tt.want) {
			t.Errorf("%d. after consume(%d) = %+v, want %+v", i, tt.consume, in, tt.want)
		}
	}
}
