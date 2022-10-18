// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package exec

import (
	"reflect"
	"testing"
)

func TestDedupEnv(t *testing.T) {
	tests := []struct {
		noCase  bool
		in      []string
		want    []string
		wantErr bool
	}{
		{
			noCase: true,
			in:     []string{"k1=v1", "k2=v2", "K1=v3"},
			want:   []string{"k2=v2", "K1=v3"},
		},
		{
			noCase: false,
			in:     []string{"k1=v1", "K1=V2", "k1=v3"},
			want:   []string{"K1=V2", "k1=v3"},
		},
		{
			in:   []string{"=a", "=b", "foo", "bar"},
			want: []string{"=b", "foo", "bar"},
		},
		{
			// #49886: preserve weird Windows keys with leading "=" signs.
			noCase: true,
			in:     []string{`=C:=C:\golang`, `=D:=D:\tmp`, `=D:=D:\`},
			want:   []string{`=C:=C:\golang`, `=D:=D:\`},
		},
		{
			// #52436: preserve invalid key-value entries (for now).
			// (Maybe filter them out or error out on them at some point.)
			in:   []string{"dodgy", "entries"},
			want: []string{"dodgy", "entries"},
		},
		{
			// Filter out entries containing NULs.
			in:      []string{"A=a\x00b", "B=b", "C\x00C=c"},
			want:    []string{"B=b"},
			wantErr: true,
		},
	}
	for _, tt := range tests {
		got, err := dedupEnvCase(tt.noCase, tt.in)
		if !reflect.DeepEqual(got, tt.want) || (err != nil) != tt.wantErr {
			t.Errorf("Dedup(%v, %q) = %q, %v; want %q, error:%v", tt.noCase, tt.in, got, err, tt.want, tt.wantErr)
		}
	}
}
