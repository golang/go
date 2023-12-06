// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package profile

import (
	"testing"
)

func TestParseContention(t *testing.T) {
	tests := []struct {
		name    string
		in      string
		wantErr bool
	}{
		{
			name: "valid",
			in: `--- mutex:
cycles/second=3491920901
sampling period=1
43227965305 1659640 @ 0x45e851 0x45f764 0x4a2be1 0x44ea31
34035731690 15760 @ 0x45e851 0x45f764 0x4a2b17 0x44ea31
`,
		},
		{
			name: "valid with comment",
			in: `--- mutex:
cycles/second=3491920901
sampling period=1
43227965305 1659640 @ 0x45e851 0x45f764 0x4a2be1 0x44ea31
#	0x45e850	sync.(*Mutex).Unlock+0x80	/go/src/sync/mutex.go:126
#	0x45f763	sync.(*RWMutex).Unlock+0x83	/go/src/sync/rwmutex.go:125
#	0x4a2be0	main.main.func3+0x70		/go/src/internal/pprof/profile/a_binary.go:58

34035731690 15760 @ 0x45e851 0x45f764 0x4a2b17 0x44ea31
#	0x45e850	sync.(*Mutex).Unlock+0x80	/go/src/sync/mutex.go:126
#	0x45f763	sync.(*RWMutex).Unlock+0x83	/go/src/sync/rwmutex.go:125
#	0x4a2b16	main.main.func2+0xd6		/go/src/internal/pprof/profile/a_binary.go:48
`,
		},
		{
			name:    "empty",
			in:      `--- mutex:`,
			wantErr: true,
		},
		{
			name: "invalid header",
			in: `--- channel:
43227965305 1659640 @ 0x45e851 0x45f764 0x4a2be1 0x44ea31`,
			wantErr: true,
		},
	}
	for _, tc := range tests {
		_, err := parseContention([]byte(tc.in))
		if tc.wantErr && err == nil {
			t.Errorf("parseContention(%q) succeeded unexpectedly", tc.name)
		}
		if !tc.wantErr && err != nil {
			t.Errorf("parseContention(%q) failed unexpectedly: %v", tc.name, err)
		}
	}

}
