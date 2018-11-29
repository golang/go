// Copyright 2018 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package profile

import (
	"testing"
)

func TestMapMapping(t *testing.T) {
	pm := &profileMerger{
		p:            &Profile{},
		mappings:     make(map[mappingKey]*Mapping),
		mappingsByID: make(map[uint64]mapInfo),
	}
	for _, tc := range []struct {
		desc       string
		m1         Mapping
		m2         Mapping
		wantMerged bool
	}{
		{
			desc: "same file name",
			m1: Mapping{
				ID:   1,
				File: "test-file-1",
			},
			m2: Mapping{
				ID:   2,
				File: "test-file-1",
			},
			wantMerged: true,
		},
		{
			desc: "same build ID",
			m1: Mapping{
				ID:      3,
				BuildID: "test-build-id-1",
			},
			m2: Mapping{
				ID:      4,
				BuildID: "test-build-id-1",
			},
			wantMerged: true,
		},
		{
			desc: "same fake mapping",
			m1: Mapping{
				ID: 5,
			},
			m2: Mapping{
				ID: 6,
			},
			wantMerged: true,
		},
		{
			desc: "different start",
			m1: Mapping{
				ID:      7,
				Start:   0x1000,
				Limit:   0x2000,
				BuildID: "test-build-id-2",
			},
			m2: Mapping{
				ID:      8,
				Start:   0x3000,
				Limit:   0x4000,
				BuildID: "test-build-id-2",
			},
			wantMerged: true,
		},
		{
			desc: "different file name",
			m1: Mapping{
				ID:   9,
				File: "test-file-2",
			},
			m2: Mapping{
				ID:   10,
				File: "test-file-3",
			},
		},
		{
			desc: "different build id",
			m1: Mapping{
				ID:      11,
				BuildID: "test-build-id-3",
			},
			m2: Mapping{
				ID:      12,
				BuildID: "test-build-id-4",
			},
		},
		{
			desc: "different size",
			m1: Mapping{
				ID:      13,
				Start:   0x1000,
				Limit:   0x3000,
				BuildID: "test-build-id-5",
			},
			m2: Mapping{
				ID:      14,
				Start:   0x1000,
				Limit:   0x5000,
				BuildID: "test-build-id-5",
			},
		},
		{
			desc: "different offset",
			m1: Mapping{
				ID:      15,
				Offset:  1,
				BuildID: "test-build-id-6",
			},
			m2: Mapping{
				ID:      16,
				Offset:  2,
				BuildID: "test-build-id-6",
			},
		},
	} {
		t.Run(tc.desc, func(t *testing.T) {
			info1 := pm.mapMapping(&tc.m1)
			info2 := pm.mapMapping(&tc.m2)
			gotM1, gotM2 := *info1.m, *info2.m

			wantM1 := tc.m1
			wantM1.ID = gotM1.ID
			if gotM1 != wantM1 {
				t.Errorf("first mapping got %v, want %v", gotM1, wantM1)
			}

			if tc.wantMerged {
				if gotM1 != gotM2 {
					t.Errorf("first mapping got %v, second mapping got %v, want equal", gotM1, gotM2)
				}
				if info1.offset != 0 {
					t.Errorf("first mapping info got offset %d, want 0", info1.offset)
				}
				if wantOffset := int64(tc.m1.Start) - int64(tc.m2.Start); wantOffset != info2.offset {
					t.Errorf("second mapping info got offset %d, want %d", info2.offset, wantOffset)
				}
			} else {
				if gotM1.ID == gotM2.ID {
					t.Errorf("first mapping got %v, second mapping got %v, want different IDs", gotM1, gotM2)
				}
				wantM2 := tc.m2
				wantM2.ID = gotM2.ID
				if gotM2 != wantM2 {
					t.Errorf("second mapping got %v, want %v", gotM2, wantM2)
				}
			}
		})
	}
}
