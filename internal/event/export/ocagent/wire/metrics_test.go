// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package wire

import (
	"reflect"
	"testing"
)

func TestMarshalJSON(t *testing.T) {
	tests := []struct {
		name string
		pt   *Point
		want string
	}{
		{
			"PointInt64",
			&Point{
				Value: PointInt64Value{
					Int64Value: 5,
				},
			},
			`{"int64Value":5}`,
		},
		{
			"PointDouble",
			&Point{
				Value: PointDoubleValue{
					DoubleValue: 3.14,
				},
			},
			`{"doubleValue":3.14}`,
		},
		{
			"PointDistribution",
			&Point{
				Value: PointDistributionValue{
					DistributionValue: &DistributionValue{
						Count: 3,
						Sum:   10,
						Buckets: []*Bucket{
							{
								Count: 1,
							},
							{
								Count: 2,
							},
						},
						BucketOptions: &BucketOptionsExplicit{
							Bounds: []float64{
								0, 5,
							},
						},
					},
				},
			},
			`{"distributionValue":{"count":3,"sum":10,"bucket_options":{"explicit":{"bounds":[0,5]}},"buckets":[{"count":1},{"count":2}]}}`,
		},
		{
			"nil point",
			nil,
			`null`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			buf, err := tt.pt.MarshalJSON()
			if err != nil {
				t.Fatalf("Got:\n%v\nWant:\n%v", err, nil)
			}
			got := string(buf)
			if !reflect.DeepEqual(got, tt.want) {
				t.Fatalf("Got:\n%s\nWant:\n%s", got, tt.want)
			}
		})
	}
}
