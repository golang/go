package ocagent

import (
	"reflect"
	"testing"
	"time"

	"golang.org/x/tools/internal/telemetry"
	"golang.org/x/tools/internal/telemetry/export/ocagent/wire"
	"golang.org/x/tools/internal/telemetry/metric"
)

func TestDataToMetricDescriptor(t *testing.T) {
	tests := []struct {
		name string
		data telemetry.MetricData
		want *wire.MetricDescriptor
	}{
		{
			"nil data",
			nil,
			nil,
		},
		{
			"Int64Data gauge",
			&metric.Int64Data{
				Info: &metric.Scalar{
					Name:        "int",
					Description: "int metric",
					Keys:        []interface{}{"hello"},
				},
				IsGauge: true,
			},
			&wire.MetricDescriptor{
				Name:        "int",
				Description: "int metric",
				Type:        wire.MetricDescriptor_GAUGE_INT64,
				LabelKeys: []*wire.LabelKey{
					&wire.LabelKey{
						Key: "hello",
					},
				},
			},
		},
		{
			"Float64Data cumulative",
			&metric.Float64Data{
				Info: &metric.Scalar{
					Name:        "float",
					Description: "float metric",
					Keys:        []interface{}{"world"},
				},
				IsGauge: false,
			},
			&wire.MetricDescriptor{
				Name:        "float",
				Description: "float metric",
				Type:        wire.MetricDescriptor_CUMULATIVE_DOUBLE,
				LabelKeys: []*wire.LabelKey{
					&wire.LabelKey{
						Key: "world",
					},
				},
			},
		},
		{
			"HistogramInt64",
			&metric.HistogramInt64Data{
				Info: &metric.HistogramInt64{
					Name:        "histogram int",
					Description: "histogram int metric",
					Keys:        []interface{}{"hello"},
				},
			},
			&wire.MetricDescriptor{
				Name:        "histogram int",
				Description: "histogram int metric",
				Type:        wire.MetricDescriptor_CUMULATIVE_DISTRIBUTION,
				LabelKeys: []*wire.LabelKey{
					&wire.LabelKey{
						Key: "hello",
					},
				},
			},
		},
		{
			"HistogramFloat64",
			&metric.HistogramFloat64Data{
				Info: &metric.HistogramFloat64{
					Name:        "histogram float",
					Description: "histogram float metric",
					Keys:        []interface{}{"hello"},
				},
			},
			&wire.MetricDescriptor{
				Name:        "histogram float",
				Description: "histogram float metric",
				Type:        wire.MetricDescriptor_CUMULATIVE_DISTRIBUTION,
				LabelKeys: []*wire.LabelKey{
					&wire.LabelKey{
						Key: "hello",
					},
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := dataToMetricDescriptor(tt.data)
			if !reflect.DeepEqual(got, tt.want) {
				t.Fatalf("Got:\n%s\nWant:\n%s", marshaled(got), marshaled(tt.want))
			}
		})
	}
}

func TestGetDescription(t *testing.T) {
	tests := []struct {
		name string
		data telemetry.MetricData
		want string
	}{
		{
			"nil data",
			nil,
			"",
		},
		{
			"Int64Data description",
			&metric.Int64Data{
				Info: &metric.Scalar{
					Description: "int metric",
				},
			},
			"int metric",
		},
		{
			"Float64Data description",
			&metric.Float64Data{
				Info: &metric.Scalar{
					Description: "float metric",
				},
			},
			"float metric",
		},
		{
			"HistogramInt64Data description",
			&metric.HistogramInt64Data{
				Info: &metric.HistogramInt64{
					Description: "histogram int metric",
				},
			},
			"histogram int metric",
		},
		{
			"HistogramFloat64Data description",
			&metric.HistogramFloat64Data{
				Info: &metric.HistogramFloat64{
					Description: "histogram float metric",
				},
			},
			"histogram float metric",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := getDescription(tt.data)
			if !reflect.DeepEqual(got, tt.want) {
				t.Fatalf("Got:\n%s\nWant:\n%s", marshaled(got), marshaled(tt.want))
			}
		})
	}

}

func TestGetLabelKeys(t *testing.T) {
	tests := []struct {
		name string
		data telemetry.MetricData
		want []*wire.LabelKey
	}{
		{
			"nil label keys",
			nil,
			nil,
		},
		{
			"Int64Data label keys",
			&metric.Int64Data{
				Info: &metric.Scalar{
					Keys: []interface{}{
						"hello",
					},
				},
			},
			[]*wire.LabelKey{
				&wire.LabelKey{
					Key: "hello",
				},
			},
		},
		{
			"Float64Data label keys",
			&metric.Float64Data{
				Info: &metric.Scalar{
					Keys: []interface{}{
						"world",
					},
				},
			},
			[]*wire.LabelKey{
				&wire.LabelKey{
					Key: "world",
				},
			},
		},
		{
			"HistogramInt64Data label keys",
			&metric.HistogramInt64Data{
				Info: &metric.HistogramInt64{
					Keys: []interface{}{
						"hello",
					},
				},
			},
			[]*wire.LabelKey{
				&wire.LabelKey{
					Key: "hello",
				},
			},
		},
		{
			"HistogramFloat64Data label keys",
			&metric.HistogramFloat64Data{
				Info: &metric.HistogramFloat64{
					Keys: []interface{}{
						"hello",
					},
				},
			},
			[]*wire.LabelKey{
				&wire.LabelKey{
					Key: "hello",
				},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := getLabelKeys(tt.data)
			if !reflect.DeepEqual(got, tt.want) {
				t.Fatalf("Got:\n%s\nWant:\n%s", marshaled(got), marshaled(tt.want))
			}
		})
	}
}

func TestDataToMetricDescriptorType(t *testing.T) {
	tests := []struct {
		name string
		data telemetry.MetricData
		want wire.MetricDescriptor_Type
	}{
		{
			"Nil data",
			nil,
			wire.MetricDescriptor_UNSPECIFIED,
		},
		{
			"Gauge Int64",
			&metric.Int64Data{
				IsGauge: true,
			},
			wire.MetricDescriptor_GAUGE_INT64,
		},
		{
			"Cumulative Int64",
			&metric.Int64Data{
				IsGauge: false,
			},
			wire.MetricDescriptor_CUMULATIVE_INT64,
		},
		{
			"Gauge Float64",
			&metric.Float64Data{
				IsGauge: true,
			},
			wire.MetricDescriptor_GAUGE_DOUBLE,
		},
		{
			"Cumulative Float64",
			&metric.Float64Data{
				IsGauge: false,
			},
			wire.MetricDescriptor_CUMULATIVE_DOUBLE,
		},
		{
			"HistogramInt64",
			&metric.HistogramInt64Data{},
			wire.MetricDescriptor_CUMULATIVE_DISTRIBUTION,
		},
		{
			"HistogramFloat64",
			&metric.HistogramFloat64Data{},
			wire.MetricDescriptor_CUMULATIVE_DISTRIBUTION,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := dataToMetricDescriptorType(tt.data)
			if !reflect.DeepEqual(got, tt.want) {
				t.Fatalf("Got:\n%s\nWant:\n%s", marshaled(got), marshaled(tt.want))
			}
		})
	}
}

func TestDataToTimeseries(t *testing.T) {
	epoch := time.Unix(0, 0)
	epochTimestamp := convertTimestamp(epoch)

	end := time.Unix(30, 0)
	endTimestamp := convertTimestamp(end)

	tests := []struct {
		name  string
		data  telemetry.MetricData
		start time.Time
		want  []*wire.TimeSeries
	}{
		{
			"nil data",
			nil,
			time.Time{},
			nil,
		},
		{
			"Int64Data",
			&metric.Int64Data{
				Rows: []int64{
					1,
					2,
					3,
				},
				EndTime: &end,
			},
			epoch,
			[]*wire.TimeSeries{
				&wire.TimeSeries{
					Points: []*wire.Point{
						&wire.Point{
							Value:     wire.PointInt64Value{Int64Value: 1},
							Timestamp: &endTimestamp,
						},
					},
					StartTimestamp: &epochTimestamp,
				},
				&wire.TimeSeries{
					Points: []*wire.Point{
						&wire.Point{
							Value:     wire.PointInt64Value{Int64Value: 2},
							Timestamp: &endTimestamp,
						},
					},
					StartTimestamp: &epochTimestamp,
				},
				&wire.TimeSeries{
					Points: []*wire.Point{
						&wire.Point{
							Value:     wire.PointInt64Value{Int64Value: 3},
							Timestamp: &endTimestamp,
						},
					},
					StartTimestamp: &epochTimestamp,
				},
			},
		},
		{
			"Float64Data",
			&metric.Float64Data{
				Rows: []float64{
					1.5,
					4.5,
				},
				EndTime: &end,
			},
			epoch,
			[]*wire.TimeSeries{
				&wire.TimeSeries{
					Points: []*wire.Point{
						&wire.Point{
							Value:     wire.PointDoubleValue{DoubleValue: 1.5},
							Timestamp: &endTimestamp,
						},
					},
					StartTimestamp: &epochTimestamp,
				},
				&wire.TimeSeries{
					Points: []*wire.Point{
						&wire.Point{
							Value:     wire.PointDoubleValue{DoubleValue: 4.5},
							Timestamp: &endTimestamp,
						},
					},
					StartTimestamp: &epochTimestamp,
				},
			},
		},
		{
			"HistogramInt64Data",
			&metric.HistogramInt64Data{
				Rows: []*metric.HistogramInt64Row{
					{
						Count: 6,
						Sum:   40,
						Values: []int64{
							1,
							2,
							3,
						},
					},
				},
				Info: &metric.HistogramInt64{
					Buckets: []int64{
						0, 5, 10,
					},
				},
				EndTime: &end,
			},
			epoch,
			[]*wire.TimeSeries{
				&wire.TimeSeries{
					Points: []*wire.Point{
						&wire.Point{
							Value: wire.PointDistributionValue{
								DistributionValue: &wire.DistributionValue{
									Count: 6,
									Sum:   40,
									Buckets: []*wire.Bucket{
										{
											Count: 1,
										},
										{
											Count: 2,
										},
										{
											Count: 3,
										},
									},
									BucketOptions: wire.BucketOptionsExplicit{
										Bounds: []float64{
											0, 5, 10,
										},
									},
								},
							},
							Timestamp: &endTimestamp,
						},
					},
					StartTimestamp: &epochTimestamp,
				},
			},
		},
		{
			"HistogramFloat64Data",
			&metric.HistogramFloat64Data{
				Rows: []*metric.HistogramFloat64Row{
					{
						Count: 3,
						Sum:   10,
						Values: []int64{
							1,
							2,
						},
					},
				},
				Info: &metric.HistogramFloat64{
					Buckets: []float64{
						0, 5,
					},
				},
				EndTime: &end,
			},
			epoch,
			[]*wire.TimeSeries{
				&wire.TimeSeries{
					Points: []*wire.Point{
						&wire.Point{
							Value: wire.PointDistributionValue{
								DistributionValue: &wire.DistributionValue{
									Count: 3,
									Sum:   10,
									Buckets: []*wire.Bucket{
										{
											Count: 1,
										},
										{
											Count: 2,
										},
									},
									BucketOptions: wire.BucketOptionsExplicit{
										Bounds: []float64{
											0, 5,
										},
									},
								},
							},
							Timestamp: &endTimestamp,
						},
					},
					StartTimestamp: &epochTimestamp,
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := dataToTimeseries(tt.data, tt.start)
			if !reflect.DeepEqual(got, tt.want) {
				t.Fatalf("Got:\n%s\nWant:\n%s", marshaled(got), marshaled(tt.want))
			}
		})
	}
}

func TestNumRows(t *testing.T) {
	tests := []struct {
		name string
		data telemetry.MetricData
		want int
	}{
		{
			"nil data",
			nil,
			0,
		},
		{
			"1 row Int64Data",
			&metric.Int64Data{
				Rows: []int64{
					0,
				},
			},
			1,
		},
		{
			"2 row Float64Data",
			&metric.Float64Data{
				Rows: []float64{
					0,
					1.0,
				},
			},
			2,
		},
		{
			"1 row HistogramInt64Data",
			&metric.HistogramInt64Data{
				Rows: []*metric.HistogramInt64Row{
					{},
				},
			},
			1,
		},
		{
			"3 row HistogramFloat64Data",
			&metric.HistogramFloat64Data{
				Rows: []*metric.HistogramFloat64Row{
					{},
					{},
					{},
				},
			},
			3,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := numRows(tt.data)
			if !reflect.DeepEqual(got, tt.want) {
				t.Fatalf("Got:\n%s\nWant:\n%s", marshaled(got), marshaled(tt.want))
			}
		})
	}
}

func TestDataToPoints(t *testing.T) {
	end := time.Unix(30, 0)
	endTimestamp := convertTimestamp(end)

	int64Data := &metric.Int64Data{
		Rows: []int64{
			0,
			10,
		},
		EndTime: &end,
	}

	float64Data := &metric.Float64Data{
		Rows: []float64{
			0.5,
			0.25,
		},
		EndTime: &end,
	}

	histogramInt64Data := &metric.HistogramInt64Data{
		Rows: []*metric.HistogramInt64Row{
			{
				Count: 6,
				Sum:   40,
				Values: []int64{
					1,
					2,
					3,
				},
			},
			{
				Count: 12,
				Sum:   80,
				Values: []int64{
					2,
					4,
					6,
				},
			},
		},
		Info: &metric.HistogramInt64{
			Buckets: []int64{
				0, 5, 10,
			},
		},
		EndTime: &end,
	}

	histogramFloat64Data := &metric.HistogramFloat64Data{
		Rows: []*metric.HistogramFloat64Row{
			{
				Count: 6,
				Sum:   40,
				Values: []int64{
					1,
					2,
					3,
				},
			},
			{
				Count: 18,
				Sum:   80,
				Values: []int64{
					3,
					6,
					9,
				},
			},
		},
		Info: &metric.HistogramFloat64{
			Buckets: []float64{
				0, 5, 10,
			},
		},
		EndTime: &end,
	}

	tests := []struct {
		name string
		data telemetry.MetricData
		i    int
		want []*wire.Point
	}{
		{
			"nil data",
			nil,
			0,
			nil,
		},
		{
			"Int64data index 0",
			int64Data,
			0,
			[]*wire.Point{
				{
					Value: wire.PointInt64Value{
						Int64Value: 0,
					},
					Timestamp: &endTimestamp,
				},
			},
		},
		{
			"Int64data index 1",
			int64Data,
			1,
			[]*wire.Point{
				{
					Value: wire.PointInt64Value{
						Int64Value: 10,
					},
					Timestamp: &endTimestamp,
				},
			},
		},
		{
			"Float64Data index 0",
			float64Data,
			0,
			[]*wire.Point{
				{
					Value: wire.PointDoubleValue{
						DoubleValue: 0.5,
					},
					Timestamp: &endTimestamp,
				},
			},
		},
		{
			"Float64Data index 1",
			float64Data,
			1,
			[]*wire.Point{
				{
					Value: wire.PointDoubleValue{
						DoubleValue: 0.25,
					},
					Timestamp: &endTimestamp,
				},
			},
		},
		{
			"HistogramInt64Data index 0",
			histogramInt64Data,
			0,
			[]*wire.Point{
				{
					Value: wire.PointDistributionValue{
						DistributionValue: &wire.DistributionValue{
							Count: 6,
							Sum:   40,
							Buckets: []*wire.Bucket{
								{
									Count: 1,
								},
								{
									Count: 2,
								},
								{
									Count: 3,
								},
							},
							BucketOptions: wire.BucketOptionsExplicit{
								Bounds: []float64{
									0, 5, 10,
								},
							},
						},
					},
					Timestamp: &endTimestamp,
				},
			},
		},
		{
			"HistogramInt64Data index 1",
			histogramInt64Data,
			1,
			[]*wire.Point{
				{
					Value: wire.PointDistributionValue{
						DistributionValue: &wire.DistributionValue{
							Count: 12,
							Sum:   80,
							Buckets: []*wire.Bucket{
								{
									Count: 2,
								},
								{
									Count: 4,
								},
								{
									Count: 6,
								},
							},
							BucketOptions: wire.BucketOptionsExplicit{
								Bounds: []float64{
									0, 5, 10,
								},
							},
						},
					},
					Timestamp: &endTimestamp,
				},
			},
		},
		{
			"HistogramFloat64Data index 0",
			histogramFloat64Data,
			0,
			[]*wire.Point{
				{
					Value: wire.PointDistributionValue{
						DistributionValue: &wire.DistributionValue{
							Count: 6,
							Sum:   40,
							Buckets: []*wire.Bucket{
								{
									Count: 1,
								},
								{
									Count: 2,
								},
								{
									Count: 3,
								},
							},
							BucketOptions: wire.BucketOptionsExplicit{
								Bounds: []float64{
									0, 5, 10,
								},
							},
						},
					},
					Timestamp: &endTimestamp,
				},
			},
		},
		{
			"HistogramFloat64Data index 1",
			histogramFloat64Data,
			1,
			[]*wire.Point{
				{
					Value: wire.PointDistributionValue{
						DistributionValue: &wire.DistributionValue{
							Count: 18,
							Sum:   80,
							Buckets: []*wire.Bucket{
								{
									Count: 3,
								},
								{
									Count: 6,
								},
								{
									Count: 9,
								},
							},
							BucketOptions: wire.BucketOptionsExplicit{
								Bounds: []float64{
									0, 5, 10,
								},
							},
						},
					},
					Timestamp: &endTimestamp,
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := dataToPoints(tt.data, tt.i)
			if !reflect.DeepEqual(got, tt.want) {
				t.Fatalf("Got:\n%s\nWant:\n%s", marshaled(got), marshaled(tt.want))
			}
		})
	}
}

func TestDistributionToPoints(t *testing.T) {
	end := time.Unix(30, 0)
	endTimestamp := convertTimestamp(end)

	tests := []struct {
		name    string
		counts  []int64
		count   int64
		sum     float64
		buckets []float64
		end     time.Time
		want    []*wire.Point
	}{
		{
			name: "3 buckets",
			counts: []int64{
				1,
				2,
				3,
			},
			count: 6,
			sum:   40,
			buckets: []float64{
				0, 5, 10,
			},
			end: end,
			want: []*wire.Point{
				{
					Value: wire.PointDistributionValue{
						DistributionValue: &wire.DistributionValue{
							Count: 6,
							Sum:   40,
							// TODO: SumOfSquaredDeviation?
							Buckets: []*wire.Bucket{
								&wire.Bucket{
									Count: 1,
								},
								&wire.Bucket{
									Count: 2,
								},
								&wire.Bucket{
									Count: 3,
								},
							},
							BucketOptions: wire.BucketOptionsExplicit{
								Bounds: []float64{
									0, 5, 10,
								},
							},
						},
					},
					Timestamp: &endTimestamp,
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := distributionToPoints(tt.counts, tt.count, tt.sum, tt.buckets, tt.end)
			if !reflect.DeepEqual(got, tt.want) {
				t.Fatalf("Got:\n%s\nWant:\n%s", marshaled(got), marshaled(tt.want))
			}
		})
	}

}

func TestInfoKeysToLabelKeys(t *testing.T) {
	tests := []struct {
		name     string
		infoKeys []interface{}
		want     []*wire.LabelKey
	}{
		{
			"empty infoKeys",
			[]interface{}{},
			[]*wire.LabelKey{},
		},
		{
			"empty string infoKey",
			[]interface{}{""},
			[]*wire.LabelKey{
				&wire.LabelKey{
					Key: "",
				},
			},
		},
		{
			"non-empty string infoKey",
			[]interface{}{"hello"},
			[]*wire.LabelKey{
				&wire.LabelKey{
					Key: "hello",
				},
			},
		},
		{
			"multiple element infoKey",
			[]interface{}{"hello", "world"},
			[]*wire.LabelKey{
				&wire.LabelKey{
					Key: "hello",
				},
				&wire.LabelKey{
					Key: "world",
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := infoKeysToLabelKeys(tt.infoKeys)
			if !reflect.DeepEqual(got, tt.want) {
				t.Fatalf("Got:\n%s\nWant:\n%s", marshaled(got), marshaled(tt.want))
			}
		})
	}
}
