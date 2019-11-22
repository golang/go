package ocagent

import (
	"reflect"
	"testing"
	"time"

	"golang.org/x/tools/internal/telemetry"
	"golang.org/x/tools/internal/telemetry/export/ocagent/wire"
	"golang.org/x/tools/internal/telemetry/metric"
)

func TestEncodeMetric(t *testing.T) {
	epoch := time.Unix(0, 0)
	epochTimestamp := epoch.Format(time.RFC3339Nano)
	end := time.Unix(30, 0)
	endTimestamp := end.Format(time.RFC3339Nano)

	tests := []struct {
		name  string
		data  telemetry.MetricData
		start time.Time
		want  *wire.Metric
	}{
		{
			name: "nil data",
		},
		{
			name: "Int64Data cumulative",
			data: &metric.Int64Data{
				Info: &metric.Scalar{
					Name:        "int",
					Description: "int metric",
					Keys:        []interface{}{"hello"},
				},
				Rows: []int64{
					1,
					2,
					3,
				},
				EndTime: &end,
			},
			start: epoch,
			want: &wire.Metric{
				MetricDescriptor: &wire.MetricDescriptor{
					Name:        "int",
					Description: "int metric",
					Type:        wire.MetricDescriptor_CUMULATIVE_INT64,
					LabelKeys: []*wire.LabelKey{
						&wire.LabelKey{
							Key: "hello",
						},
					},
				},
				Timeseries: []*wire.TimeSeries{
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
		},
		{
			name: "Int64Data gauge",
			data: &metric.Int64Data{
				Info: &metric.Scalar{
					Name:        "int-gauge",
					Description: "int metric gauge",
					Keys:        []interface{}{"hello"},
				},
				IsGauge: true,
			},
			start: epoch,
			want: &wire.Metric{
				MetricDescriptor: &wire.MetricDescriptor{
					Name:        "int-gauge",
					Description: "int metric gauge",
					Type:        wire.MetricDescriptor_GAUGE_INT64,
					LabelKeys: []*wire.LabelKey{
						&wire.LabelKey{
							Key: "hello",
						},
					},
				},
				Timeseries: []*wire.TimeSeries{},
			},
		},
		{
			name: "Float64Data cumulative",
			data: &metric.Float64Data{
				Info: &metric.Scalar{
					Name:        "float",
					Description: "float metric",
					Keys:        []interface{}{"world"},
				},
				Rows: []float64{
					1.5,
					4.5,
				},
				EndTime: &end,
			},
			start: epoch,
			want: &wire.Metric{
				MetricDescriptor: &wire.MetricDescriptor{
					Name:        "float",
					Description: "float metric",
					Type:        wire.MetricDescriptor_CUMULATIVE_DOUBLE,
					LabelKeys: []*wire.LabelKey{
						&wire.LabelKey{
							Key: "world",
						},
					},
				},
				Timeseries: []*wire.TimeSeries{
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
		},
		{
			name: "Float64Data gauge",
			data: &metric.Float64Data{
				Info: &metric.Scalar{
					Name:        "float-gauge",
					Description: "float metric gauge",
					Keys:        []interface{}{"world"},
				},
				IsGauge: true,
			},
			start: epoch,
			want: &wire.Metric{
				MetricDescriptor: &wire.MetricDescriptor{
					Name:        "float-gauge",
					Description: "float metric gauge",
					Type:        wire.MetricDescriptor_GAUGE_DOUBLE,
					LabelKeys: []*wire.LabelKey{
						&wire.LabelKey{
							Key: "world",
						},
					},
				},
				Timeseries: []*wire.TimeSeries{},
			},
		},
		{
			name: "HistogramInt64",
			data: &metric.HistogramInt64Data{
				Info: &metric.HistogramInt64{
					Name:        "histogram int",
					Description: "histogram int metric",
					Keys:        []interface{}{"hello"},
					Buckets: []int64{
						0, 5, 10,
					},
				},
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
				EndTime: &end,
			},
			start: epoch,
			want: &wire.Metric{
				MetricDescriptor: &wire.MetricDescriptor{
					Name:        "histogram int",
					Description: "histogram int metric",
					Type:        wire.MetricDescriptor_CUMULATIVE_DISTRIBUTION,
					LabelKeys: []*wire.LabelKey{
						&wire.LabelKey{
							Key: "hello",
						},
					},
				},
				Timeseries: []*wire.TimeSeries{
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
										BucketOptions: &wire.BucketOptionsExplicit{
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
		},
		{
			name: "HistogramFloat64",
			data: &metric.HistogramFloat64Data{
				Info: &metric.HistogramFloat64{
					Name:        "histogram float",
					Description: "histogram float metric",
					Keys:        []interface{}{"hello"},
					Buckets: []float64{
						0, 5,
					},
				},
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
				EndTime: &end,
			},
			start: epoch,
			want: &wire.Metric{
				MetricDescriptor: &wire.MetricDescriptor{
					Name:        "histogram float",
					Description: "histogram float metric",
					Type:        wire.MetricDescriptor_CUMULATIVE_DISTRIBUTION,
					LabelKeys: []*wire.LabelKey{
						&wire.LabelKey{
							Key: "hello",
						},
					},
				},
				Timeseries: []*wire.TimeSeries{
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
										BucketOptions: &wire.BucketOptionsExplicit{
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
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := convertMetric(tt.data, tt.start)
			if !reflect.DeepEqual(got, tt.want) {
				t.Fatalf("Got:\n%#v\nWant:\n%#v", got, tt.want)
			}
		})
	}
}
