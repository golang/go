package ocagent_test

import (
	"context"
	"errors"
	"testing"

	"golang.org/x/tools/internal/event"
	"golang.org/x/tools/internal/event/keys"
)

func TestEncodeMetric(t *testing.T) {
	exporter := registerExporter()
	const prefix = testNodeStr + `
	"metrics":[`
	const suffix = `]}`
	tests := []struct {
		name string
		run  func(ctx context.Context)
		want string
	}{
		{
			name: "HistogramFloat64, HistogramInt64",
			run: func(ctx context.Context) {
				ctx = event.Label(ctx, keyMethod.Of("godoc.ServeHTTP"))
				event.Metric(ctx, latencyMs.Of(96.58))
				ctx = event.Label(ctx, keys.Err.Of(errors.New("panic: fatal signal")))
				event.Metric(ctx, bytesIn.Of(97e2))
			},
			want: prefix + `
			{
				"metric_descriptor": {
					"name": "latency_ms",
					"description": "The latency of calls in milliseconds",
					"type": 6,
					"label_keys": [
						{
							"key": "method"
						},
						{
							"key": "route"
						}
					]
				},
				"timeseries": [
					{
						"start_timestamp": "1970-01-01T00:00:00Z",
						"points": [
							{
								"timestamp": "1970-01-01T00:00:40Z",
								"distributionValue": {
									"count": 1,
									"sum": 96.58,
									"bucket_options": {
										"explicit": {
											"bounds": [
												0,
												5,
												10,
												25,
												50
											]
										}
									},
									"buckets": [
										{},
										{},
										{},
										{},
										{}
									]
								}
							}
						]
					}
				]
			},
			{
				"metric_descriptor": {
					"name": "latency_ms",
					"description": "The latency of calls in milliseconds",
					"type": 6,
					"label_keys": [
						{
							"key": "method"
						},
						{
							"key": "route"
						}
					]
				},
				"timeseries": [
					{
						"start_timestamp": "1970-01-01T00:00:00Z",
						"points": [
							{
								"timestamp": "1970-01-01T00:00:40Z",
								"distributionValue": {
									"count": 1,
									"sum": 9700,
									"bucket_options": {
										"explicit": {
											"bounds": [
												0,
												10,
												50,
												100,
												500,
												1000,
												2000
											]
										}
									},
									"buckets": [
										{},
										{},
										{},
										{},
										{},
										{},
										{}
									]
								}
							}
						]
					}
				]
			}` + suffix,
		},
	}

	ctx := context.TODO()
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tt.run(ctx)
			got := exporter.Output("/v1/metrics")
			checkJSON(t, got, []byte(tt.want))
		})
	}
}
