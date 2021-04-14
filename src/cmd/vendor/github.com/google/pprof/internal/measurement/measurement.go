// Copyright 2014 Google Inc. All Rights Reserved.
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

// Package measurement export utility functions to manipulate/format performance profile sample values.
package measurement

import (
	"fmt"
	"math"
	"strings"
	"time"

	"github.com/google/pprof/profile"
)

// ScaleProfiles updates the units in a set of profiles to make them
// compatible. It scales the profiles to the smallest unit to preserve
// data.
func ScaleProfiles(profiles []*profile.Profile) error {
	if len(profiles) == 0 {
		return nil
	}
	periodTypes := make([]*profile.ValueType, 0, len(profiles))
	for _, p := range profiles {
		if p.PeriodType != nil {
			periodTypes = append(periodTypes, p.PeriodType)
		}
	}
	periodType, err := CommonValueType(periodTypes)
	if err != nil {
		return fmt.Errorf("period type: %v", err)
	}

	// Identify common sample types
	numSampleTypes := len(profiles[0].SampleType)
	for _, p := range profiles[1:] {
		if numSampleTypes != len(p.SampleType) {
			return fmt.Errorf("inconsistent samples type count: %d != %d", numSampleTypes, len(p.SampleType))
		}
	}
	sampleType := make([]*profile.ValueType, numSampleTypes)
	for i := 0; i < numSampleTypes; i++ {
		sampleTypes := make([]*profile.ValueType, len(profiles))
		for j, p := range profiles {
			sampleTypes[j] = p.SampleType[i]
		}
		sampleType[i], err = CommonValueType(sampleTypes)
		if err != nil {
			return fmt.Errorf("sample types: %v", err)
		}
	}

	for _, p := range profiles {
		if p.PeriodType != nil && periodType != nil {
			period, _ := Scale(p.Period, p.PeriodType.Unit, periodType.Unit)
			p.Period, p.PeriodType.Unit = int64(period), periodType.Unit
		}
		ratios := make([]float64, len(p.SampleType))
		for i, st := range p.SampleType {
			if sampleType[i] == nil {
				ratios[i] = 1
				continue
			}
			ratios[i], _ = Scale(1, st.Unit, sampleType[i].Unit)
			p.SampleType[i].Unit = sampleType[i].Unit
		}
		if err := p.ScaleN(ratios); err != nil {
			return fmt.Errorf("scale: %v", err)
		}
	}
	return nil
}

// CommonValueType returns the finest type from a set of compatible
// types.
func CommonValueType(ts []*profile.ValueType) (*profile.ValueType, error) {
	if len(ts) <= 1 {
		return nil, nil
	}
	minType := ts[0]
	for _, t := range ts[1:] {
		if !compatibleValueTypes(minType, t) {
			return nil, fmt.Errorf("incompatible types: %v %v", *minType, *t)
		}
		if ratio, _ := Scale(1, t.Unit, minType.Unit); ratio < 1 {
			minType = t
		}
	}
	rcopy := *minType
	return &rcopy, nil
}

func compatibleValueTypes(v1, v2 *profile.ValueType) bool {
	if v1 == nil || v2 == nil {
		return true // No grounds to disqualify.
	}
	// Remove trailing 's' to permit minor mismatches.
	if t1, t2 := strings.TrimSuffix(v1.Type, "s"), strings.TrimSuffix(v2.Type, "s"); t1 != t2 {
		return false
	}

	return v1.Unit == v2.Unit ||
		(isTimeUnit(v1.Unit) && isTimeUnit(v2.Unit)) ||
		(isMemoryUnit(v1.Unit) && isMemoryUnit(v2.Unit))
}

// Scale a measurement from an unit to a different unit and returns
// the scaled value and the target unit. The returned target unit
// will be empty if uninteresting (could be skipped).
func Scale(value int64, fromUnit, toUnit string) (float64, string) {
	// Avoid infinite recursion on overflow.
	if value < 0 && -value > 0 {
		v, u := Scale(-value, fromUnit, toUnit)
		return -v, u
	}
	if m, u, ok := memoryLabel(value, fromUnit, toUnit); ok {
		return m, u
	}
	if t, u, ok := timeLabel(value, fromUnit, toUnit); ok {
		return t, u
	}
	// Skip non-interesting units.
	switch toUnit {
	case "count", "sample", "unit", "minimum", "auto":
		return float64(value), ""
	default:
		return float64(value), toUnit
	}
}

// Label returns the label used to describe a certain measurement.
func Label(value int64, unit string) string {
	return ScaledLabel(value, unit, "auto")
}

// ScaledLabel scales the passed-in measurement (if necessary) and
// returns the label used to describe a float measurement.
func ScaledLabel(value int64, fromUnit, toUnit string) string {
	v, u := Scale(value, fromUnit, toUnit)
	sv := strings.TrimSuffix(fmt.Sprintf("%.2f", v), ".00")
	if sv == "0" || sv == "-0" {
		return "0"
	}
	return sv + u
}

// Percentage computes the percentage of total of a value, and encodes
// it as a string. At least two digits of precision are printed.
func Percentage(value, total int64) string {
	var ratio float64
	if total != 0 {
		ratio = math.Abs(float64(value)/float64(total)) * 100
	}
	switch {
	case math.Abs(ratio) >= 99.95 && math.Abs(ratio) <= 100.05:
		return "  100%"
	case math.Abs(ratio) >= 1.0:
		return fmt.Sprintf("%5.2f%%", ratio)
	default:
		return fmt.Sprintf("%5.2g%%", ratio)
	}
}

// isMemoryUnit returns whether a name is recognized as a memory size
// unit.
func isMemoryUnit(unit string) bool {
	switch strings.TrimSuffix(strings.ToLower(unit), "s") {
	case "byte", "b", "kilobyte", "kb", "megabyte", "mb", "gigabyte", "gb":
		return true
	}
	return false
}

func memoryLabel(value int64, fromUnit, toUnit string) (v float64, u string, ok bool) {
	fromUnit = strings.TrimSuffix(strings.ToLower(fromUnit), "s")
	toUnit = strings.TrimSuffix(strings.ToLower(toUnit), "s")

	switch fromUnit {
	case "byte", "b":
	case "kb", "kbyte", "kilobyte":
		value *= 1024
	case "mb", "mbyte", "megabyte":
		value *= 1024 * 1024
	case "gb", "gbyte", "gigabyte":
		value *= 1024 * 1024 * 1024
	case "tb", "tbyte", "terabyte":
		value *= 1024 * 1024 * 1024 * 1024
	case "pb", "pbyte", "petabyte":
		value *= 1024 * 1024 * 1024 * 1024 * 1024
	default:
		return 0, "", false
	}

	if toUnit == "minimum" || toUnit == "auto" {
		switch {
		case value < 1024:
			toUnit = "b"
		case value < 1024*1024:
			toUnit = "kb"
		case value < 1024*1024*1024:
			toUnit = "mb"
		case value < 1024*1024*1024*1024:
			toUnit = "gb"
		case value < 1024*1024*1024*1024*1024:
			toUnit = "tb"
		default:
			toUnit = "pb"
		}
	}

	var output float64
	switch toUnit {
	default:
		output, toUnit = float64(value), "B"
	case "kb", "kbyte", "kilobyte":
		output, toUnit = float64(value)/1024, "kB"
	case "mb", "mbyte", "megabyte":
		output, toUnit = float64(value)/(1024*1024), "MB"
	case "gb", "gbyte", "gigabyte":
		output, toUnit = float64(value)/(1024*1024*1024), "GB"
	case "tb", "tbyte", "terabyte":
		output, toUnit = float64(value)/(1024*1024*1024*1024), "TB"
	case "pb", "pbyte", "petabyte":
		output, toUnit = float64(value)/(1024*1024*1024*1024*1024), "PB"
	}
	return output, toUnit, true
}

// isTimeUnit returns whether a name is recognized as a time unit.
func isTimeUnit(unit string) bool {
	unit = strings.ToLower(unit)
	if len(unit) > 2 {
		unit = strings.TrimSuffix(unit, "s")
	}

	switch unit {
	case "nanosecond", "ns", "microsecond", "millisecond", "ms", "s", "second", "sec", "hr", "day", "week", "year":
		return true
	}
	return false
}

func timeLabel(value int64, fromUnit, toUnit string) (v float64, u string, ok bool) {
	fromUnit = strings.ToLower(fromUnit)
	if len(fromUnit) > 2 {
		fromUnit = strings.TrimSuffix(fromUnit, "s")
	}

	toUnit = strings.ToLower(toUnit)
	if len(toUnit) > 2 {
		toUnit = strings.TrimSuffix(toUnit, "s")
	}

	var d time.Duration
	switch fromUnit {
	case "nanosecond", "ns":
		d = time.Duration(value) * time.Nanosecond
	case "microsecond":
		d = time.Duration(value) * time.Microsecond
	case "millisecond", "ms":
		d = time.Duration(value) * time.Millisecond
	case "second", "sec", "s":
		d = time.Duration(value) * time.Second
	case "cycle":
		return float64(value), "", true
	default:
		return 0, "", false
	}

	if toUnit == "minimum" || toUnit == "auto" {
		switch {
		case d < 1*time.Microsecond:
			toUnit = "ns"
		case d < 1*time.Millisecond:
			toUnit = "us"
		case d < 1*time.Second:
			toUnit = "ms"
		case d < 1*time.Minute:
			toUnit = "sec"
		case d < 1*time.Hour:
			toUnit = "min"
		case d < 24*time.Hour:
			toUnit = "hour"
		case d < 15*24*time.Hour:
			toUnit = "day"
		case d < 120*24*time.Hour:
			toUnit = "week"
		default:
			toUnit = "year"
		}
	}

	var output float64
	dd := float64(d)
	switch toUnit {
	case "ns", "nanosecond":
		output, toUnit = dd/float64(time.Nanosecond), "ns"
	case "us", "microsecond":
		output, toUnit = dd/float64(time.Microsecond), "us"
	case "ms", "millisecond":
		output, toUnit = dd/float64(time.Millisecond), "ms"
	case "min", "minute":
		output, toUnit = dd/float64(time.Minute), "mins"
	case "hour", "hr":
		output, toUnit = dd/float64(time.Hour), "hrs"
	case "day":
		output, toUnit = dd/float64(24*time.Hour), "days"
	case "week", "wk":
		output, toUnit = dd/float64(7*24*time.Hour), "wks"
	case "year", "yr":
		output, toUnit = dd/float64(365*24*time.Hour), "yrs"
	default:
		// "sec", "second", "s" handled by default case.
		output, toUnit = dd/float64(time.Second), "s"
	}
	return output, toUnit, true
}
