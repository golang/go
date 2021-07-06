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
		(timeUnits.sniffUnit(v1.Unit) != nil && timeUnits.sniffUnit(v2.Unit) != nil) ||
		(memoryUnits.sniffUnit(v1.Unit) != nil && memoryUnits.sniffUnit(v2.Unit) != nil) ||
		(gcuUnits.sniffUnit(v1.Unit) != nil && gcuUnits.sniffUnit(v2.Unit) != nil)
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
	if m, u, ok := memoryUnits.convertUnit(value, fromUnit, toUnit); ok {
		return m, u
	}
	if t, u, ok := timeUnits.convertUnit(value, fromUnit, toUnit); ok {
		return t, u
	}
	if g, u, ok := gcuUnits.convertUnit(value, fromUnit, toUnit); ok {
		return g, u
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

// unit includes a list of aliases representing a specific unit and a factor
// which one can multiple a value in the specified unit by to get the value
// in terms of the base unit.
type unit struct {
	canonicalName string
	aliases       []string
	factor        float64
}

// unitType includes a list of units that are within the same category (i.e.
// memory or time units) and a default unit to use for this type of unit.
type unitType struct {
	defaultUnit unit
	units       []unit
}

// findByAlias returns the unit associated with the specified alias. It returns
// nil if the unit with such alias is not found.
func (ut unitType) findByAlias(alias string) *unit {
	for _, u := range ut.units {
		for _, a := range u.aliases {
			if alias == a {
				return &u
			}
		}
	}
	return nil
}

// sniffUnit simpifies the input alias and returns the unit associated with the
// specified alias. It returns nil if the unit with such alias is not found.
func (ut unitType) sniffUnit(unit string) *unit {
	unit = strings.ToLower(unit)
	if len(unit) > 2 {
		unit = strings.TrimSuffix(unit, "s")
	}
	return ut.findByAlias(unit)
}

// autoScale takes in the value with units of the base unit and returns
// that value scaled to a reasonable unit if a reasonable unit is
// found.
func (ut unitType) autoScale(value float64) (float64, string, bool) {
	var f float64
	var unit string
	for _, u := range ut.units {
		if u.factor >= f && (value/u.factor) >= 1.0 {
			f = u.factor
			unit = u.canonicalName
		}
	}
	if f == 0 {
		return 0, "", false
	}
	return value / f, unit, true
}

// convertUnit converts a value from the fromUnit to the toUnit, autoscaling
// the value if the toUnit is "minimum" or "auto". If the fromUnit is not
// included in the unitType, then a false boolean will be returned. If the
// toUnit is not in the unitType, the value will be returned in terms of the
// default unitType.
func (ut unitType) convertUnit(value int64, fromUnitStr, toUnitStr string) (float64, string, bool) {
	fromUnit := ut.sniffUnit(fromUnitStr)
	if fromUnit == nil {
		return 0, "", false
	}
	v := float64(value) * fromUnit.factor
	if toUnitStr == "minimum" || toUnitStr == "auto" {
		if v, u, ok := ut.autoScale(v); ok {
			return v, u, true
		}
		return v / ut.defaultUnit.factor, ut.defaultUnit.canonicalName, true
	}
	toUnit := ut.sniffUnit(toUnitStr)
	if toUnit == nil {
		return v / ut.defaultUnit.factor, ut.defaultUnit.canonicalName, true
	}
	return v / toUnit.factor, toUnit.canonicalName, true
}

var memoryUnits = unitType{
	units: []unit{
		{"B", []string{"b", "byte"}, 1},
		{"kB", []string{"kb", "kbyte", "kilobyte"}, float64(1 << 10)},
		{"MB", []string{"mb", "mbyte", "megabyte"}, float64(1 << 20)},
		{"GB", []string{"gb", "gbyte", "gigabyte"}, float64(1 << 30)},
		{"TB", []string{"tb", "tbyte", "terabyte"}, float64(1 << 40)},
		{"PB", []string{"pb", "pbyte", "petabyte"}, float64(1 << 50)},
	},
	defaultUnit: unit{"B", []string{"b", "byte"}, 1},
}

var timeUnits = unitType{
	units: []unit{
		{"ns", []string{"ns", "nanosecond"}, float64(time.Nanosecond)},
		{"us", []string{"μs", "us", "microsecond"}, float64(time.Microsecond)},
		{"ms", []string{"ms", "millisecond"}, float64(time.Millisecond)},
		{"s", []string{"s", "sec", "second"}, float64(time.Second)},
		{"hrs", []string{"hour", "hr"}, float64(time.Hour)},
	},
	defaultUnit: unit{"s", []string{}, float64(time.Second)},
}

var gcuUnits = unitType{
	units: []unit{
		{"n*GCU", []string{"nanogcu"}, 1e-9},
		{"u*GCU", []string{"microgcu"}, 1e-6},
		{"m*GCU", []string{"milligcu"}, 1e-3},
		{"GCU", []string{"gcu"}, 1},
		{"k*GCU", []string{"kilogcu"}, 1e3},
		{"M*GCU", []string{"megagcu"}, 1e6},
		{"G*GCU", []string{"gigagcu"}, 1e9},
		{"T*GCU", []string{"teragcu"}, 1e12},
		{"P*GCU", []string{"petagcu"}, 1e15},
	},
	defaultUnit: unit{"GCU", []string{}, 1.0},
}
