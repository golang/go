// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package time

import (
	"sync"
)

func ResetLocalOnceForTest() {
	localOnce = sync.Once{}
	localLoc = Location{}
}

func ForceUSPacificForTesting() {
	ResetLocalOnceForTest()
	localOnce.Do(initTestingZone)
}

func ZoneinfoForTesting() *string {
	return zoneinfo
}

func ResetZoneinfoForTesting() {
	zoneinfo = nil
	zoneinfoOnce = sync.Once{}
}

var (
	ForceZipFileForTesting = forceZipFileForTesting
	ParseTimeZone          = parseTimeZone
	SetMono                = (*Time).setMono
	GetMono                = (*Time).mono
	ErrLocation            = errLocation
	ReadFile               = readFile
	LoadTzinfo             = loadTzinfo
	NextStdChunk           = nextStdChunk
	Tzset                  = tzset
	TzsetName              = tzsetName
	TzsetOffset            = tzsetOffset
)

func LoadFromEmbeddedTZData(zone string) (string, error) {
	return loadFromEmbeddedTZData(zone)
}

type RuleKind int

const (
	RuleJulian       = RuleKind(ruleJulian)
	RuleDOY          = RuleKind(ruleDOY)
	RuleMonthWeekDay = RuleKind(ruleMonthWeekDay)
	UnixToInternal   = unixToInternal
)

type Rule struct {
	Kind RuleKind
	Day  int
	Week int
	Mon  int
	Time int
}

func TzsetRule(s string) (Rule, string, bool) {
	r, rs, ok := tzsetRule(s)
	rr := Rule{
		Kind: RuleKind(r.kind),
		Day:  r.day,
		Week: r.week,
		Mon:  r.mon,
		Time: r.time,
	}
	return rr, rs, ok
}

// StdChunkNames maps from nextStdChunk results to the matched strings.
var StdChunkNames = map[int]string{
	0:                               "",
	stdLongMonth:                    "January",
	stdMonth:                        "Jan",
	stdNumMonth:                     "1",
	stdZeroMonth:                    "01",
	stdLongWeekDay:                  "Monday",
	stdWeekDay:                      "Mon",
	stdDay:                          "2",
	stdUnderDay:                     "_2",
	stdZeroDay:                      "02",
	stdUnderYearDay:                 "__2",
	stdZeroYearDay:                  "002",
	stdHour:                         "15",
	stdHour12:                       "3",
	stdZeroHour12:                   "03",
	stdMinute:                       "4",
	stdZeroMinute:                   "04",
	stdSecond:                       "5",
	stdZeroSecond:                   "05",
	stdLongYear:                     "2006",
	stdYear:                         "06",
	stdPM:                           "PM",
	stdpm:                           "pm",
	stdTZ:                           "MST",
	stdISO8601TZ:                    "Z0700",
	stdISO8601SecondsTZ:             "Z070000",
	stdISO8601ShortTZ:               "Z07",
	stdISO8601ColonTZ:               "Z07:00",
	stdISO8601ColonSecondsTZ:        "Z07:00:00",
	stdNumTZ:                        "-0700",
	stdNumSecondsTz:                 "-070000",
	stdNumShortTZ:                   "-07",
	stdNumColonTZ:                   "-07:00",
	stdNumColonSecondsTZ:            "-07:00:00",
	stdFracSecond0 | 1<<stdArgShift: ".0",
	stdFracSecond0 | 2<<stdArgShift: ".00",
	stdFracSecond0 | 3<<stdArgShift: ".000",
	stdFracSecond0 | 4<<stdArgShift: ".0000",
	stdFracSecond0 | 5<<stdArgShift: ".00000",
	stdFracSecond0 | 6<<stdArgShift: ".000000",
	stdFracSecond0 | 7<<stdArgShift: ".0000000",
	stdFracSecond0 | 8<<stdArgShift: ".00000000",
	stdFracSecond0 | 9<<stdArgShift: ".000000000",
	stdFracSecond9 | 1<<stdArgShift: ".9",
	stdFracSecond9 | 2<<stdArgShift: ".99",
	stdFracSecond9 | 3<<stdArgShift: ".999",
	stdFracSecond9 | 4<<stdArgShift: ".9999",
	stdFracSecond9 | 5<<stdArgShift: ".99999",
	stdFracSecond9 | 6<<stdArgShift: ".999999",
	stdFracSecond9 | 7<<stdArgShift: ".9999999",
	stdFracSecond9 | 8<<stdArgShift: ".99999999",
	stdFracSecond9 | 9<<stdArgShift: ".999999999",
}

var Quote = quote
