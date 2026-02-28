// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package time

type testingT interface {
	Error(args ...any)
	Errorf(format string, args ...any)
	Fail()
	FailNow()
	Failed() bool
	Fatal(args ...any)
	Fatalf(format string, args ...any)
	Helper()
	Log(args ...any)
	Logf(format string, args ...any)
	Skip(args ...any)
	SkipNow()
	Skipf(format string, args ...any)
}

var InternalTests = []struct {
	Name string
	Test func(testingT)
}{
	{"AbsDaysSplit", testAbsDaysSplit},
	{"AbsYdaySplit", testAbsYdaySplit},
	{"AbsDate", testAbsDate},
	{"DateToAbsDays", testDateToAbsDays},
	{"DaysIn", testDaysIn},
	{"DaysBefore", testDaysBefore},
}

func testAbsDaysSplit(t testingT) {
	isLeap := func(year uint64) bool {
		return year%4 == 0 && (year%100 != 0 || year%400 == 0)
	}
	bad := 0
	wantYear := uint64(0)
	wantYday := absYday(0)
	for days := range absDays(1e6) {
		century, cyear, yday := days.split()
		if century != absCentury(wantYear/100) || cyear != absCyear(wantYear%100) || yday != wantYday {
			t.Errorf("absDays(%d).split() = %d, %d, %d, want %d, %d, %d",
				days, century, cyear, yday,
				wantYear/100, wantYear%100, wantYday)
			if bad++; bad >= 20 {
				t.Fatalf("too many errors")
			}
		}
		end := absYday(365)
		if isLeap(wantYear + 1) {
			end = 366
		}
		if wantYday++; wantYday == end {
			wantYear++
			wantYday = 0
		}
	}
}

func testAbsYdaySplit(t testingT) {
	ends := []int{31, 30, 31, 30, 31, 31, 30, 31, 30, 31, 31, 29}
	bad := 0
	wantMonth := absMonth(3)
	wantDay := 1
	for yday := range absYday(366) {
		month, day := yday.split()
		if month != wantMonth || day != wantDay {
			t.Errorf("absYday(%d).split() = %d, %d, want %d, %d", yday, month, day, wantMonth, wantDay)
			if bad++; bad >= 20 {
				t.Fatalf("too many errors")
			}
		}
		if wantDay++; wantDay > ends[wantMonth-3] {
			wantMonth++
			wantDay = 1
		}
	}
}

func testAbsDate(t testingT) {
	ends := []int{31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31}
	isLeap := func(year int) bool {
		y := uint64(year) + absoluteYears
		return y%4 == 0 && (y%100 != 0 || y%400 == 0)
	}
	wantYear := 0
	wantMonth := March
	wantMday := 1
	wantYday := 31 + 29 + 1
	bad := 0
	absoluteYears := int64(absoluteYears)
	for days := range absDays(1e6) {
		year, month, mday := days.date()
		year += int(absoluteYears)
		if year != wantYear || month != wantMonth || mday != wantMday {
			t.Errorf("days(%d).date() = %v, %v, %v, want %v, %v, %v", days,
				year, month, mday,
				wantYear, wantMonth, wantMday)
			if bad++; bad >= 20 {
				t.Fatalf("too many errors")
			}
		}

		year, yday := days.yearYday()
		year += int(absoluteYears)
		if year != wantYear || yday != wantYday {
			t.Errorf("days(%d).yearYday() = %v, %v, want %v, %v, ", days,
				year, yday,
				wantYear, wantYday)
			if bad++; bad >= 20 {
				t.Fatalf("too many errors")
			}
		}

		if wantMday++; wantMday == ends[wantMonth-1]+1 || wantMonth == February && wantMday == 29 && !isLeap(year) {
			wantMonth++
			wantMday = 1
		}
		wantYday++
		if wantMonth == December+1 {
			wantYear++
			wantMonth = January
			wantMday = 1
			wantYday = 1
		}
	}
}

func testDateToAbsDays(t testingT) {
	isLeap := func(year int64) bool {
		return year%4 == 0 && (year%100 != 0 || year%400 == 0)
	}
	wantDays := absDays(marchThruDecember)
	bad := 0
	for year := int64(1); year < 10000; year++ {
		days := dateToAbsDays(year-absoluteYears, January, 1)
		if days != wantDays {
			t.Errorf("dateToAbsDays(abs %d, Jan, 1) = %d, want %d", year, days, wantDays)
			if bad++; bad >= 20 {
				t.Fatalf("too many errors")
			}
		}
		wantDays += 365
		if isLeap(year) {
			wantDays++
		}
	}
}

func testDaysIn(t testingT) {
	isLeap := func(year int) bool {
		return year%4 == 0 && (year%100 != 0 || year%400 == 0)
	}
	want := []int{0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31}
	bad := 0
	for year := 0; year <= 1600; year++ {
		for m := January; m <= December; m++ {
			w := want[m]
			if m == February && isLeap(year) {
				w++
			}
			d := daysIn(m, year-800)
			if d != w {
				t.Errorf("daysIn(%v, %d) = %d, want %d", m, year-800, d, w)
				if bad++; bad >= 20 {
					t.Fatalf("too many errors")
				}
			}
		}
	}
}

func testDaysBefore(t testingT) {
	for m, want := range []int{0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365} {
		d := daysBefore(Month(m + 1))
		if d != want {
			t.Errorf("daysBefore(%d) = %d, want %d", m, d, want)
		}
	}
}
