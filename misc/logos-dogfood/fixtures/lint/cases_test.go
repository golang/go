package lintcases

import "testing"

func TestExecutedCloseForms(t *testing.T) {
	for _, closeResource := range []func() error{CloseChecked, CloseAssigned, CloseReturned} {
		if err := closeResource(); err != nil {
			t.Fatal(err)
		}
	}
}

func TestShadowedPanicReturnsTypedNil(t *testing.T) {
	if err := TypedNilAfterShadowedPanic(); err == nil {
		t.Fatal("a typed nil stored in error must not compare equal to nil")
	}
}
