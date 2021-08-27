//go:build ((darwin || freebsd || openbsd || netbsd) && cgo && !osusergo) || windows

package user

import (
	"errors"
	"testing"
)

// As BSDs (including darwin) do not support fgetpwent(3)/fgetgrent(3), attempt
// to check if at least 1 user/group record can be retrieved.
// On Windows, it is not possible to easily mock registry. Checking if at
// least one user and group can be retrieved via iteration will suffice.

var _stopErr = errors.New("terminate iteration")

func TestIterateUser(t *testing.T) {
	gotAtLeastOne := false
	err := iterateUsers(func(user *User) error {
		if *user == (User{}) {
			t.Errorf("parsed user is empty: %+v", user)
		}
		gotAtLeastOne = true
		return _stopErr
	})

	if err != _stopErr {
		t.Errorf("iterating users: %w", err)
	}

	if !gotAtLeastOne {
		t.Errorf("no users were iterated")
	}
}

func TestIterateGroup(t *testing.T) {
	gotAtLeastOne := false
	err := iterateGroups(func(group *Group) error {
		if *group == (Group{}) {
			t.Errorf("parsed group is empty: %+v", group)
		}
		gotAtLeastOne = true
		return _stopErr
	})

	if err != _stopErr {
		t.Errorf("iterating groups: %w", err)
	}

	if !gotAtLeastOne {
		t.Errorf("no groups were iterated")
	}
}
