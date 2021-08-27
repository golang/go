package user

import (
	"internal/syscall/windows/registry"
	"syscall"
)

// _profileListKey registry key contains all local user/group SIDs
// (Security Identifiers are Windows version of user/group ids on unix systems)
// as sub keys. It is a sub key of HKEY_LOCAL_MACHINE. Since
// HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Windows NT\CurrentVersion\ProfileList
// registry key does not contain SIDs of Administrator, Guest, or other default
// system users, some users or groups might not be provided when using
// iterateUsers or iterateGroups
const _profileListKey = `SOFTWARE\Microsoft\Windows NT\CurrentVersion\ProfileList`

// iterateSIDS iterates through _profileListKey sub keys and calls provided
// fn with enumerated sub key name as parameter. If fn returns non-nil error,
// iteration is terminated.
func iterateSIDS(fn func(string) error) error {
	k, err := registry.OpenKey(registry.LOCAL_MACHINE, _profileListKey, registry.QUERY_VALUE|registry.ENUMERATE_SUB_KEYS)
	if err != nil {
		return err
	}
	return k.ReadSubKeyNames(fn)
}

// iterateUsers iterates through _profileListKey SIDs, looks up for user
// with each given SID and calls user provided fn with each *User entry. Each iterated SID can be either user or group. Only user SIDs are processed.
func iterateUsers(fn NextUserFunc) error {
	return iterateSIDS(func(sid string) error {
		SID, err := syscall.StringToSid(sid)
		if err != nil {
			return err
		}

		// Skip non-user SID
		if _, _, accType, _ := SID.LookupAccount(""); accType != syscall.SidTypeUser {
			return nil
		}
		u, err := newUserFromSid(SID)
		if err != nil {
			return err
		}

		// Callback to user supplied fn, with user
		return fn(u)
	})
}

// iterateGroups iterates through _profileListKey SIDs, looks up for group with
// each given SID and calls user provided fn with each *Group entry. Each
// iterated SID can be either user or group. Only group SIDs are processed.
func iterateGroups(fn NextGroupFunc) error {
	return iterateSIDS(func(sid string) error {
		SID, err := syscall.StringToSid(sid)
		if err != nil {
			return err
		}

		groupname, _, t, err := SID.LookupAccount("")
		if err != nil {
			return err
		}
		// Skip non-group SID
		if isNotGroup(t) {
			return nil
		}
		g := &Group{Name: groupname, Gid: sid}

		// Callback to user supplied fn, with group
		return fn(g)
	})
}
