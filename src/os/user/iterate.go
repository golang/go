package user

// NextUserFunc is used in users iteration process. It receives *User for each user record.
// If non-nil error is returned from NextUserFunc - iteration process is terminated.
type NextUserFunc func(*User) error

// NextGroupFunc is used in groups iteration process. It receives *Group for each group record.
// If non-nil error is returned from NextGroupFunc - iteration process is terminated.
type NextGroupFunc func(*Group) error

// IterateUsers iterates over user entries. For each retrieved *User entry provided NextUserFunc is called.
//
// On UNIX, if CGO is enabled, getpwent(3) is used in the underlying implementation. Since getpwent(3) is not thread-safe,
// locking is strongly advised.
func IterateUsers(n NextUserFunc) error {
	return iterateUsers(n)
}

// IterateGroups iterates over group entries. For each retrieved *Group entry provided NextGroupFunc is called.
//
// On UNIX, if CGO is enabled, getgrent(3) is used in the underlying implementation. Since getgrent(3) is not thread-safe,
// locking is strongly advised.
func IterateGroups(n NextGroupFunc) error {
	return iterateGroups(n)
}
