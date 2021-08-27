//go:build (aix || dragonfly || freebsd || (!android && linux) || netbsd || openbsd || solaris) && cgo && !osusergo
// +build aix dragonfly freebsd !android,linux netbsd openbsd solaris
// +build cgo
// +build !osusergo

package user

// On darwin, there seems to be some issues when using getpwent(3)
// and getgrent(3). Until the issues are fixed, it is not recommended
// relying on these libc library calls. As such, cgo version of
// users and groups iterators should be disabled on darwin.
// https://developer.apple.com/forums/thread/689613

/*
#include <unistd.h>
#include <sys/types.h>
#include <pwd.h>
#include <grp.h>
#include <stdlib.h>
#include <stdio.h>
#include <errno.h>

static void resetErrno(){
	errno = 0;
}
*/
import "C"

// usersHelper defines the methods used in users iteration process within
// iterateUsers. This interface allows testing iterateUsers functionality.
// iterate_test_fgetent.go file defines test related struct that implements
// usersHelper.
type usersHelper interface {
	// set sets up internal state before iteration
	set()

	// get sequentially returns a passwd structure which is later processed into *User entry
	get() (*C.struct_passwd, error)

	// end cleans up internal state after iteration is done
	end()
}

type iterateUsersHelper struct{}

func (i iterateUsersHelper) set() {
	C.setpwent()
}

func (i iterateUsersHelper) get() (*C.struct_passwd, error) {
	var result *C.struct_passwd
	result, err := C.getpwent()
	return result, err
}

func (i iterateUsersHelper) end() {
	C.endpwent()
}

// This helper is used to retrieve users via c library call. A global
// variable which implements usersHelper interface is needed in order to
// separate testing logic from production. Since cgo can not be used directly
// in tests, iterate_test_fgetent.go file provides iterateUsersHelperTest
// structure which implements usersHelper interface and can substitute
// default userIterator value.
var userIterator usersHelper = iterateUsersHelper{}

// iterateUsers iterates over users database via getpwent(3). If fn returns non
// nil error, then iteration is terminated. A nil result from getpwent means
// there were no more entries, or an error occurred, as such, iteration is
// terminated, and if error was encountered it is returned.
//
// Since iterateUsers uses getpwent(3), which is not thread safe, iterateUsers
// can not bet used concurrently. If concurrent usage is required, it is
// recommended to use locking mechanism such as sync.Mutex when calling
// iterateUsers from multiple goroutines.
func iterateUsers(fn NextUserFunc) error {
	userIterator.set()
	defer userIterator.end()
	for {
		var result *C.struct_passwd
		C.resetErrno()
		result, err := userIterator.get()

		// If result is nil - getpwent iterated through entire users database or there was an error
		if result == nil {
			return err
		}

		if err = fn(buildUser(result)); err != nil {
			// User provided non-nil error means that iteration should be terminated
			return err
		}
	}
}

// groupsHelper defines the methods used in groups iteration process within iterateGroups. This interface allows testing
// iterateGroups functionality. iterate_test_fgetent.go file defines test related struct that implements groupsHelper.
type groupsHelper interface {
	// set sets up internal state before iteration
	set()

	// get sequentially returns a group structure which is later processed into *Group entry
	get() (*C.struct_group, error)

	// end cleans up internal state after iteration is done
	end()
}

type iterateGroupsHelper struct{}

func (i iterateGroupsHelper) set() {
	C.setgrent()
}

func (i iterateGroupsHelper) get() (*C.struct_group, error) {
	var result *C.struct_group
	result, err := C.getgrent()
	return result, err
}

func (i iterateGroupsHelper) end() {
	C.endgrent()
}

// This helper is used to retrieve groups via c library call. A global
// variable which implements groupsHelper interface is needed in order to
// separate testing logic from production. Since cgo can not be used directly
// in tests, iterate_test_fgetent.go file provides iterateGroupsHelperTest
// structure which implements groupsHelper interface and can substitute
// default groupIterator value.
var groupIterator groupsHelper = iterateGroupsHelper{}

// iterateGroups iterates over groups database via getgrent(3). If fn returns
// non nil error, then iteration is terminated. A nil result from getgrent means
// there were no more entries, or an error occurred, as such, iteration is
// terminated, and if error was encountered it is returned.
//
// Since iterateGroups uses getgrent(3), which is not thread safe, iterateGroups
// can not bet used concurrently. If concurrent usage is required, it is
// recommended to use locking mechanism such as sync.Mutex when calling
// iterateGroups from multiple goroutines.
func iterateGroups(fn NextGroupFunc) error {
	groupIterator.set()
	defer groupIterator.end()
	for {
		var result *C.struct_group
		C.resetErrno()
		result, err := groupIterator.get()

		// If result is nil - getgrent iterated through entire groups database or there was an error
		if result == nil {
			return err
		}

		if err = fn(buildGroup(result)); err != nil {
			// User provided non-nil error means that iteration should be terminated
			return err
		}
	}
}
